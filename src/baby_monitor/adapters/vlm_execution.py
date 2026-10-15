"""Fail-fast, per-user/per-device local VLM exclusion across POSIX processes.

Lock files persist: unlinking a live lock would allow a second inode and a
second model. Holding this gate does not reserve the GPU against other tools.
Requires a private POSIX user runtime directory; neither TMPDIR nor a service's
private /tmp namespace changes the shared execution gate.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from contextlib import contextmanager, suppress
from pathlib import Path


class VLMExecutionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(
            "local VLM execution is busy"
            if code == "local_vlm_busy"
            else "local VLM execution gate is unavailable or unsafe"
        )


def _identity(metadata):
    return metadata.st_dev, metadata.st_ino, metadata.st_uid, metadata.st_mode


def _execution_root() -> Path:
    """Use the stable user runtime root, never an environment-selected fallback."""
    return Path("/run/user") / str(os.getuid())


def _private(metadata, *, directory: bool):
    expected = 0o700 if directory else 0o600
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != expected
        or (not stat.S_ISDIR(metadata.st_mode) if directory else not stat.S_ISREG(metadata.st_mode))
        or (not directory and metadata.st_nlink != 1)
    ):
        raise VLMExecutionError("local_vlm_lock_unsafe")


@contextmanager
def local_vlm_execution(gpu_device: str):
    """Acquire a nonblocking gate and yield its inheritable-by-opt-in lease fd.

    Normally only this process holds the close-on-exec descriptor. A resident
    runner may explicitly pass it to its final model process: the flock then
    survives even an abrupt owner death until the last model holder exits.
    Never explicitly LOCK_UN a shared open-file description during cleanup.
    """
    if not isinstance(gpu_device, str) or re.fullmatch(r"[0-9]{1,3}", gpu_device) is None:
        raise VLMExecutionError("local_vlm_lock_unsafe")
    if os.name != "posix":
        raise VLMExecutionError("local_vlm_execution_unavailable")
    import fcntl

    descriptors = []
    descriptor = None
    try:
        root = _execution_root()
        if not root.is_absolute() or root.resolve() != root or root.is_symlink():
            raise VLMExecutionError("local_vlm_lock_unsafe")
        root_before = root.lstat()
        _private(root_before, directory=True)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        root_fd = os.open(root, directory_flags)
        descriptors.append(root_fd)
        _private(os.fstat(root_fd), directory=True)
        if _identity(root_before) != _identity(os.fstat(root_fd)):
            raise VLMExecutionError("local_vlm_lock_unsafe")
        name = "baby-monitor-vlm-execution"
        with suppress(FileExistsError):
            os.mkdir(name, mode=0o700, dir_fd=root_fd)
        before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        _private(before, directory=True)
        directory_fd = os.open(name, directory_flags, dir_fd=root_fd)
        descriptors.append(directory_fd)
        if _identity(before) != _identity(os.fstat(directory_fd)):
            raise VLMExecutionError("local_vlm_lock_unsafe")
        # Canonicalize numeric aliases such as 00, preventing two gates for GPU 0.
        filename = f"gpu_{int(gpu_device)}.lock"
        try:
            file_before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            file_before = None
        if file_before is not None:
            _private(file_before, directory=False)
        descriptor = os.open(
            filename,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600,
            dir_fd=directory_fd,
        )
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        _private(opened, directory=False)
        if file_before is not None and _identity(file_before) != _identity(opened):
            raise VLMExecutionError("local_vlm_lock_unsafe")
        if _identity(opened) != _identity(
            os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        ):
            raise VLMExecutionError("local_vlm_lock_unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise VLMExecutionError("local_vlm_busy") from None
            raise
        _private(os.fstat(descriptor), directory=False)
        if (
            _identity(opened) != _identity(os.fstat(descriptor))
            or _identity(opened)
            != _identity(os.stat(filename, dir_fd=directory_fd, follow_symlinks=False))
            or _identity(before) != _identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False))
            or _identity(root_before) != _identity(root.lstat())
        ):
            raise VLMExecutionError("local_vlm_lock_unsafe")
    except VLMExecutionError:
        raise
    except (OSError, ValueError):
        raise VLMExecutionError("local_vlm_lock_unsafe") from None
    else:
        yield descriptor
    finally:
        for item in reversed(descriptors):
            with suppress(OSError):
                os.close(item)
