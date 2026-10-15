#!/usr/bin/env python3
"""Generate a deterministic, non-private VLM evaluation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

WIDTH = 160
HEIGHT = 120
BASE_QUESTION = (
    "只根据这些帧中直接可见的内容回答。observations 只能写可见事实，"
    "inferences 必须明确表达不确定性；不要把抽象色块解释为人物、宝宝、成人、宠物或风险。"
    "多帧完全相同时不得声称发生移动；证据不足时 event_type=unknown、risk_level=none。"
)


def _solid(color: tuple[int, int, int]) -> bytes:
    return bytes(color) * (WIDTH * HEIGHT)


def _scene(
    *,
    background: tuple[int, int, int] = (238, 238, 238),
    rectangles: Sequence[tuple[int, int, int, int, tuple[int, int, int]]] = (),
) -> bytes:
    pixels = bytearray(_solid(background))
    for x1, y1, x2, y2, color in rectangles:
        if not 0 <= x1 < x2 <= WIDTH or not 0 <= y1 < y2 <= HEIGHT:
            raise ValueError("rectangle is outside the synthetic frame")
        row = bytes(color) * (x2 - x1)
        for y in range(y1, y2):
            offset = (y * WIDTH + x1) * 3
            pixels[offset : offset + len(row)] = row
    return bytes(pixels)


def _write_ppm(path: Path, pixels: bytes) -> None:
    expected = WIDTH * HEIGHT * 3
    if len(pixels) != expected:
        raise ValueError("synthetic frame has an invalid byte count")
    path.write_bytes(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + pixels)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected(
    *,
    event_type: str = "unknown",
    subjects: Sequence[str] = (),
    observations: Sequence[str] = (),
    inferences: Sequence[str] = (),
    observation_groups: Sequence[Sequence[str]] = (),
    inference_groups: Sequence[Sequence[str]] = (),
) -> dict[str, object]:
    allowed_observations = tuple(
        dict.fromkeys([*observations, *(item for group in observation_groups for item in group)])
    )
    allowed_inferences = tuple(
        dict.fromkeys([*inferences, *(item for group in inference_groups for item in group)])
    )
    return {
        "event_type": event_type,
        "risk_level": "none",
        "allowed_subjects": list(subjects),
        "allowed_observations": list(allowed_observations),
        "allowed_inferences": list(allowed_inferences),
        "required_observation_groups": [list(group) for group in observation_groups],
        "required_inference_groups": [list(group) for group in inference_groups],
    }


def _case(
    *,
    case_id: str,
    paths: Sequence[Path],
    root: Path,
    expected: dict[str, object],
    question: str = BASE_QUESTION,
) -> dict[str, object]:
    return {
        "version": 2,
        "case_id": case_id,
        "frames": [
            {
                "frame_id": f"f{index:02d}",
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for index, path in enumerate(paths, start=1)
        ],
        "question": question,
        "expected": expected,
    }


def _generate_synthetic_frames(root: Path) -> list[dict[str, object]]:
    frames = root / "frames"
    frames.mkdir()

    def write(name: str, pixels: bytes) -> Path:
        path = frames / f"{name}.ppm"
        _write_ppm(path, pixels)
        return path

    cases: list[dict[str, object]] = []
    for name, color in (
        ("blank_black", (0, 0, 0)),
        ("blank_gray", (96, 96, 96)),
        ("blank_white", (255, 255, 255)),
    ):
        path = write(name, _solid(color))
        cases.append(case_id_case(name, [path], root, _expected()))

    red = (210, 30, 30)
    blue = (25, 80, 210)
    green = (30, 170, 70)
    gray = (120, 120, 120)
    red_fact = (
        "白色背景上可见一个红色矩形。",
        "a red rectangle is visible on a light background",
    )
    two_fact = (
        "浅色背景上可见一个红色矩形和一个蓝色矩形。",
        "a red rectangle and a blue rectangle are visible on a light background",
    )
    occlusion_fact = (
        "一个蓝色矩形的一部分被灰色矩形遮挡。",
        "part of a blue rectangle is covered by a gray rectangle",
    )

    red_path = write(
        "single_red_rectangle",
        _scene(rectangles=((45, 30, 115, 90, red),)),
    )
    cases.append(
        case_id_case(
            "single_red_rectangle",
            [red_path],
            root,
            _expected(observation_groups=(red_fact,)),
        )
    )

    two_path = write(
        "two_rectangles",
        _scene(rectangles=((20, 30, 65, 90, red), (95, 30, 140, 90, blue))),
    )
    cases.append(
        case_id_case(
            "two_rectangles",
            [two_path],
            root,
            _expected(
                observations=("red rectangle", "blue rectangle"),
                observation_groups=(
                    ("red rectangle", *two_fact),
                    ("blue rectangle", *two_fact),
                ),
            ),
        )
    )

    occluded_path = write(
        "partial_occlusion",
        _scene(rectangles=((30, 25, 130, 95, blue), (80, 0, 105, 120, gray))),
    )
    cases.append(
        case_id_case(
            "partial_occlusion",
            [occluded_path],
            root,
            _expected(
                observations=("blue rectangle", "gray vertical bar"),
                observation_groups=(
                    ("blue rectangle",),
                    ("gray vertical bar",),
                    occlusion_fact,
                ),
            ),
        )
    )

    static_pixels = _scene(rectangles=((50, 35, 110, 85, green),))
    static_paths = [write(f"static_sequence_{index}", static_pixels) for index in range(1, 4)]
    cases.append(
        case_id_case(
            "static_sequence",
            static_paths,
            root,
            _expected(
                observation_groups=(
                    (
                        "三帧中绿色矩形位于相同位置。",
                        "the green rectangle is in the same position in all three frames",
                    ),
                )
            ),
        )
    )

    movement_paths = [
        write(
            f"position_change_{index}",
            _scene(rectangles=((x, 40, x + 35, 80, red),)),
        )
        for index, x in enumerate((15, 62, 110), start=1)
    ]
    cases.append(
        case_id_case(
            "position_change_sequence",
            movement_paths,
            root,
            _expected(
                event_type="movement",
                observation_groups=(
                    (
                        "红色矩形在三帧中的水平位置不同。",
                        "the red rectangle appears at different horizontal "
                        "positions across the frames",
                    ),
                ),
                inference_groups=(
                    (
                        "红色矩形可能在帧之间发生了移动。",
                        "the red rectangle may have moved between frames",
                    ),
                ),
            ),
        )
    )

    disappearance_paths = [
        write("disappearance_1", _scene(rectangles=((55, 35, 105, 85, blue),))),
        write("disappearance_2", _scene(rectangles=((80, 35, 130, 85, blue),))),
        write("disappearance_3", _scene()),
    ]
    cases.append(
        case_id_case(
            "disappearance_sequence",
            disappearance_paths,
            root,
            _expected(
                observation_groups=(
                    (
                        "蓝色矩形在最后一帧中不可见。",
                        "the blue rectangle is not visible in the final frame",
                    ),
                ),
                inferences=(
                    "蓝色矩形可能移出了画面。",
                    "the blue rectangle may have moved out of view",
                ),
            ),
        )
    )

    exposure_paths = [
        write(
            f"exposure_change_{index}",
            _scene(
                background=(background, background, background),
                rectangles=((50, 35, 110, 85, green),),
            ),
        )
        for index, background in enumerate((40, 130, 230), start=1)
    ]
    cases.append(
        case_id_case(
            "exposure_change_sequence",
            exposure_paths,
            root,
            _expected(
                observation_groups=(
                    (
                        "三帧的背景亮度不同，绿色矩形均可见。",
                        "the background brightness differs while a green rectangle remains visible",
                    ),
                )
            ),
        )
    )

    appearance_paths = [
        write("appearance_1", _scene()),
        write("appearance_2", _scene(rectangles=((55, 35, 105, 85, red),))),
        write("appearance_3", _scene(rectangles=((55, 35, 105, 85, red),))),
    ]
    cases.append(
        case_id_case(
            "appearance_sequence",
            appearance_paths,
            root,
            _expected(
                inferences=(
                    "红色矩形可能进入了画面。",
                    "the red rectangle may have entered the view",
                ),
                observation_groups=(
                    (
                        "第一帧中没有红色矩形，后两帧中可见红色矩形。",
                        "the red rectangle is absent in the first frame and visible "
                        "in the later frames",
                    ),
                ),
            ),
        )
    )

    count_change_paths = [
        write("count_change_1", _scene(rectangles=((25, 35, 70, 85, red),))),
        write(
            "count_change_2",
            _scene(rectangles=((25, 35, 70, 85, red), (95, 35, 140, 85, red))),
        ),
        write(
            "count_change_3",
            _scene(rectangles=((25, 35, 70, 85, red), (95, 35, 140, 85, red))),
        ),
    ]
    cases.append(
        case_id_case(
            "count_change_sequence",
            count_change_paths,
            root,
            _expected(
                observation_groups=(
                    (
                        "第一帧有一个红色矩形，后两帧有两个红色矩形。",
                        "one red rectangle is visible in the first frame "
                        "and two in the later frames",
                    ),
                )
            ),
        )
    )

    size_change_paths = [
        write(
            f"size_change_{index}",
            _scene(
                rectangles=(
                    (
                        80 - size // 2,
                        60 - size // 2,
                        80 + size // 2,
                        60 + size // 2,
                        blue,
                    ),
                )
            ),
        )
        for index, size in enumerate((24, 48, 80), start=1)
    ]
    cases.append(
        case_id_case(
            "size_change_sequence",
            size_change_paths,
            root,
            _expected(
                inferences=(
                    "蓝色矩形可能正在靠近，或者画面发生了缩放。",
                    "the blue rectangle may be closer or the view may have zoomed",
                ),
                observation_groups=(
                    (
                        "蓝色矩形在三帧中逐渐变大。",
                        "the blue rectangle becomes larger across the three frames",
                    ),
                ),
            ),
        )
    )

    occlusion_progression_paths = [
        write(
            "occlusion_progression_1",
            _scene(rectangles=((30, 25, 130, 95, blue),)),
        ),
        write(
            "occlusion_progression_2",
            _scene(rectangles=((30, 25, 130, 95, blue), (75, 0, 90, 120, gray))),
        ),
        write(
            "occlusion_progression_3",
            _scene(rectangles=((30, 25, 130, 95, blue), (60, 0, 110, 120, gray))),
        ),
    ]
    cases.append(
        case_id_case(
            "occlusion_progression_sequence",
            occlusion_progression_paths,
            root,
            _expected(
                observation_groups=(
                    (
                        "灰色遮挡区域逐帧变宽，蓝色矩形的可见区域逐帧变小。",
                        "the gray occluding area becomes wider and less of the blue "
                        "rectangle remains visible",
                    ),
                )
            ),
        )
    )
    return cases


def case_id_case(
    case_id: str,
    paths: Sequence[Path],
    root: Path,
    expected: dict[str, object],
) -> dict[str, object]:
    return _case(case_id=case_id, paths=paths, root=root, expected=expected)


def _add_public_image_cases(
    root: Path,
    public_image: Path,
) -> list[dict[str, object]]:
    if (
        public_image.is_symlink()
        or not public_image.is_file()
        or not 0 < public_image.stat().st_size <= 50 * 1024 * 1024
    ):
        raise ValueError("public image must be a bounded regular file, not a symlink")
    frames = root / "frames"
    suffix = public_image.suffix.lower()
    if not suffix or len(suffix) > 10:
        raise ValueError("public image suffix is invalid")
    single = frames / f"public_bus_single{suffix}"
    shutil.copyfile(public_image, single)
    static_paths = []
    for index in range(1, 4):
        path = frames / f"public_bus_static_{index}{suffix}"
        shutil.copyfile(public_image, path)
        static_paths.append(path)
    bus_facts = (
        "画面中可见一辆蓝色公交车。",
        "a blue bus is present on the street",
        "a blue bus with text and a logo is present on the street",
        "a blue bus with text and logos is present on the street",
        "a blue and white bus",
        "a blue public bus with text and a logo is present",
    )
    person_facts = (
        "画面中可见人物。",
        "two individuals are standing near the bus",
        "two people standing near the bus",
        "two people are standing near the bus",
        "people are visible near the bus",
    )
    optional_facts = ("other people walking", "other people are walking in the area")
    question = BASE_QUESTION + "普通 person 不得被写成 baby 或 adult。"
    return [
        _case(
            case_id="public_bus_single",
            paths=[single],
            root=root,
            question=question,
            expected=_expected(
                subjects=("person",),
                observations=optional_facts,
                observation_groups=(bus_facts, person_facts),
            ),
        ),
        _case(
            case_id="public_bus_static_sequence",
            paths=static_paths,
            root=root,
            question=question + "三帧完全相同，不得据此声称人物正在行走。",
            expected=_expected(
                subjects=("person",),
                observations=optional_facts,
                observation_groups=(bus_facts, person_facts),
            ),
        ),
    ]


def generate(output_dir: Path, *, public_image: Path | None = None) -> tuple[int, int]:
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing to overwrite")
    output_dir.mkdir(parents=True)
    try:
        cases = _generate_synthetic_frames(output_dir)
        if public_image is not None:
            cases.extend(_add_public_image_cases(output_dir, public_image))
        dataset = output_dir / "dataset.jsonl"
        dataset.write_text(
            "".join(
                json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases
            ),
            encoding="utf-8",
            newline="\n",
        )
        frame_count = sum(len(case["frames"]) for case in cases)  # type: ignore[arg-type]
        return len(cases), frame_count
    except BaseException:
        # The caller supplied a new, previously absent target. Remove only this
        # partial output so a failed generation is never mistaken for a dataset.
        shutil.rmtree(output_dir)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--public-image", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases, frames = generate(args.output_dir, public_image=args.public_image)
    print(
        json.dumps(
            {
                "version": 1,
                "case_versions": [2],
                "cases": cases,
                "frames": frames,
                "private_media_used": False,
                "paths_disclosed": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
