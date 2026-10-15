from __future__ import annotations

import base64
import hashlib
import hmac
import math
import os
import re
import secrets
import sqlite3
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.analysis_queue import (
    AnalysisJob,
    AnalysisJobStatus,
    AnalysisQueue,
    AnalysisQueueError,
)
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)
from baby_monitor.visual_predicate_review import (
    VisualPredicateReviewError,
    VisualPredicateReviewReason,
    VisualPredicateReviewState,
    assess_visual_predicate_review,
)
from baby_monitor.visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
    VisualPredicateShadowObservation,
    VisualPredicateShadowQueueError,
)

_REVIEW_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_CLIP_BYTES = 64 * 1024 * 1024
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
}
_HUMAN_ANSWERS = {
    VisualPredicate.FACE_VISIBILITY: frozenset(
        {
            VisualPredicateAnswer.CLEAR,
            VisualPredicateAnswer.PARTIALLY_OBSCURED,
            VisualPredicateAnswer.NOT_VISIBLE,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
    VisualPredicate.SUPPORT_CONTACT_VISIBLE: frozenset(
        {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
    VisualPredicate.POSTURE_TRANSITION_VISIBLE: frozenset(
        {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
    VisualPredicate.COVERING_NEAR_FACE_VISIBLE: frozenset(
        {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
}
_REVIEW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS visual_predicate_human_reviews (
    review_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT NOT NULL UNIQUE,
    event_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    result_binding_sha256 TEXT NOT NULL,
    model_answer TEXT NOT NULL,
    model_confidence REAL NOT NULL CHECK (
        model_confidence >= 0.0 AND model_confidence <= 1.0
    ),
    human_answer TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES analysis_jobs(event_id)
);

CREATE INDEX IF NOT EXISTS idx_visual_predicate_human_reviews_latest
    ON visual_predicate_human_reviews (
        event_id, result_binding_sha256, predicate, review_sequence
    );
"""
_REVIEW_REASON_CODES = frozenset(
    {
        VisualPredicateReviewReason.FACE_NOT_VISIBLE,
        VisualPredicateReviewReason.FACE_PARTIALLY_OBSCURED,
        VisualPredicateReviewReason.COVERING_NEAR_FACE_VISIBLE,
    }
)


class HomeShadowReviewError(RuntimeError):
    """Path-free local shadow review failure."""


@dataclass(frozen=True, slots=True)
class HomeShadowReviewAnswer:
    predicate: str
    model_answer: str
    model_confidence: float
    human_answer: str | None
    allowed_human_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            predicate = VisualPredicate(self.predicate)
            model_answer = VisualPredicateAnswer(self.model_answer)
            human_answer = (
                VisualPredicateAnswer(self.human_answer) if self.human_answer is not None else None
            )
            allowed = _HUMAN_ANSWERS[predicate]
        except (KeyError, ValueError):
            raise HomeShadowReviewError("shadow review answer is invalid") from None
        if (
            model_answer not in allowed
            or type(self.model_confidence) not in {int, float}
            or not math.isfinite(self.model_confidence)
            or not 0.0 <= float(self.model_confidence) <= 1.0
            or human_answer is not None
            and human_answer not in allowed
            or tuple(item.value for item in sorted(allowed, key=lambda item: item.value))
            != self.allowed_human_answers
        ):
            raise HomeShadowReviewError("shadow review answer is invalid")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "predicate": self.predicate,
            "model_answer": self.model_answer,
            "model_confidence": self.model_confidence,
            "human_answer": self.human_answer,
            "allowed_human_answers": list(self.allowed_human_answers),
        }


@dataclass(frozen=True, slots=True)
class HomeShadowReviewItem:
    review_token: str
    recorded_at: str
    gate_source: str
    review_state: str
    review_reasons: tuple[str, ...]
    manual_review_recommended: bool
    answers: tuple[HomeShadowReviewAnswer, ...]
    frame_count: int
    clip_available: bool

    def __post_init__(self) -> None:
        try:
            review_state = VisualPredicateReviewState(self.review_state)
            review_reasons = tuple(
                VisualPredicateReviewReason(item) for item in self.review_reasons
            )
        except (TypeError, ValueError):
            raise HomeShadowReviewError("shadow review item is invalid") from None
        has_review_reason = any(item in _REVIEW_REASON_CODES for item in review_reasons)
        valid_review = (
            len(set(review_reasons)) == len(review_reasons)
            and type(self.manual_review_recommended) is bool
            and (
                review_state is VisualPredicateReviewState.REVIEW_RECOMMENDED
                and has_review_reason
                and self.manual_review_recommended
                or review_state is VisualPredicateReviewState.UNKNOWN
                and review_reasons
                and not has_review_reason
                and self.manual_review_recommended
                or review_state is VisualPredicateReviewState.NO_RULE_MATCH
                and not review_reasons
                and not self.manual_review_recommended
            )
        )
        if (
            _REVIEW_TOKEN.fullmatch(self.review_token) is None
            or self.gate_source
            not in {
                "hardware_signal",
                "person_detection",
                "hardware_and_person",
                "periodic_sample",
            }
            or not isinstance(self.answers, tuple)
            or not self.answers
            or any(not isinstance(item, HomeShadowReviewAnswer) for item in self.answers)
            or type(self.frame_count) is not int
            or not 1 <= self.frame_count <= 8
            or type(self.clip_available) is not bool
            or not valid_review
        ):
            raise HomeShadowReviewError("shadow review item is invalid")
        try:
            parsed = datetime.fromisoformat(self.recorded_at)
        except ValueError:
            raise HomeShadowReviewError("shadow review item is invalid") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise HomeShadowReviewError("shadow review item is invalid")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "review_token": self.review_token,
            "recorded_at": self.recorded_at,
            "gate_source": self.gate_source,
            "review_state": self.review_state,
            "review_reasons": list(self.review_reasons),
            "manual_review_recommended": self.manual_review_recommended,
            "answers": [item.to_public_dict() for item in self.answers],
            "frame_count": self.frame_count,
            "clip_available": self.clip_available,
            "paths_disclosed": False,
            "event_identity_disclosed": False,
            "risk_inference": False,
            "notification_decision": False,
        }


@dataclass(frozen=True, slots=True)
class HomeShadowReviewMedia:
    content_type: str
    payload: bytes = field(repr=False)
    media_kind: str

    def __post_init__(self) -> None:
        maximum = _MAX_IMAGE_BYTES if self.media_kind == "frame" else _MAX_CLIP_BYTES
        if (
            self.content_type not in set(_MEDIA_TYPES.values())
            or not isinstance(self.payload, bytes)
            or not 0 < len(self.payload) <= maximum
            or self.media_kind not in {"frame", "clip"}
        ):
            raise HomeShadowReviewError("shadow review media is invalid")


@dataclass(frozen=True, slots=True)
class HomeShadowReviewReceipt:
    predicate: str
    human_answer: str
    reviewed_at: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "predicate": self.predicate,
            "human_answer": self.human_answer,
            "reviewed_at": self.reviewed_at,
            "paths_disclosed": False,
            "event_identity_disclosed": False,
            "risk_inference": False,
            "notification_decision": False,
        }


def _admit_database(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    parent = candidate.parent
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        parent_metadata = parent.lstat()
        resolved_parent = parent.resolve(strict=True)
    except OSError:
        raise HomeShadowReviewError("shadow review database is unavailable") from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or resolved != candidate
        or parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
        or resolved_parent != parent
    ):
        raise HomeShadowReviewError("shadow review database is unavailable")
    return resolved


class HomeShadowReviewRepository:
    def __init__(self, database_path: str | Path, *, token_secret: bytes):
        self._database_path = _admit_database(database_path)
        if not isinstance(token_secret, bytes) or len(token_secret) != 32:
            raise HomeShadowReviewError("shadow review token secret is invalid")
        self._token_secret = token_secret

    def __repr__(self) -> str:
        return "HomeShadowReviewRepository(private=True)"

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        self._database_path = _admit_database(self._database_path)
        mode = "ro" if read_only else "rw"
        try:
            connection = sqlite3.connect(
                f"{self._database_path.as_uri()}?mode={mode}",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            if read_only:
                connection.execute("PRAGMA query_only = ON")
            return connection
        except sqlite3.Error:
            raise HomeShadowReviewError("shadow review database is unavailable") from None

    def _review_token(self, observation: VisualPredicateShadowObservation) -> str:
        digest = observation.attempt.result_binding_sha256
        if digest is None or _SHA256.fullmatch(digest) is None:
            raise HomeShadowReviewError("shadow review observation is invalid")
        message = (
            "home-shadow-review-v1\x00" + observation.attempt.event_id + "\x00" + digest
        ).encode("utf-8")
        raw = hmac.new(self._token_secret, message, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _completed_observations(
        self,
        *,
        limit: int,
    ) -> tuple[VisualPredicateShadowObservation, ...]:
        try:
            observations = VisualPredicateShadowAttemptQueue(
                self._database_path
            ).recent_observations_read_only(limit=limit)
        except VisualPredicateShadowQueueError:
            raise HomeShadowReviewError("shadow review database is unavailable") from None
        return tuple(item for item in observations if item.results)

    def _analysis_job(self, event_id: str) -> AnalysisJob | None:
        try:
            return AnalysisQueue(self._database_path).get_read_only(event_id)
        except AnalysisQueueError:
            raise HomeShadowReviewError("shadow review database is unavailable") from None

    @staticmethod
    def _gate_source(observation: VisualPredicateShadowObservation) -> str:
        hardware = bool(observation.attempt.gate.hardware_signal_count)
        person = bool(observation.attempt.gate.person_detection_count)
        if hardware and person:
            return "hardware_and_person"
        if hardware:
            return "hardware_signal"
        if person:
            return "person_detection"
        return "periodic_sample"

    def _latest_labels(
        self,
        *,
        event_id: str,
        binding_sha256: str,
    ) -> dict[str, str]:
        connection = self._connect(read_only=True)
        try:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'visual_predicate_human_reviews'
                """
            ).fetchone()
            if table is None:
                return {}
            rows = connection.execute(
                """
                SELECT predicate, human_answer
                FROM visual_predicate_human_reviews
                WHERE event_id = ? AND result_binding_sha256 = ?
                ORDER BY review_sequence DESC
                """,
                (event_id, binding_sha256),
            ).fetchall()
        except sqlite3.Error:
            raise HomeShadowReviewError("shadow review database is unavailable") from None
        finally:
            connection.close()
        latest: dict[str, str] = {}
        for row in rows:
            predicate = str(row["predicate"])
            if predicate not in latest:
                latest[predicate] = str(row["human_answer"])
        return latest

    def list_items(self, *, limit: int = 8) -> tuple[HomeShadowReviewItem, ...]:
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("review limit must be between 1 and 32")
        items: list[HomeShadowReviewItem] = []
        ranked: list[
            tuple[
                int,
                int,
                VisualPredicateShadowObservation,
                VisualPredicateReviewState,
                tuple[VisualPredicateReviewReason, ...],
                bool,
            ]
        ] = []
        priority = {
            VisualPredicateReviewState.REVIEW_RECOMMENDED: 0,
            VisualPredicateReviewState.UNKNOWN: 1,
            VisualPredicateReviewState.NO_RULE_MATCH: 2,
        }
        for index, observation in enumerate(self._completed_observations(limit=1_000)):
            try:
                recommendation = assess_visual_predicate_review(observation)
            except VisualPredicateReviewError:
                raise HomeShadowReviewError("shadow review observation is invalid") from None
            ranked.append(
                (
                    priority[recommendation.state],
                    index,
                    observation,
                    recommendation.state,
                    recommendation.reasons,
                    recommendation.manual_review_recommended,
                )
            )
        for _, _, observation, review_state, review_reasons, manual_review in sorted(ranked):
            job = self._analysis_job(observation.attempt.event_id)
            if job is None or job.status not in {
                AnalysisJobStatus.PENDING,
                AnalysisJobStatus.COMPLETED,
            }:
                continue
            digest = observation.attempt.result_binding_sha256
            assert digest is not None
            labels = self._latest_labels(
                event_id=observation.attempt.event_id,
                binding_sha256=digest,
            )
            answers = tuple(
                self._public_answer(result, labels.get(result.predicate.value))
                for result in observation.results
            )
            assert observation.recorded_at is not None
            items.append(
                HomeShadowReviewItem(
                    review_token=self._review_token(observation),
                    recorded_at=observation.recorded_at.astimezone(UTC).isoformat(),
                    gate_source=self._gate_source(observation),
                    review_state=review_state.value,
                    review_reasons=tuple(item.value for item in review_reasons),
                    manual_review_recommended=manual_review,
                    answers=answers,
                    frame_count=len(job.media.frame_paths),
                    clip_available=bool(job.media.clip_path),
                )
            )
            if len(items) >= limit:
                break
        return tuple(items)

    def calibration_report(self, *, limit: int = 1_000) -> dict[str, object]:
        if type(limit) is not int or not 1 <= limit <= 1_000:
            raise ValueError("calibration limit must be between 1 and 1000")
        observations = self._completed_observations(limit=limit)
        groups: dict[tuple[str, str, str, str], dict[str, object]] = {}
        labeled_observation_count = 0
        answer_count = 0
        labeled_answer_count = 0
        exact_agreement_count = 0
        for observation in observations:
            digest = observation.attempt.result_binding_sha256
            assert digest is not None
            labels = self._latest_labels(
                event_id=observation.attempt.event_id,
                binding_sha256=digest,
            )
            observation_labeled = False
            for result in observation.results:
                answer_count += 1
                key = (
                    result.model_id,
                    result.model_revision,
                    result.prompt_version,
                    result.predicate.value,
                )
                group = groups.setdefault(
                    key,
                    {
                        "answer_count": 0,
                        "labeled_answer_count": 0,
                        "exact_agreement_count": 0,
                        "human_unknown_count": 0,
                        "model_unknown_count": 0,
                        "confusion": {},
                    },
                )
                group["answer_count"] = int(group["answer_count"]) + 1
                if result.answer is VisualPredicateAnswer.UNKNOWN:
                    group["model_unknown_count"] = int(group["model_unknown_count"]) + 1
                human_answer = labels.get(result.predicate.value)
                if human_answer is None:
                    continue
                allowed_human_answers = {item.value for item in _HUMAN_ANSWERS[result.predicate]}
                if human_answer not in allowed_human_answers:
                    raise HomeShadowReviewError("shadow review label is invalid")
                observation_labeled = True
                labeled_answer_count += 1
                group["labeled_answer_count"] = int(group["labeled_answer_count"]) + 1
                if human_answer == VisualPredicateAnswer.UNKNOWN.value:
                    group["human_unknown_count"] = int(group["human_unknown_count"]) + 1
                if human_answer == result.answer.value:
                    exact_agreement_count += 1
                    group["exact_agreement_count"] = int(group["exact_agreement_count"]) + 1
                confusion = group["confusion"]
                assert isinstance(confusion, dict)
                cell = (result.answer.value, human_answer)
                confusion[cell] = int(confusion.get(cell, 0)) + 1
            if observation_labeled:
                labeled_observation_count += 1

        public_groups: list[dict[str, object]] = []
        for key in sorted(groups):
            group = groups[key]
            labeled = int(group["labeled_answer_count"])
            exact = int(group["exact_agreement_count"])
            confusion = group["confusion"]
            assert isinstance(confusion, dict)
            public_groups.append(
                {
                    "model_id": key[0],
                    "model_revision": key[1],
                    "prompt_version": key[2],
                    "predicate": key[3],
                    "answer_count": int(group["answer_count"]),
                    "labeled_answer_count": labeled,
                    "unlabeled_answer_count": int(group["answer_count"]) - labeled,
                    "exact_agreement_count": exact,
                    "exact_agreement_rate": (None if labeled == 0 else round(exact / labeled, 12)),
                    "human_unknown_count": int(group["human_unknown_count"]),
                    "model_unknown_count": int(group["model_unknown_count"]),
                    "confusion": [
                        {
                            "model_answer": model_answer,
                            "human_answer": human_answer,
                            "count": count,
                        }
                        for (model_answer, human_answer), count in sorted(confusion.items())
                    ],
                }
            )
        return {
            "version": 1,
            "observation_count": len(observations),
            "labeled_observation_count": labeled_observation_count,
            "answer_count": answer_count,
            "labeled_answer_count": labeled_answer_count,
            "unlabeled_answer_count": answer_count - labeled_answer_count,
            "exact_agreement_count": exact_agreement_count,
            "exact_agreement_rate": (
                None
                if labeled_answer_count == 0
                else round(exact_agreement_count / labeled_answer_count, 12)
            ),
            "groups": public_groups,
            "human_reviewed_visible_fact_labels": True,
            "exact_answer_agreement_evaluated": labeled_answer_count > 0,
            "risk_accuracy_evaluated": False,
            "media_read": False,
            "paths_disclosed": False,
            "event_identity_disclosed": False,
            "subject_identity_inference": False,
            "notification_decision": False,
        }

    @staticmethod
    def _public_answer(
        result: VisualPredicateResult,
        human_answer: str | None,
    ) -> HomeShadowReviewAnswer:
        allowed = tuple(
            item.value
            for item in sorted(
                _HUMAN_ANSWERS[result.predicate],
                key=lambda item: item.value,
            )
        )
        return HomeShadowReviewAnswer(
            predicate=result.predicate.value,
            model_answer=result.answer.value,
            model_confidence=result.confidence,
            human_answer=human_answer,
            allowed_human_answers=allowed,
        )

    def _resolve(
        self,
        review_token: str,
    ) -> tuple[VisualPredicateShadowObservation, AnalysisJob]:
        if not isinstance(review_token, str) or _REVIEW_TOKEN.fullmatch(review_token) is None:
            raise HomeShadowReviewError("shadow review token is invalid")
        for observation in self._completed_observations(limit=1_000):
            if not secrets.compare_digest(
                self._review_token(observation),
                review_token,
            ):
                continue
            job = self._analysis_job(observation.attempt.event_id)
            if job is None or job.status not in {
                AnalysisJobStatus.PENDING,
                AnalysisJobStatus.COMPLETED,
            }:
                break
            return observation, job
        raise HomeShadowReviewError("shadow review token is invalid")

    @staticmethod
    def _read_private_media(
        path: str,
        *,
        storage_root: str,
        media_kind: str,
    ) -> HomeShadowReviewMedia:
        candidate = Path(path)
        root = Path(storage_root)
        maximum = _MAX_IMAGE_BYTES if media_kind == "frame" else _MAX_CLIP_BYTES
        try:
            root_metadata = root.lstat()
            resolved_root = root.resolve(strict=True)
            candidate_metadata = candidate.lstat()
            resolved_candidate = candidate.resolve(strict=True)
        except OSError:
            raise HomeShadowReviewError("shadow review media is unavailable") from None
        content_type = _MEDIA_TYPES.get(candidate.suffix.lower())
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) & 0o022
            or resolved_root != root
            or not candidate.is_absolute()
            or candidate.is_symlink()
            or not stat.S_ISREG(candidate_metadata.st_mode)
            or candidate_metadata.st_nlink != 1
            or candidate_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(candidate_metadata.st_mode) & 0o077
            or content_type is None
            or not resolved_candidate.is_relative_to(resolved_root)
        ):
            raise HomeShadowReviewError("shadow review media is unavailable")
        if media_kind == "frame" and not content_type.startswith("image/"):
            raise HomeShadowReviewError("shadow review media is unavailable")
        if media_kind == "clip" and not content_type.startswith("video/"):
            raise HomeShadowReviewError("shadow review media is unavailable")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError:
            raise HomeShadowReviewError("shadow review media is unavailable") from None
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_dev != candidate_metadata.st_dev
                or metadata.st_ino != candidate_metadata.st_ino
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
                or not 0 < metadata.st_size <= maximum
            ):
                raise HomeShadowReviewError("shadow review media is unavailable")
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise HomeShadowReviewError("shadow review media is unavailable")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise HomeShadowReviewError("shadow review media is unavailable")
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
        return HomeShadowReviewMedia(
            content_type=content_type,
            payload=payload,
            media_kind=media_kind,
        )

    def read_media(
        self,
        review_token: str,
        *,
        media_kind: str,
        index: int,
    ) -> HomeShadowReviewMedia:
        if media_kind not in {"frame", "clip"} or type(index) is not int or index < 0:
            raise HomeShadowReviewError("shadow review media request is invalid")
        _observation, job = self._resolve(review_token)
        if media_kind == "clip":
            if index != 0:
                raise HomeShadowReviewError("shadow review media request is invalid")
            selected = job.media.clip_path
        else:
            if index >= len(job.media.frame_paths):
                raise HomeShadowReviewError("shadow review media request is invalid")
            selected = job.media.frame_paths[index]
        return self._read_private_media(
            selected,
            storage_root=job.storage_root,
            media_kind=media_kind,
        )

    def record_label(
        self,
        review_token: str,
        *,
        predicate: str,
        human_answer: str,
        reviewed_at: datetime | None = None,
    ) -> HomeShadowReviewReceipt:
        observation, _job = self._resolve(review_token)
        try:
            parsed_predicate = VisualPredicate(predicate)
            parsed_answer = VisualPredicateAnswer(human_answer)
        except ValueError:
            raise HomeShadowReviewError("shadow review label is invalid") from None
        matching = tuple(item for item in observation.results if item.predicate is parsed_predicate)
        if len(matching) != 1 or parsed_answer not in _HUMAN_ANSWERS[parsed_predicate]:
            raise HomeShadowReviewError("shadow review label is invalid")
        result = matching[0]
        digest = observation.attempt.result_binding_sha256
        assert digest is not None
        if reviewed_at is not None and (
            reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None
        ):
            raise HomeShadowReviewError("shadow review label is invalid")
        timestamp = (reviewed_at or datetime.now(UTC)).astimezone(UTC)
        connection = self._connect(read_only=False)
        try:
            with connection:
                connection.executescript(_REVIEW_SCHEMA_SQL)
                connection.execute(
                    """
                    INSERT INTO visual_predicate_human_reviews (
                        review_id, event_id, predicate, result_binding_sha256,
                        model_answer, model_confidence, human_answer, reviewed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        secrets.token_hex(16),
                        observation.attempt.event_id,
                        parsed_predicate.value,
                        digest,
                        result.answer.value,
                        result.confidence,
                        parsed_answer.value,
                        timestamp.isoformat(),
                    ),
                )
        except sqlite3.Error:
            raise HomeShadowReviewError("shadow review database is unavailable") from None
        finally:
            connection.close()
        return HomeShadowReviewReceipt(
            predicate=parsed_predicate.value,
            human_answer=parsed_answer.value,
            reviewed_at=timestamp.isoformat(),
        )
