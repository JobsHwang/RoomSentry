"""Dashboard application state: sessions, actions, and health aggregation."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field

from baby_monitor.home_dashboard.copy import (
    _ACTIONS,
    _PREDICATE_LABELS,
    _REVIEW_TOKEN,
)
from baby_monitor.home_dashboard.errors import HomeDashboardError
from baby_monitor.home_dashboard.ports import (
    ActionExecutor,
    DailyDigestCatalogProvider,
    DailyDigestExecutor,
    DailyDigestVideoProvider,
    DailyReportExecutor,
    DailyReportProvider,
    NotificationTestExecutor,
    ReviewCalibrationProvider,
    ReviewLabelRecorder,
    ReviewMediaProvider,
    ReviewProvider,
    StatusProvider,
)


@dataclass(slots=True)
class HomeDashboardApplication:
    status_provider: StatusProvider = field(repr=False)
    action_executor: ActionExecutor = field(repr=False)
    review_provider: ReviewProvider | None = field(default=None, repr=False)
    review_media_provider: ReviewMediaProvider | None = field(default=None, repr=False)
    review_label_recorder: ReviewLabelRecorder | None = field(default=None, repr=False)
    review_calibration_provider: ReviewCalibrationProvider | None = field(default=None, repr=False)
    notification_test_executor: NotificationTestExecutor | None = field(
        default=None,
        repr=False,
    )
    daily_report_executor: DailyReportExecutor | None = field(default=None, repr=False)
    daily_report_provider: DailyReportProvider | None = field(default=None, repr=False)
    daily_digest_catalog_provider: DailyDigestCatalogProvider | None = field(
        default=None,
        repr=False,
    )
    daily_digest_video_provider: DailyDigestVideoProvider | None = field(
        default=None,
        repr=False,
    )
    daily_digest_executor: DailyDigestExecutor | None = field(default=None, repr=False)
    control_session_token: str = field(
        default_factory=lambda: secrets.token_urlsafe(32),
        repr=False,
    )
    _control_nonces: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _review_nonces: dict[str, tuple[str, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _nonce_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not callable(self.status_provider) or not callable(self.action_executor):
            raise HomeDashboardError("home dashboard callbacks are invalid")
        review_callbacks = (
            self.review_provider,
            self.review_media_provider,
            self.review_label_recorder,
        )
        if any(item is not None for item in review_callbacks) and not all(
            callable(item) for item in review_callbacks
        ):
            raise HomeDashboardError("home dashboard review callbacks are invalid")
        if self.review_calibration_provider is not None and (
            not callable(self.review_calibration_provider)
            or not all(callable(item) for item in review_callbacks)
        ):
            raise HomeDashboardError("home dashboard calibration callback is invalid")
        if self.notification_test_executor is not None and not callable(
            self.notification_test_executor
        ):
            raise HomeDashboardError("home dashboard notification callback is invalid")
        daily_report_callbacks = (
            self.daily_report_executor,
            self.daily_report_provider,
        )
        if any(item is not None for item in daily_report_callbacks) and not all(
            callable(item) for item in daily_report_callbacks
        ):
            raise HomeDashboardError("home dashboard daily report callbacks are invalid")
        daily_digest_callbacks = (
            self.daily_digest_catalog_provider,
            self.daily_digest_video_provider,
        )
        if any(item is not None for item in daily_digest_callbacks) and not all(
            callable(item) for item in daily_digest_callbacks
        ):
            raise HomeDashboardError("home dashboard daily digest callbacks are invalid")
        if self.daily_digest_executor is not None and (
            not callable(self.daily_digest_executor)
            or not callable(self.daily_digest_catalog_provider)
            or not callable(self.daily_digest_video_provider)
        ):
            raise HomeDashboardError("home dashboard daily digest executor is invalid")
        if not isinstance(self.control_session_token, str) or not (
            32 <= len(self.control_session_token) <= 128
        ):
            raise HomeDashboardError("home dashboard session is invalid")

    @property
    def notification_test_enabled(self) -> bool:
        return callable(self.notification_test_executor)

    @property
    def daily_report_enabled(self) -> bool:
        return callable(self.daily_report_executor) and callable(self.daily_report_provider)

    @property
    def review_enabled(self) -> bool:
        return self.review_provider is not None

    @property
    def daily_digest_enabled(self) -> bool:
        return callable(self.daily_digest_catalog_provider) and callable(
            self.daily_digest_video_provider
        )

    @property
    def daily_digest_generation_enabled(self) -> bool:
        return callable(self.daily_digest_executor)

    def issue_review_nonces(
        self,
        items: tuple[dict[str, object], ...],
    ) -> dict[tuple[str, str], str]:
        result: dict[tuple[str, str], str] = {}
        with self._nonce_lock:
            if len(self._review_nonces) > 512:
                self._review_nonces.clear()
            for item in items:
                if not isinstance(item, dict):
                    continue
                token = item.get("review_token")
                answers = item.get("answers")
                if (
                    not isinstance(token, str)
                    or _REVIEW_TOKEN.fullmatch(token) is None
                    or not isinstance(answers, list)
                ):
                    continue
                for answer in answers:
                    predicate = answer.get("predicate") if isinstance(answer, dict) else None
                    if not isinstance(predicate, str) or predicate not in _PREDICATE_LABELS:
                        continue
                    nonce = secrets.token_urlsafe(24)
                    self._review_nonces[nonce] = (token, predicate)
                    result[(token, predicate)] = nonce
        return result

    def consume_review_nonce(self, review_token: str, predicate: str, nonce: str) -> bool:
        if not all(isinstance(item, str) for item in (review_token, predicate, nonce)):
            return False
        with self._nonce_lock:
            expected = self._review_nonces.pop(nonce, None)
        if expected is None:
            return False
        return secrets.compare_digest(expected[0], review_token) and secrets.compare_digest(
            expected[1],
            predicate,
        )

    def issue_control_nonces(self) -> dict[str, str]:
        with self._nonce_lock:
            if len(self._control_nonces) > 128:
                self._control_nonces.clear()
            result: dict[str, str] = {}
            for action in _ACTIONS:
                nonce = secrets.token_urlsafe(24)
                self._control_nonces[nonce] = action
                result[action] = nonce
            return result

    def consume_control_nonce(self, action: str, nonce: str) -> bool:
        if not isinstance(action, str) or not isinstance(nonce, str):
            return False
        with self._nonce_lock:
            expected_action = self._control_nonces.pop(nonce, None)
        return expected_action is not None and secrets.compare_digest(expected_action, action)
