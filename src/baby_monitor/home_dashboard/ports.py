"""Typed dependency-injection ports consumed by the dashboard application."""

from __future__ import annotations

from collections.abc import Callable

StatusProvider = Callable[[], dict[str, object]]


ActionExecutor = Callable[[str, bool], dict[str, object]]


ReviewProvider = Callable[[int], tuple[dict[str, object], ...]]


ReviewMediaProvider = Callable[[str, str, int], tuple[str, bytes]]


ReviewLabelRecorder = Callable[[str, str, str], dict[str, object]]


ReviewCalibrationProvider = Callable[[], dict[str, object]]


NotificationTestExecutor = Callable[[], dict[str, object]]


DailyReportExecutor = Callable[[], dict[str, object]]


DailyReportProvider = Callable[[], bytes]


DailyDigestCatalogProvider = Callable[[int], tuple[dict[str, object], ...]]


DailyDigestVideoProvider = Callable[[str], bytes]


DailyDigestExecutor = Callable[[], dict[str, object]]
