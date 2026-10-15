from __future__ import annotations

import http.client
import io
import json
import re
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock
from urllib.parse import urlencode

from baby_monitor.cli import main
from baby_monitor.home_dashboard import (
    HomeDashboardApplication,
    make_home_dashboard_server,
    render_home_dashboard,
)


def sampling_health_fixture() -> dict[str, object]:
    """Synthetic v4 sampling projection, reusable without media or live services."""
    return {
        "sampled_frame_count": 3,
        "zero_detection_sample_count": 2,
        "decoded_frame_count": 51,
        "timestamp_fallback_frame_count": 1,
        "max_adjacent_container_pts_gap_seconds": 0.125,
        "container_pts_regression_count": 1,
        "end_reason": "read_returned_false",
        "source_exhaustion_verified": False,
    }


def sampling_dashboard_status(
    *, state: str = "idle", freshness: str = "fresh"
) -> dict[str, object]:
    """A one-camera dashboard fixture with valid latest completed scan metadata."""
    status = _status()
    camera = status["candidate_health"]["cameras"][0]
    status["candidate_health"]["cameras"] = [camera]
    camera["freshness"] = freshness
    camera["operational"] = state == "idle" and freshness == "fresh"
    camera["health"].update(
        version=4,
        state=state,
        last_scan_completed_at="2026-09-12T12:00:00+00:00",
        last_sampling=sampling_health_fixture(),
        media_read=False,
    )
    return status


def _status() -> dict[str, object]:
    return {
        "version": 1,
        "base_operational": False,
        "manual_start_only": True,
        "components": [
            {
                "component": "xiaomi_stream_gateway",
                "state": "inactive",
                "enabled": False,
            },
            {
                "component": "ha_signal_buffer_dispatch",
                "state": "inactive",
                "enabled": False,
            },
        ],
        "ha_signal_outbox": {
            "status": "available",
            "pending": 7,
            "running": 0,
            "failed": 0,
            "failed_error_counts": {},
        },
        "analysis_queue": {
            "status": "available",
            "pending": 2,
            "running": 0,
            "completed": 11,
            "failed": 0,
            "missing_media_files": 0,
        },
        "candidate_health": {
            "status": "available",
            "attention_required": False,
            "cameras": [
                {
                    "freshness": "fresh",
                    "operational": True,
                    "history_summary": {
                        "scans_succeeded": 120,
                        "candidate_groups_emitted": 8,
                        "candidate_groups_suppressed": 5,
                        "execution_lock_wait_p95_upper_bound_seconds": 3.0,
                        "execution_lock_wait_max_seconds": 4.7,
                        "sessions_failed": 2,
                        "unclean_restarts": 0,
                    },
                    "health": {
                        "camera_id": "camera_01",
                        "state": "idle",
                        "last_execution_lock_wait_seconds": 1.25,
                        "last_scan_duration_seconds": 5.5,
                    },
                },
                {
                    "freshness": "fresh",
                    "operational": False,
                    "health": {
                        "camera_id": "camera_02",
                        "state": "stopped",
                        "last_execution_lock_wait_seconds": 2.0,
                        "last_scan_duration_seconds": 6.0,
                    },
                },
            ],
        },
        "candidate_soak": {
            "status": "available",
            "attention_required": False,
            "latest": {
                "state": "completed",
                "candidate_groups_emitted": 8,
                "candidate_groups_suppressed": 5,
                "elapsed_seconds": 86_405.0,
                "expected_duration_seconds": 86_400,
                "scans_succeeded": 240,
                "scans_failed": 0,
                "unclean_restarts": 0,
            },
        },
        "room_presence": {
            "version": 1,
            "status": "available",
            "rooms": [
                {
                    "room_id": "room_a",
                    "display_name": "宝宝房",
                    "status": "person_present",
                    "reason_codes": ["person_confirmed"],
                },
                {
                    "room_id": "room_b",
                    "display_name": "客厅",
                    "status": "coverage_gap",
                    "reason_codes": ["stream_coverage_gap"],
                },
            ],
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        },
        "visual_predicate_shadow": {
            "status": "available",
            "pending": 1,
            "running": 0,
            "completed": 3,
            "failed": 0,
            "skipped": 4,
            "review_policy_version": "atomic_visual_review_v1",
            "recent_assessed_count": 3,
            "recent_review_recommended": 1,
            "recent_unknown": 1,
            "recent_no_rule_match": 1,
            "recent_answers": [
                {
                    "hardware_signal_gate": True,
                    "person_detection_gate": False,
                    "answers": [
                        {
                            "predicate": "face_visibility",
                            "answer": "clear",
                            "confidence": 0.78,
                        },
                        {
                            "predicate": "covering_near_face_visible",
                            "answer": "no",
                            "confidence": 0.82,
                        },
                    ],
                }
            ],
        },
        "notification_queue": {
            "status": "available",
            "enabled": True,
            "external_delivery_enabled": False,
            "local_test_available": True,
            "pending": 1,
            "running": 0,
            "delivered": 2,
            "failed": 0,
        },
        "event_digest": {
            "status": "available",
            "local_date": "2026-08-18",
            "today_total": 2,
            "event_type_counts": {"movement": 1, "unknown": 1},
            "risk_level_counts": {"none": 1, "low": 1},
            "recent": [
                {
                    "local_time": "08-18 09:00:00",
                    "event_type": "movement",
                    "risk_level": "low",
                    "confidence": 0.7,
                    "needs_review": False,
                }
            ],
        },
        "daily_report": {
            "status": "available",
            "local_date": "2026-08-18",
            "event_count": 2,
            "available": True,
            "media_read": False,
            "paths_disclosed": False,
        },
        "daily_digest": {
            "status": "available",
            "plan_available": True,
            "video_available": True,
            "latest_local_date": "2026-08-11",
            "latest_selected_event_count": 4,
            "latest_duration_seconds": 32.0,
            "attention_required": False,
            "media_read": False,
            "paths_disclosed": False,
        },
        "private_path": "/private/must-not-render",
        "remote_upload_enabled": False,
    }


def _digest_catalog(token: str = "D" * 43) -> tuple[dict[str, object], ...]:
    return (
        {
            "review_token": token,
            "local_date": "2026-08-11",
            "selected_event_count": 1,
            "selected_duration_seconds": 8.0,
            "output_duration_seconds": 8.0,
            "target_met": False,
            "selected_camera_count": 1,
            "selected_hours": [9],
            "gap_reason_counts": {"buffer_started_after_event": 1},
            "items": [
                {
                    "rank": 1,
                    "local_time": "09:15:00",
                    "duration_seconds": 8.0,
                    "selection_score": 0.65,
                    "selection_reasons": ["event_recorded", "motion_fact"],
                }
            ],
            "facts_only": True,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        },
    )


class HomeDashboardTests(unittest.TestCase):
    def _server(
        self,
        *,
        action_executor: object | None = None,
        review_provider: object | None = None,
        review_media_provider: object | None = None,
        review_label_recorder: object | None = None,
        review_calibration_provider: object | None = None,
        notification_test_executor: object | None = None,
        daily_report_executor: object | None = None,
        daily_report_provider: object | None = None,
        daily_digest_catalog_provider: object | None = None,
        daily_digest_video_provider: object | None = None,
        daily_digest_executor: object | None = None,
    ) -> tuple[object, threading.Thread, list[tuple[str, bool]]]:
        actions: list[tuple[str, bool]] = []

        def execute(action: str, confirmed: bool) -> dict[str, object]:
            actions.append((action, confirmed))
            return _status()

        review_callbacks = (
            {
                "review_provider": review_provider,
                "review_media_provider": review_media_provider,
                "review_label_recorder": review_label_recorder,
                "review_calibration_provider": review_calibration_provider,
            }
            if review_provider is not None
            else {}
        )
        application = HomeDashboardApplication(
            status_provider=_status,
            action_executor=action_executor or execute,  # type: ignore[arg-type]
            notification_test_executor=notification_test_executor,  # type: ignore[arg-type]
            daily_report_executor=daily_report_executor,  # type: ignore[arg-type]
            daily_report_provider=daily_report_provider,  # type: ignore[arg-type]
            daily_digest_catalog_provider=daily_digest_catalog_provider,  # type: ignore[arg-type]
            daily_digest_video_provider=daily_digest_video_provider,  # type: ignore[arg-type]
            daily_digest_executor=daily_digest_executor,  # type: ignore[arg-type]
            **review_callbacks,  # type: ignore[arg-type]
        )
        server = make_home_dashboard_server(application, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop() -> None:
            server.shutdown()
            thread.join(2)
            server.server_close()

        self.addCleanup(stop)
        return server, thread, actions

    def _control_session(
        self,
        connection: http.client.HTTPConnection,
        server: object,
        action: str,
    ) -> tuple[str, str]:
        connection.request("GET", "/control")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        self.assertEqual(200, response.status)
        set_cookie = response.getheader("Set-Cookie", "")
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=Strict", set_cookie)
        cookie = set_cookie.split(";", 1)[0]
        matched = re.search(
            rf'name="nonce" value="([^"]+)"><input type="hidden" '
            rf'name="action" value="{re.escape(action)}">',
            body,
        )
        self.assertIsNotNone(matched)
        return cookie, matched.group(1)  # type: ignore[union-attr]

    def test_get_is_loopback_only_path_free_and_never_starts_monitoring(self) -> None:
        server, _thread, actions = self._server()
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        connection.request("GET", "/")
        response = connection.getresponse()
        body = response.read().decode("utf-8")

        self.assertEqual(200, response.status)
        self.assertIn("仅手动启动", body)
        self.assertIn("7 条待处理", body)
        self.assertIn("11 条已完成", body)
        self.assertIn("2 条待分析", body)
        self.assertIn("INT8 Mage-VL Shadow", body)
        self.assertIn("3 条已复核", body)
        self.assertIn("HA 硬件事件 · 脸部可见 清晰 78%", body)
        self.assertIn("建议人工回看 1 条 · 信息不足 1 条 · 未命中规则 1 条", body)
        self.assertIn("不代表危险或安全，也不会触发通知", body)
        self.assertIn("结果仅写 shadow，不进入风险判断和通知", body)
        self.assertIn("通知队列", body)
        self.assertIn("2 条已投递", body)
        self.assertIn("外部投递：关闭", body)
        self.assertIn("2 条事件", body)
        self.assertIn("隐私日报：可用", body)
        self.assertIn('href="/daily-report"', body)
        self.assertIn("每日精华", body)
        self.assertIn("4 个片段 ·\n32 秒", body)
        self.assertIn('href="/daily-digest"', body)
        self.assertIn("08-18 09:00:00 · 活动", body)
        self.assertNotIn("/private/must-not-render", body)
        self.assertIn('href="/control"', body)
        self.assertNotIn("<form", body)
        self.assertEqual([], actions)
        self.assertIsNone(response.getheader("Set-Cookie"))
        self.assertIn("default-src 'none'", response.getheader("Content-Security-Policy", ""))
        self.assertEqual("DENY", response.getheader("X-Frame-Options"))
        connection.close()

    def test_daily_digest_video_requires_session_and_supports_range(self) -> None:
        payload = b"synthetic-private-digest"
        calls: list[str] = []
        token = "D" * 43

        def provide(review_token: str) -> bytes:
            calls.append(review_token)
            return payload

        server, _thread, _actions = self._server(
            daily_digest_catalog_provider=lambda _limit: _digest_catalog(token),
            daily_digest_video_provider=provide,
        )
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        connection.request("GET", f"/daily-digest-video/{token}")
        denied = connection.getresponse()
        denied.read()
        self.assertEqual(403, denied.status)
        self.assertEqual([], calls)

        connection.request("GET", "/daily-digest")
        page = connection.getresponse()
        body = page.read().decode("utf-8")
        self.assertEqual(200, page.status)
        self.assertIn(f'src="/daily-digest-video/{token}"', body)
        self.assertIn("已记录事件", body)
        self.assertIn("检测到运动事实", body)
        self.assertIn("本地缓冲开始较晚 1 处", body)
        self.assertNotIn("camera_", body)
        self.assertNotIn("event_", body)
        cookie = page.getheader("Set-Cookie", "").split(";", 1)[0]
        self.assertTrue(cookie.startswith("bm_control="))

        connection.request("GET", f"/daily-digest/{'A' * 43}")
        missing = connection.getresponse()
        missing.read()
        self.assertEqual(404, missing.status)

        connection.request(
            "GET",
            f"/daily-digest-video/{token}",
            headers={"Cookie": cookie, "Range": "bytes=1-4"},
        )
        video = connection.getresponse()
        self.assertEqual(206, video.status)
        self.assertEqual("bytes 1-4/24", video.getheader("Content-Range"))
        self.assertEqual(payload[1:5], video.read())
        self.assertEqual([token], calls)
        connection.close()

    def test_daily_digest_generation_requires_typed_one_shot_confirmation(self) -> None:
        generated: list[str] = []

        def generate() -> dict[str, object]:
            generated.append("digest")
            return {"video_generated": True, "external_delivery": False}

        server, _thread, actions = self._server(
            daily_digest_catalog_provider=lambda _limit: _digest_catalog(),
            daily_digest_video_provider=lambda _token: b"private-digest",
            daily_digest_executor=generate,
        )
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        cookie, nonce = self._control_session(connection, server, "daily-digest")
        payload = urlencode(
            {
                "csrf": server.application.control_session_token,
                "nonce": nonce,
                "action": "daily-digest",
                "confirmation": "DIGEST",
            }
        )
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cookie": cookie,
            "Origin": server.canonical_origin,
        }
        connection.request("POST", "/action", body=payload, headers=headers)
        response = connection.getresponse()
        response.read()
        self.assertEqual(303, response.status)
        self.assertEqual("/daily-digest", response.getheader("Location"))
        self.assertEqual(["digest"], generated)
        self.assertEqual([], actions)

        connection.request("POST", "/action", body=payload, headers=headers)
        replay = connection.getresponse()
        replay.read()
        self.assertEqual(409, replay.status)
        self.assertEqual(["digest"], generated)
        connection.close()

    def test_review_media_and_atomic_labels_are_local_session_bound(self) -> None:
        token = "A" * 43
        media_calls: list[tuple[str, str, int]] = []
        labels: list[tuple[str, str, str]] = []

        def reviews(limit: int) -> tuple[dict[str, object], ...]:
            self.assertEqual(6, limit)
            return (
                {
                    "review_token": token,
                    "recorded_at": "2026-08-19T00:00:02+00:00",
                    "gate_source": "hardware_signal",
                    "review_state": "review_recommended",
                    "review_reasons": ["face_not_visible"],
                    "manual_review_recommended": True,
                    "frame_count": 1,
                    "clip_available": True,
                    "answers": [
                        {
                            "predicate": "face_visibility",
                            "model_answer": "clear",
                            "model_confidence": 0.5,
                            "human_answer": None,
                            "allowed_human_answers": [
                                "clear",
                                "partially_obscured",
                                "not_visible",
                                "unknown",
                            ],
                        }
                    ],
                    "event_id": "must-not-render",
                    "path": "/private/must-not-render",
                },
            )

        def media(review_token: str, kind: str, index: int) -> tuple[str, bytes]:
            if review_token != token or kind != "frame" or index != 0:
                raise ValueError("unavailable")
            media_calls.append((review_token, kind, index))
            return "image/jpeg", b"local-frame"

        def label(
            review_token: str,
            predicate: str,
            human_answer: str,
        ) -> dict[str, object]:
            labels.append((review_token, predicate, human_answer))
            return {
                "risk_inference": False,
                "notification_decision": False,
            }

        def calibration() -> dict[str, object]:
            return {
                "version": 1,
                "observation_count": 1,
                "labeled_observation_count": 1,
                "answer_count": 2,
                "labeled_answer_count": 1,
                "unlabeled_answer_count": 1,
                "exact_agreement_count": 1,
                "exact_agreement_rate": 1.0,
                "groups": [],
                "human_reviewed_visible_fact_labels": True,
                "exact_answer_agreement_evaluated": True,
                "risk_accuracy_evaluated": False,
                "media_read": False,
                "paths_disclosed": False,
                "event_identity_disclosed": False,
                "subject_identity_inference": False,
                "notification_decision": False,
            }

        server, _thread, actions = self._server(
            review_provider=reviews,
            review_media_provider=media,
            review_label_recorder=label,
            review_calibration_provider=calibration,
        )
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        connection.request("GET", "/review")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        policy = response.getheader("Content-Security-Policy", "")

        self.assertEqual(200, response.status)
        self.assertIn("建议优先人工回看：模型未看到脸部", body)
        self.assertIn("只决定回看顺序，不是风险或通知结论", body)
        self.assertIn("HttpOnly", response.getheader("Set-Cookie", ""))
        self.assertIn("候选事件回看", body)
        self.assertIn("模型：清晰 50%（未校准）", body)
        self.assertIn("人工标注进度", body)
        self.assertIn("已完成 1/2 个问题", body)
        self.assertIn("精确一致 1/1（100.0%）", body)
        self.assertIn("不用填写文字", body)
        self.assertIn("不是危险识别率", body)
        self.assertIn(f"/review-media/{token}/frame/0", body)
        self.assertNotIn("must-not-render", body)
        self.assertNotIn("/private", body)
        self.assertIn("img-src 'self'", policy)
        self.assertIn("media-src 'self'", policy)
        self.assertEqual([], media_calls)
        self.assertEqual([], actions)

        matched = re.search(
            rf'name="nonce" value="([^"]+)"><input type="hidden" '
            rf'name="review_token" value="{token}"><input type="hidden" '
            r'name="predicate" value="face_visibility">',
            body,
        )
        self.assertIsNotNone(matched)
        nonce = matched.group(1)  # type: ignore[union-attr]

        other = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        other.request("GET", f"/review-media/{token}/frame/0")
        forbidden = other.getresponse()
        forbidden.read()
        self.assertEqual(403, forbidden.status)
        other.close()

        connection.request(
            "GET",
            f"/review-media/{token}/frame/0",
            headers={"Cookie": cookie},
        )
        local_media = connection.getresponse()
        self.assertEqual(200, local_media.status)
        self.assertEqual("image/jpeg", local_media.getheader("Content-Type"))
        self.assertEqual("bytes", local_media.getheader("Accept-Ranges"))
        self.assertEqual(b"local-frame", local_media.read())
        self.assertEqual([(token, "frame", 0)], media_calls)

        connection.request(
            "GET",
            f"/review-media/{token}/frame/0",
            headers={"Cookie": cookie, "Range": "bytes=0-4"},
        )
        partial = connection.getresponse()
        self.assertEqual(206, partial.status)
        self.assertEqual("bytes 0-4/11", partial.getheader("Content-Range"))
        self.assertEqual(b"local", partial.read())

        connection.request(
            "GET",
            f"/review-media/{token}/frame/0",
            headers={"Cookie": cookie, "Range": "bytes=99-100"},
        )
        invalid_range = connection.getresponse()
        invalid_range.read()
        self.assertEqual(416, invalid_range.status)
        self.assertEqual("bytes */11", invalid_range.getheader("Content-Range"))
        self.assertEqual(3, len(media_calls))

        payload = urlencode(
            {
                "csrf": server.application.control_session_token,
                "nonce": nonce,
                "review_token": token,
                "predicate": "face_visibility",
                "human_answer": "clear",
            }
        )
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cookie": cookie,
            "Origin": "https://example.invalid",
        }
        connection.request("POST", "/review-label", body=payload, headers=headers)
        cross_origin = connection.getresponse()
        cross_origin.read()
        self.assertEqual(403, cross_origin.status)
        self.assertEqual([], labels)

        headers["Origin"] = server.canonical_origin
        connection.request("POST", "/review-label", body=payload, headers=headers)
        accepted = connection.getresponse()
        accepted.read()
        self.assertEqual(303, accepted.status)
        self.assertEqual("/review", accepted.getheader("Location"))
        self.assertEqual([(token, "face_visibility", "clear")], labels)
        self.assertEqual([], actions)

        connection.request("POST", "/review-label", body=payload, headers=headers)
        replay = connection.getresponse()
        replay.read()
        self.assertEqual(409, replay.status)
        self.assertEqual(1, len(labels))
        connection.close()

    def test_review_callbacks_must_be_configured_as_one_closed_set(self) -> None:
        with self.assertRaisesRegex(Exception, "review callbacks are invalid"):
            HomeDashboardApplication(
                status_provider=_status,
                action_executor=lambda _action, _confirmed: _status(),
                review_provider=lambda _limit: (),
            )
        with self.assertRaisesRegex(Exception, "calibration callback is invalid"):
            HomeDashboardApplication(
                status_provider=_status,
                action_executor=lambda _action, _confirmed: _status(),
                review_calibration_provider=lambda: {},
            )
        with self.assertRaisesRegex(Exception, "calibration callback is invalid"):
            HomeDashboardApplication(
                status_provider=_status,
                action_executor=lambda _action, _confirmed: _status(),
                review_calibration_provider="not-callable",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(Exception, "notification callback is invalid"):
            HomeDashboardApplication(
                status_provider=_status,
                action_executor=lambda _action, _confirmed: _status(),
                notification_test_executor="not-callable",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(Exception, "daily report callbacks are invalid"):
            HomeDashboardApplication(
                status_provider=_status,
                action_executor=lambda _action, _confirmed: _status(),
                daily_report_executor=lambda: {},
            )
        with self.assertRaisesRegex(Exception, "daily digest executor is invalid"):
            HomeDashboardApplication(
                status_provider=_status,
                action_executor=lambda _action, _confirmed: _status(),
                daily_digest_executor=lambda: {},
            )
        with self.assertRaisesRegex(Exception, "daily digest callbacks are invalid"):
            HomeDashboardApplication(
                status_provider=_status,
                action_executor=lambda _action, _confirmed: _status(),
                daily_digest_catalog_provider=lambda _limit: _digest_catalog(),
            )

    def test_local_notification_test_requires_test_confirmation_and_one_shot_nonce(
        self,
    ) -> None:
        delivered: list[str] = []

        def notification_test() -> dict[str, object]:
            delivered.append("local_test")
            return {
                "status": "delivered",
                "test_only": True,
                "external_delivery_used": False,
                "event_authority": False,
                "risk_authority": False,
            }

        server, _thread, actions = self._server(
            notification_test_executor=notification_test,
        )
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        cookie, nonce = self._control_session(connection, server, "notification-test")
        payload = urlencode(
            {
                "csrf": server.application.control_session_token,
                "nonce": nonce,
                "action": "notification-test",
                "confirmation": "TEST",
            }
        )
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cookie": cookie,
            "Origin": server.canonical_origin,
        }
        connection.request("POST", "/action", body=payload, headers=headers)
        response = connection.getresponse()
        response.read()
        self.assertEqual(303, response.status)
        self.assertEqual("/", response.getheader("Location"))
        self.assertEqual(["local_test"], delivered)
        self.assertEqual([], actions)

        connection.request("POST", "/action", body=payload, headers=headers)
        replay = connection.getresponse()
        replay.read()
        self.assertEqual(409, replay.status)
        self.assertEqual(["local_test"], delivered)
        connection.close()

    def test_daily_report_generation_and_read_are_local_one_shot_actions(self) -> None:
        generated: list[str] = []

        def generate() -> dict[str, object]:
            generated.append("report")
            return {
                "status": "generated",
                "media_read": False,
                "paths_disclosed": False,
                "notification_decision": False,
            }

        def provide() -> bytes:
            return b"<!doctype html><html><body>private-safe-report</body></html>"

        server, _thread, actions = self._server(
            daily_report_executor=generate,
            daily_report_provider=provide,
        )
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        cookie, nonce = self._control_session(connection, server, "daily-report")
        payload = urlencode(
            {
                "csrf": server.application.control_session_token,
                "nonce": nonce,
                "action": "daily-report",
                "confirmation": "",
            }
        )
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cookie": cookie,
            "Origin": server.canonical_origin,
        }
        connection.request("POST", "/action", body=payload, headers=headers)
        response = connection.getresponse()
        response.read()
        self.assertEqual(303, response.status)
        self.assertEqual("/daily-report", response.getheader("Location"))
        self.assertEqual(["report"], generated)
        self.assertEqual([], actions)

        connection.request("GET", "/daily-report")
        report = connection.getresponse()
        body = report.read().decode("utf-8")
        self.assertEqual(200, report.status)
        self.assertIn("private-safe-report", body)
        self.assertIn("default-src 'none'", report.getheader("Content-Security-Policy", ""))

        connection.request("POST", "/action", body=payload, headers=headers)
        replay = connection.getresponse()
        replay.read()
        self.assertEqual(409, replay.status)
        self.assertEqual(["report"], generated)
        connection.close()

    def test_valid_post_requires_cookie_origin_csrf_and_typed_confirmation(self) -> None:
        server, _thread, actions = self._server()
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        cookie, nonce = self._control_session(connection, server, "start")
        payload = urlencode(
            {
                "csrf": server.application.control_session_token,
                "nonce": nonce,
                "action": "start",
                "confirmation": "START",
            }
        )
        connection.request(
            "POST",
            "/action",
            body=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Cookie": cookie,
                "Origin": server.canonical_origin,
            },
        )
        response = connection.getresponse()
        response.read()

        self.assertEqual(303, response.status)
        self.assertEqual("/control", response.getheader("Location"))
        self.assertEqual([("start", True)], actions)

        connection.request(
            "POST",
            "/action",
            body=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Cookie": cookie,
                "Origin": server.canonical_origin,
            },
        )
        replay = connection.getresponse()
        replay.read()
        self.assertEqual(409, replay.status)
        self.assertEqual([("start", True)], actions)
        connection.close()

    def test_cross_origin_or_missing_confirmation_cannot_execute_action(self) -> None:
        server, _thread, actions = self._server()
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        cookie, nonce = self._control_session(connection, server, "shadow-start")

        def post(origin: str, confirmation: str) -> int:
            payload = urlencode(
                {
                    "csrf": server.application.control_session_token,
                    "nonce": nonce,
                    "action": "shadow-start",
                    "confirmation": confirmation,
                }
            )
            connection.request(
                "POST",
                "/action",
                body=payload,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Cookie": cookie,
                    "Origin": origin,
                },
            )
            result = connection.getresponse()
            result.read()
            return result.status

        self.assertEqual(403, post("https://example.invalid", "SHADOW"))
        self.assertEqual(400, post(server.canonical_origin, ""))
        self.assertEqual([], actions)
        connection.close()

    def test_soak24_is_a_one_shot_manual_action_with_explicit_scope(self) -> None:
        server, _thread, actions = self._server()
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        cookie, nonce = self._control_session(connection, server, "soak-start")
        payload = urlencode(
            {
                "csrf": server.application.control_session_token,
                "nonce": nonce,
                "action": "soak-start",
                "confirmation": "SOAK24",
            }
        )
        connection.request(
            "POST",
            "/action",
            body=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Cookie": cookie,
                "Origin": server.canonical_origin,
            },
        )
        response = connection.getresponse()
        response.read()

        self.assertEqual(303, response.status)
        self.assertEqual([("soak-start", True)], actions)

        connection.request("GET", "/control", headers={"Cookie": cookie})
        controls = connection.getresponse()
        body = controls.read().decode("utf-8")
        self.assertIn("启动四路 YOLO presence 24 小时稳定性测试", body)
        self.assertIn("24 小时后自动停止", body)
        self.assertIn("不启动 Mage-VL、身份、风险、主分析或通知 worker", body)

        long_status = _status()
        long_status["candidate_soak"]["latest"]["state"] = "running"
        long_status["candidate_soak"]["latest"]["elapsed_seconds"] = 691_200
        long_html = render_home_dashboard(long_status).decode("utf-8")
        self.assertIn("已运行 8.0 天", long_html)
        connection.close()

    def test_action_failure_is_fixed_and_does_not_echo_exception(self) -> None:
        def fail(_action: str, _confirmed: bool) -> dict[str, object]:
            raise RuntimeError("/private/path and secret detail")

        server, _thread, _actions = self._server(action_executor=fail)
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=2)
        cookie, nonce = self._control_session(connection, server, "stop")
        payload = urlencode(
            {
                "csrf": server.application.control_session_token,
                "nonce": nonce,
                "action": "stop",
                "confirmation": "",
            }
        )
        connection.request(
            "POST",
            "/action",
            body=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Cookie": cookie,
                "Origin": server.canonical_origin,
            },
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8")

        self.assertEqual(503, response.status)
        self.assertEqual("action failed", body)
        self.assertNotIn("private", body)
        self.assertNotIn("secret", body)
        connection.close()

    def test_renderer_ignores_unknown_components_and_escapes_status(self) -> None:
        status = _status()
        status["components"] = [
            {"component": "unknown", "state": "<script>", "enabled": True},
            {
                "component": "xiaomi_stream_gateway",
                "state": "<inactive>",
                "enabled": False,
            },
        ]
        page = render_home_dashboard(status).decode("utf-8")

        self.assertNotIn("<script>", page)
        self.assertIn("&lt;inactive&gt;", page)

    def test_renderer_explains_fixed_queue_failures_without_private_details(self) -> None:
        status = _status()
        status["base_operational"] = True
        status["ha_signal_outbox"] = {
            "status": "available",
            "pending": 0,
            "running": 0,
            "failed": 14,
            "failed_error_counts": {"route_missing": 14, "private_detail": 99},
        }

        page = render_home_dashboard(status).decode("utf-8")

        self.assertIn("运行中 · 需处理", page)
        self.assertIn("摄像头路由未接通 14", page)
        self.assertNotIn("private_detail", page)

    def test_renderer_reports_missing_analysis_media_without_reading_it(self) -> None:
        status = _status()
        status["base_operational"] = True
        status["analysis_queue"] = {
            "status": "available",
            "pending": 1,
            "running": 0,
            "completed": 4,
            "failed": 0,
            "missing_media_files": 2,
        }

        page = render_home_dashboard(status).decode("utf-8")

        self.assertIn("运行中 · 需处理", page)
        self.assertIn("2 个证据文件缺失", page)

    def test_renderer_shows_path_free_candidate_wait_and_lifecycle(self) -> None:
        page = render_home_dashboard(_status()).decode("utf-8")

        self.assertIn("四路 YOLO 候选健康", page)
        self.assertIn("1/2 路运行正常", page)
        self.assertIn("C500 主视角 · 等待新片段", page)
        self.assertIn("等显卡 1.2 秒 · 片段 5.5 秒", page)
        self.assertIn(
            "累计 120 段 · 候选发出 8 / 抑制 5 · 等显卡 P95≤3.0 秒 / 峰值 4.7 秒",
            page,
        )
        self.assertIn("失败会话 2 次 · 非正常重启 0 次", page)
        self.assertIn("候选发出 8 / 抑制 5", page)
        self.assertIn("SOAK24：</strong>已完成", page)
        self.assertIn("已运行 24.0 小时 / 24.0 小时", page)
        self.assertIn(
            "成功 240 段 · 候选发出 8 / 抑制 5 · 扫描失败 0 次",
            page,
        )
        self.assertIn("C500 第二视角 · 已停止", page)
        self.assertNotIn("camera_01", page)
        self.assertNotIn("camera_02", page)

    def test_sampling_evidence_shows_actual_calls_and_boundaries_without_mutation(self) -> None:
        status = sampling_dashboard_status()
        before = json.dumps(status, sort_keys=True)
        page = render_home_dashboard(status).decode("utf-8")
        for expected in (
            "最近完成扫描（历史记录，非实时）",
            "实际模型调用 3 次 · 零框 2 次 · 解码 51 帧",
            "全部解码帧时间戳回退 1 次 · 容器 PTS 回退 1 次",
            "最大相邻源帧 PTS 间隔：0.125 秒",
            "不是模型采样间隔或确认丢失时长",
            "源完整性未验证",
            "零框不等于无人或安全",
            "不保证输入连续、检出率或安全",
        ):
            self.assertIn(expected, page)
        self.assertEqual(before, json.dumps(status, sort_keys=True))
        self.assertFalse(status["candidate_health"]["cameras"][0]["health"]["media_read"])
        self.assertNotIn("camera_01", page)
        self.assertNotIn("<img", page)
        self.assertIn("1/1 路运行正常", page)

    def test_sampling_zero_boxes_are_distinct_from_unknown_legacy_evidence(self) -> None:
        status = sampling_dashboard_status()
        health = status["candidate_health"]["cameras"][0]["health"]
        health["last_sampling"]["zero_detection_sample_count"] = 0
        self.assertIn("零框 0 次", render_home_dashboard(status).decode("utf-8"))
        for version, evidence in (
            (1, sampling_health_fixture()),
            (3, None),
            (4, None),
            (True, None),
        ):
            with self.subTest(version=version):
                health["version"] = version
                health["last_sampling"] = evidence
                page = render_home_dashboard(status).decode("utf-8")
                self.assertIn("实际采样证据未知", page)
                self.assertNotIn("实际模型调用", page)
                self.assertNotIn("零框 0 次", page)

    def test_sampling_stopped_stale_and_clock_skew_are_explicit_history(self) -> None:
        for state, freshness in (("stopped", "fresh"), ("idle", "stale"), ("idle", "clock_skew")):
            with self.subTest(state=state, freshness=freshness):
                page = render_home_dashboard(
                    sampling_dashboard_status(state=state, freshness=freshness)
                ).decode("utf-8")
                self.assertIn("历史采样记录（已停止或状态不新鲜）", page)
                self.assertIn("实际模型调用 3 次", page)
                self.assertIn("不证明当前看护", page)
                self.assertIn("0/1 路运行正常", page)

    def test_sampling_inflight_and_failed_states_reject_leftover_summary(self) -> None:
        for state in ("starting", "scanning", "failed"):
            with self.subTest(state=state):
                page = render_home_dashboard(sampling_dashboard_status(state=state)).decode("utf-8")
                self.assertIn("实际采样证据未知", page)
                self.assertNotIn("实际模型调用", page)

    def test_sampling_requires_valid_completed_timestamp(self) -> None:
        for completed in (None, 1, "", "PRIVATE_PATH", "2026-09-12T12:00:00", "x" * 100):
            with self.subTest(completed=completed):
                status = sampling_dashboard_status()
                status["candidate_health"]["cameras"][0]["health"]["last_scan_completed_at"] = (
                    completed
                )
                page = render_home_dashboard(status).decode("utf-8")
                self.assertIn("实际采样证据未知", page)
                self.assertNotIn("PRIVATE_PATH", page)
                self.assertNotIn("实际模型调用", page)

    def test_sampling_rejects_invalid_numbers_relationships_and_exhaustion_claims(self) -> None:
        cases = {
            "sampled_frame_count": (True, 0, -1, 100_001, 52, 1.5),
            "zero_detection_sample_count": (True, -1, 4, float("nan")),
            "decoded_frame_count": (True, 2, 10_000_001, float("inf")),
            "timestamp_fallback_frame_count": (True, -1, 52),
            "container_pts_regression_count": (True, -1, 50),
            "max_adjacent_container_pts_gap_seconds": (
                True,
                -1,
                float("nan"),
                float("inf"),
                10**1000,
            ),
            "end_reason": ("PRIVATE_PATH", None, "verified_eof"),
            "source_exhaustion_verified": (True, 0, None),
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    status = sampling_dashboard_status()
                    status["candidate_health"]["cameras"][0]["health"]["last_sampling"][field] = (
                        value
                    )
                    page = render_home_dashboard(status).decode("utf-8")
                    self.assertIn("实际采样证据未知", page)
                    self.assertNotIn("PRIVATE_PATH", page)
                    self.assertNotIn("实际模型调用", page)

    def test_sampling_rejects_malicious_or_incomplete_summary_without_echo(self) -> None:
        for change in ("extra", "missing", "private_count", "not_dict", "insufficient_pts"):
            with self.subTest(change=change):
                status = sampling_dashboard_status()
                health = status["candidate_health"]["cameras"][0]["health"]
                evidence = health["last_sampling"]
                if change == "extra":
                    evidence["private_path"] = "<script>PRIVATE_PATH</script>"
                elif change == "missing":
                    del evidence["decoded_frame_count"]
                elif change == "private_count":
                    evidence["sampled_frame_count"] = "<script>PRIVATE_PATH</script>"
                elif change == "not_dict":
                    health["last_sampling"] = "PRIVATE_PATH"
                else:
                    evidence["timestamp_fallback_frame_count"] = 50
                    evidence["container_pts_regression_count"] = 0
                page = render_home_dashboard(status).decode("utf-8")
                self.assertIn("实际采样证据未知", page)
                self.assertNotIn("PRIVATE_PATH", page)
                self.assertNotIn("实际模型调用", page)

    def test_sampling_gap_none_zero_tiny_and_huge_are_bounded_and_distinct(self) -> None:
        for gap, expected in (
            (None, "未知"),
            (0, "0.000 秒"),
            (0.00001, "&lt;0.001 秒"),
            (1e300, "1.000e+300 秒"),
        ):
            with self.subTest(gap=gap):
                status = sampling_dashboard_status()
                status["candidate_health"]["cameras"][0]["health"]["last_sampling"][
                    "max_adjacent_container_pts_gap_seconds"
                ] = gap
                page = render_home_dashboard(status).decode("utf-8")
                self.assertIn(f"最大相邻源帧 PTS 间隔：{expected}", page)
                self.assertIn("实际模型调用 3 次", page)
                self.assertLess(len(page), 30_000)

    def test_renderer_shows_room_person_facts_without_claiming_baby_location(self) -> None:
        page = render_home_dashboard(_status()).decode("utf-8")

        self.assertIn("房间人员存在", page)
        self.assertIn("1 个房间检测到人员", page)
        self.assertIn("宝宝房", page)
        self.assertIn("检测到人员", page)
        self.assertIn("客厅", page)
        self.assertIn("覆盖中断", page)
        self.assertIn("同一台摄像头的不同视角可以覆盖不同房间", page)
        self.assertIn("它不是宝宝身份或宝宝位置", page)
        self.assertNotIn("同一 C500 双视角不会算成两个房间", page)
        self.assertNotIn("room_a", page)
        self.assertNotIn("camera_01", page)

    def test_renderer_ignores_invalid_timeline_values_and_private_fields(self) -> None:
        status = _status()
        status["event_digest"] = {
            "status": "available",
            "local_date": "2026-08-18",
            "today_total": 1,
            "event_type_counts": {"invented": 99},
            "recent": [
                {
                    "local_time": "<private>",
                    "event_type": "invented",
                    "risk_level": "high",
                    "confidence": 1,
                    "needs_review": True,
                    "event_id": "must-not-render",
                }
            ],
        }

        page = render_home_dashboard(status).decode("utf-8")

        self.assertNotIn("must-not-render", page)
        self.assertNotIn("&lt;private&gt;", page)
        self.assertIn("暂无历史事件记录", page)

    def test_cli_wires_loopback_dashboard_without_starting_monitoring(self) -> None:
        output = io.StringIO()
        stopped = {
            "version": 1,
            "status": "stopped",
            "requests": 0,
            "bind": "loopback",
            "remote_used": False,
            "media_read": False,
            "paths_disclosed": False,
        }

        def serve(application: object, **kwargs: object) -> dict[str, object]:
            ready = kwargs["ready"]
            ready(8765)  # type: ignore[operator]
            self.assertNotIn("/private/outbox.sqlite3", repr(application))
            self.assertNotIn("/private/analysis.sqlite3", repr(application))
            return stopped

        with (
            mock.patch("baby_monitor.cli_commands.home_dashboard.control_home_stack") as control,
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.serve_home_dashboard", side_effect=serve
            ),
            redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "home-dashboard",
                    "--ha-outbox-database",
                    "/private/outbox.sqlite3",
                    "--analysis-database",
                    "/private/analysis.sqlite3",
                    "--port",
                    "8765",
                    "--max-runtime-seconds",
                    "60",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertFalse(control.called)
        decoder = json.JSONDecoder()
        first, offset = decoder.raw_decode(output.getvalue())
        second, _offset = decoder.raw_decode(output.getvalue()[offset:].lstrip())
        self.assertEqual("ready", first["status"])
        self.assertFalse(first["automatic_monitoring_start"])
        self.assertEqual(stopped, second)
        self.assertNotIn("/private", output.getvalue())


if __name__ == "__main__":
    unittest.main()
