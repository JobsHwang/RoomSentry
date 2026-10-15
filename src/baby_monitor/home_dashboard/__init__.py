"""Local loopback dashboard: rendering, ports, application state, and server.

Historically a single ``home_dashboard.py``; the public surface is
re-exported here so existing imports keep working.
"""

from __future__ import annotations

from baby_monitor.home_dashboard.application import (  # noqa: F401
    HomeDashboardApplication,
    HomeDashboardError,
)
from baby_monitor.home_dashboard.rendering import (  # noqa: F401
    render_daily_digest_video,
    render_home_controls,
    render_home_dashboard,
    render_home_review,
)
from baby_monitor.home_dashboard.server import (  # noqa: F401
    make_home_dashboard_server,
    serve_home_dashboard,
)

__all__ = [
    "HomeDashboardApplication",
    "HomeDashboardError",
    "make_home_dashboard_server",
    "render_daily_digest_video",
    "render_home_controls",
    "render_home_dashboard",
    "render_home_review",
    "serve_home_dashboard",
]
