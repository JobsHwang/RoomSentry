"""Check the package-level import contract for the baby-monitor core.

The rules encode the intended layering so it cannot regress silently:

- The domain layer (``baby_monitor.domain``) stays pure: it may only import
  stdlib and itself.
- Business rules may only depend on the domain and other business rules.
- The CLI layer (``cli``, ``cli_support``, ``cli_commands.*``) is a leaf:
  nothing else may import it, except ``baby_monitor.__main__``.
- Adapters and local model runners must not depend on application
  infrastructure (configuration loading, databases, notification outboxes,
  pipelines, workers, dashboards, reporting, archival) — adapters receive
  ports via injection.
- The import graph must be acyclic.

Every exception is explicit in ``ALLOWED`` below with a reason. The script is
standard library only and exits non-zero on the first rule violation set.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "baby_monitor"

CLI_LAYER = (
    "baby_monitor.cli",
    "baby_monitor.cli_support",
)
CLI_PREFIX = "baby_monitor.cli_commands"

INFRASTRUCTURE = (
    "baby_monitor.config",
    "baby_monitor.database",
    "baby_monitor.evidence_binding",
    "baby_monitor.local_review",
    "baby_monitor.local_review_service",
    "baby_monitor.local_review_reports",
    "baby_monitor.live_monitor",
    "baby_monitor.live_people",
    "baby_monitor.live_people_view",
    "baby_monitor.live_transition_capture",
    "baby_monitor.live_confirmation_shadow",
    "baby_monitor.notification",
    "baby_monitor.pipeline",
    "baby_monitor.storage",
    "baby_monitor.candidate_worker",
    "baby_monitor.analysis_worker",
    "baby_monitor.analysis_service",
    "baby_monitor.presence_history",
    "baby_monitor.recognition_attempt",
    "baby_monitor.remote_review",
    "baby_monitor.identity_enrollment",
    "baby_monitor.identity_store",
    "baby_monitor.home_dashboard",
    "baby_monitor.home_control",
    "baby_monitor.home_daily_report",
    "baby_monitor.home_event_digest",
    "baby_monitor.recent_observations",
    "baby_monitor.home_notification",
    "baby_monitor.home_shadow_review",
    "baby_monitor.daily_digest",
    "baby_monitor.daily_digest_export",
    "baby_monitor.daily_digest_source",
    "baby_monitor.daily_digest_store",
    "baby_monitor.archive",
    "baby_monitor.archive_binding",
    "baby_monitor.archive_queue",
    "baby_monitor.reporting",
    "baby_monitor.reporting_rendering",
    "baby_monitor.report_queue",
    "baby_monitor.report_service",
    "baby_monitor.report_lock",
    "baby_monitor.runtime_status",
    "baby_monitor.runtime_trial",
)

BUSINESS_RULES = (
    "baby_monitor.fact_policy",
    "baby_monitor.signal_binding",
    "baby_monitor.policy",
    "baby_monitor.event_policy",
    "baby_monitor.live_detection",
    "baby_monitor.live_review_cadence",
    "baby_monitor.candidate_confirmation",
)

DASHBOARD_RENDERING_CONTRACTS = (
    "baby_monitor.home_dashboard.copy",
    "baby_monitor.home_dashboard.ports",
    "baby_monitor.home_dashboard.errors",
)

# Offline measurement contracts: pure functions over already-admitted
# artifacts. They must stay independent of worker runtimes and deployment
# surfaces so evaluation results cannot be coupled to service internals.
EVALUATION_CONTRACTS = (
    "baby_monitor.head_point_review",
    "baby_monitor.frame_role_reference",
    "baby_monitor.tracker_evaluation",
    "baby_monitor.detector_evaluation",
    "baby_monitor.detector_threshold_sweep",
    "baby_monitor.candidate_trigger_evaluation",
    "baby_monitor.household_ground_truth",
    "baby_monitor.household_sampled_coverage",
    "baby_monitor.household_model_diagnostic",
    "baby_monitor.household_model_temporal_diagnostic",
    "baby_monitor.household_model_temporal_batch",
    "baby_monitor.household_pose_fusion_diagnostic",
    "baby_monitor.household_pose_fusion_holdout",
    "baby_monitor.household_human_presence_audit",
    "baby_monitor.pose_quality",
    "baby_monitor.caviar_tracking",
    "baby_monitor.event_policy_cascade",
)

RUNTIME_SURFACE = (
    "baby_monitor.live_monitor",
    "baby_monitor.live_confirmation_shadow",
    "baby_monitor.candidate_worker",
    "baby_monitor.candidate_health",
    "baby_monitor.candidate_soak",
    "baby_monitor.candidate_presence",
    "baby_monitor.analysis_worker",
    "baby_monitor.analysis_queue",
    "baby_monitor.signal_dispatch",
    "baby_monitor.signal_outbox",
    "baby_monitor.ring_buffer",
    "baby_monitor.rtsp_soak",
    "baby_monitor.media_health",
    "baby_monitor.local_perception",
    "baby_monitor.event_policy_runtime",
    "baby_monitor.identity_runtime",
    "baby_monitor.visual_predicate_shadow_queue",
    "baby_monitor.visual_predicate_shadow_runtime",
    "baby_monitor.visual_predicate_shadow_worker",
    "baby_monitor.home_dashboard",
    "baby_monitor.home_control",
    "baby_monitor.home_daily_report",
    "baby_monitor.home_event_digest",
    "baby_monitor.home_notification",
    "baby_monitor.home_shadow_review",
    "baby_monitor.systemd_deployment",
    "baby_monitor.runtime_status",
)

# Reviewed exceptions: (importer, imported-prefix) with the standing reason.
ALLOWED: dict[tuple[str, str], str] = {
    ("baby_monitor.__main__", "baby_monitor.cli"): ("module entry point delegates to the CLI"),
    ("baby_monitor.adapters.media", "baby_monitor.storage"): (
        "media processors consume the injected storage port to allocate "
        "evidence files; they never construct storage themselves"
    ),
}


def _module_name(path: Path, package_root: Path | None = None) -> str:
    root = PACKAGE_ROOT if package_root is None else package_root
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((root.name, *parts))


def _from_import_target(node: ast.ImportFrom, *, name: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = name.split(".") if is_package else name.split(".")[:-1]
    if node.level > len(package_parts):
        raise ValueError(f"relative import escapes top-level package: {name}")
    base = ".".join(package_parts[: len(package_parts) - node.level + 1])
    return f"{base}.{node.module}" if node.module else base


def build_graph(package_root: Path | None = None) -> dict[str, set[str]]:
    """Collect explicit internal imports without executing package code.

    ``package_root`` is the package directory, not its parent. Accepting an
    alternate root lets tests exercise real Python syntax in isolated fixtures.
    """
    root = PACKAGE_ROOT if package_root is None else package_root
    modules = {_module_name(path, root): path for path in sorted(root.rglob("*.py"))}
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = _from_import_target(node, name=name, is_package=path.name == "__init__.py")
                candidates = (target, *(f"{target}.{alias.name}" for alias in node.names))
            else:
                continue
            for candidate in candidates:
                if candidate in modules and candidate != name:
                    graph[name].add(candidate)
    return graph


def _is_cli_layer(module: str) -> bool:
    return module in CLI_LAYER or module.startswith(CLI_PREFIX)


def _is_infrastructure(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in INFRASTRUCTURE)


def _is_runtime_surface(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in RUNTIME_SURFACE)


def _is_evaluation_contract(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".") for prefix in EVALUATION_CONTRACTS
    )


def _is_business_rule(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in BUSINESS_RULES)


def _allowed(importer: str, imported: str) -> bool:
    for (allowed_importer, prefix), _reason in ALLOWED.items():
        if importer == allowed_importer and (
            imported == prefix or imported.startswith(prefix + ".")
        ):
            return True
    return False


def check_rules(graph: dict[str, set[str]]) -> list[str]:
    violations: list[str] = []

    for importer, deps in sorted(graph.items()):
        for imported in sorted(deps):
            if _allowed(importer, imported):
                continue
            if importer.startswith("baby_monitor.domain") and not (
                imported.startswith("baby_monitor.domain")
            ):
                violations.append(f"domain purity: {importer} imports {imported}")
            if _is_business_rule(importer) and not (
                imported == "baby_monitor.domain"
                or imported.startswith("baby_monitor.domain.")
                or _is_business_rule(imported)
            ):
                violations.append(f"business rule purity: {importer} imports {imported}")
            if _is_cli_layer(imported) and not (
                _is_cli_layer(importer) or importer == "baby_monitor.__main__"
            ):
                violations.append(f"cli isolation: {importer} imports {imported}")
            if (
                (
                    importer == "baby_monitor.home_dashboard.rendering"
                    or importer.startswith("baby_monitor.home_dashboard.rendering.")
                )
                and (
                    imported == "baby_monitor.home_dashboard"
                    or imported.startswith("baby_monitor.home_dashboard.")
                )
                and not any(
                    imported == prefix or imported.startswith(prefix + ".")
                    for prefix in DASHBOARD_RENDERING_CONTRACTS
                )
            ):
                violations.append(f"dashboard rendering isolation: {importer} imports {imported}")
            if importer.startswith("baby_monitor.adapters") and _is_infrastructure(imported):
                violations.append(f"adapter purity: {importer} imports {imported}")
            if importer.startswith("baby_monitor.local_runners") and _is_infrastructure(imported):
                violations.append(f"runner purity: {importer} imports {imported}")
            if _is_evaluation_contract(importer) and (
                _is_infrastructure(imported) or _is_runtime_surface(imported)
            ):
                violations.append(f"evaluation purity: {importer} imports {imported}")

    # Acyclicity via iterative depth-first search with coloring.
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = defaultdict(int)
    for start in sorted(graph):
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, set[str], bool]] = [(start, set(graph[start]), False)]
        while stack:
            node, pending, expanding = stack.pop()
            if expanding:
                color[node] = BLACK
                continue
            if color[node] != WHITE:
                continue
            color[node] = GRAY
            stack.append((node, pending, True))
            for neighbor in sorted(pending):
                if color[neighbor] == GRAY:
                    violations.append(
                        f"import cycle: {neighbor} is part of a cycle reached from {start}"
                    )
                elif color[neighbor] == WHITE:
                    stack.append((neighbor, set(graph[neighbor]), False))
    return violations


def main(argv: list[str] | None = None) -> int:
    graph = build_graph()
    violations = check_rules(graph)
    if violations:
        for violation in violations:
            print(f"error: {violation}", file=sys.stderr)
        print(
            f"import contract violated: {len(violations)} problem(s); see "
            "docs/architecture.md for the layering rules",
            file=sys.stderr,
        )
        return 1
    print(
        f"import contract holds: {len(graph)} modules, "
        f"{sum(len(deps) for deps in graph.values())} internal imports, "
        f"{len(ALLOWED)} reviewed exceptions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
