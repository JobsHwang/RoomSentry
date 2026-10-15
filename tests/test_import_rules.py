from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_import_rules

NEW_RUNTIME_SURFACES = (
    "baby_monitor.live_monitor",
    "baby_monitor.event_policy_runtime",
    "baby_monitor.identity_runtime",
    "baby_monitor.local_perception",
    "baby_monitor.runtime_trial",
    "baby_monitor.visual_predicate_shadow_queue",
    "baby_monitor.visual_predicate_shadow_runtime",
    "baby_monitor.visual_predicate_shadow_worker",
)


class ImportRulesTest(unittest.TestCase):
    def _fixture_graph(self, sources: dict[str, str]) -> dict[str, set[str]]:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-import-rules-") as directory:
            package_root = Path(directory) / "baby_monitor"
            for relative, source in {"__init__.py": "", **sources}.items():
                path = package_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            return check_import_rules.build_graph(package_root)

    def test_graph_resolves_plain_imports_and_aliases_without_executing_modules(self) -> None:
        graph = self._fixture_graph(
            {
                "consumer.py": (
                    "import baby_monitor.database as db, baby_monitor.domain.enums\n"
                    "import sqlite3, baby_monitor_unrelated\n"
                    "raise RuntimeError('graph construction must not execute code')\n"
                ),
                "database.py": "",
                "domain/__init__.py": "",
                "domain/enums.py": "",
            }
        )
        self.assertEqual(
            {"baby_monitor.database", "baby_monitor.domain.enums"},
            graph["baby_monitor.consumer"],
        )
        self.assertEqual(set(), graph["baby_monitor.database"])

    def test_graph_resolves_relative_imports_from_module_files(self) -> None:
        graph = self._fixture_graph(
            {
                "feature/__init__.py": "",
                "feature/consumer.py": (
                    "from . import support as local_support\n"
                    "from .support import helper as aliased_helper\n"
                    "from .. import database as db\n"
                    "from ..domain.enums import Kind\n"
                ),
                "feature/support.py": "helper = None\n",
                "database.py": "",
                "domain/__init__.py": "",
                "domain/enums.py": "Kind = None\n",
            }
        )
        self.assertEqual(
            {
                "baby_monitor",
                "baby_monitor.database",
                "baby_monitor.domain.enums",
                "baby_monitor.feature",
                "baby_monitor.feature.support",
            },
            graph["baby_monitor.feature.consumer"],
        )

    def test_graph_resolves_relative_imports_from_package_initializers(self) -> None:
        graph = self._fixture_graph(
            {
                "__init__.py": "from . import database as db\n",
                "database.py": "",
                "feature/__init__.py": (
                    "from . import child as child_alias\n"
                    "from .support import helper\n"
                    "from ..domain import enums as shared_enums\n"
                ),
                "feature/child.py": "",
                "feature/support.py": "helper = None\n",
                "feature/nested/__init__.py": "from .. import support\n",
                "domain/__init__.py": "from .enums import Kind\n",
                "domain/enums.py": "Kind = None\n",
            }
        )
        self.assertEqual({"baby_monitor.database"}, graph["baby_monitor"])
        self.assertEqual(
            {
                "baby_monitor.feature.child",
                "baby_monitor.feature.support",
                "baby_monitor.domain",
                "baby_monitor.domain.enums",
            },
            graph["baby_monitor.feature"],
        )
        self.assertEqual(
            {"baby_monitor.feature", "baby_monitor.feature.support"},
            graph["baby_monitor.feature.nested"],
        )
        self.assertEqual({"baby_monitor.domain.enums"}, graph["baby_monitor.domain"])

    def test_domain_purity_cannot_be_bypassed_by_import_spelling(self) -> None:
        for source in (
            "import baby_monitor.database as db\n",
            "from .. import database as db\n",
            "from ..database import EventDatabase\n",
        ):
            with self.subTest(source=source):
                graph = self._fixture_graph(
                    {
                        "domain/__init__.py": "",
                        "domain/consumer.py": source,
                        "database.py": "EventDatabase = None\n",
                    }
                )
                self.assertIn(
                    "domain purity: baby_monitor.domain.consumer imports baby_monitor.database",
                    check_import_rules.check_rules(graph),
                )

    def test_cycles_are_detected_across_plain_and_relative_imports(self) -> None:
        graph = self._fixture_graph(
            {
                "first.py": "import baby_monitor.second as other\n",
                "second.py": "from . import first as other\n",
            }
        )
        self.assertTrue(
            any(item.startswith("import cycle:") for item in check_import_rules.check_rules(graph))
        )

    def test_relative_import_cannot_escape_the_package_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes top-level package"):
            self._fixture_graph({"consumer.py": "from ..outside import external\n"})

    def test_business_rules_only_depend_on_domain_and_other_business_rules(self) -> None:
        for importer in check_import_rules.BUSINESS_RULES:
            with self.subTest(importer=importer):
                allowed = {"baby_monitor.domain", "baby_monitor.domain.signal"}
                allowed.update(set(check_import_rules.BUSINESS_RULES) - {importer})
                graph = {module: set() for module in allowed}
                graph[importer] = allowed
                self.assertEqual([], check_import_rules.check_rules(graph))

    def test_business_rules_reject_application_and_infrastructure_dependencies(self) -> None:
        for importer in check_import_rules.BUSINESS_RULES:
            for imported in (
                "baby_monitor.cli_commands.synthetic",
                "baby_monitor.adapters.detector",
                "baby_monitor.config",
                "baby_monitor.database",
                "baby_monitor.analysis_service",
                "baby_monitor.presence_history",
                "baby_monitor.recognition_attempt",
                "baby_monitor.remote_review",
                "baby_monitor.identity_enrollment",
                "baby_monitor.identity_store",
                "baby_monitor.report_queue",
                "baby_monitor.report_service",
                "baby_monitor.report_lock",
                "baby_monitor.runtime_trial",
            ):
                with self.subTest(importer=importer, imported=imported):
                    self.assertIn(
                        f"business rule purity: {importer} imports {imported}",
                        check_import_rules.check_rules({importer: {imported}, imported: set()}),
                    )

    def test_stores_and_analysis_service_stay_in_the_infrastructure_layer(self) -> None:
        for importer in (
            "baby_monitor.adapters.synthetic",
            "baby_monitor.local_runners.synthetic",
            "baby_monitor.household_pose_fusion_diagnostic.synthetic",
        ):
            for imported in (
                "baby_monitor.analysis_service",
                "baby_monitor.presence_history",
                "baby_monitor.recognition_attempt",
                "baby_monitor.remote_review",
                "baby_monitor.identity_enrollment",
                "baby_monitor.identity_store",
                "baby_monitor.report_queue",
                "baby_monitor.report_service",
                "baby_monitor.report_lock",
            ):
                with self.subTest(importer=importer, imported=imported):
                    self.assertTrue(
                        check_import_rules.check_rules({importer: {imported}, imported: set()})
                    )

    def test_adapters_can_import_the_extracted_ports_and_value_contracts(self) -> None:
        importer = "baby_monitor.adapters.synthetic"
        for imported in (
            "baby_monitor.remote_review_port",
            "baby_monitor.identity_enrollment_contracts",
        ):
            with self.subTest(imported=imported):
                self.assertEqual(
                    [], check_import_rules.check_rules({importer: {imported}, imported: set()})
                )

    def test_dashboard_rendering_only_imports_shared_dashboard_contracts(self) -> None:
        importer = "baby_monitor.home_dashboard.rendering"
        for imported in (
            "baby_monitor.home_dashboard.copy",
            "baby_monitor.home_dashboard.ports",
            "baby_monitor.home_dashboard.errors",
        ):
            with self.subTest(imported=imported):
                self.assertEqual(
                    [], check_import_rules.check_rules({importer: {imported}, imported: set()})
                )
        for imported in (
            "baby_monitor.home_dashboard.application",
            "baby_monitor.home_dashboard.server",
            "baby_monitor.home_dashboard.server.synthetic",
            "baby_monitor.home_dashboard",
        ):
            with self.subTest(imported=imported):
                self.assertIn(
                    f"dashboard rendering isolation: {importer} imports {imported}",
                    check_import_rules.check_rules({importer: {imported}, imported: set()}),
                )

    def test_dashboard_error_reexports_preserve_exception_identity(self) -> None:
        from baby_monitor.home_dashboard import HomeDashboardError as public_error
        from baby_monitor.home_dashboard.application import HomeDashboardError as application_error
        from baby_monitor.home_dashboard.errors import HomeDashboardError as shared_error
        from baby_monitor.home_dashboard.rendering import HomeDashboardError as rendering_error

        self.assertIs(shared_error, public_error)
        self.assertIs(shared_error, application_error)
        self.assertIs(shared_error, rendering_error)

    def test_report_rendering_stays_in_the_infrastructure_layer(self) -> None:
        imported = "baby_monitor.reporting_rendering"
        for importer in (
            "baby_monitor.adapters.synthetic",
            "baby_monitor.local_runners.synthetic",
            "baby_monitor.event_policy_cascade.synthetic",
        ):
            with self.subTest(importer=importer):
                violations = check_import_rules.check_rules({importer: {imported}, imported: set()})
                self.assertTrue(violations)
                self.assertTrue(any(imported in violation for violation in violations))

    def test_repository_graph_obeys_import_contract(self) -> None:
        self.assertEqual(
            check_import_rules.check_rules(check_import_rules.build_graph()),
            [],
        )

    def test_evaluation_contracts_cannot_import_new_runtime_surfaces(self) -> None:
        importer = "baby_monitor.event_policy_cascade.synthetic"

        for imported in NEW_RUNTIME_SURFACES:
            with self.subTest(imported=imported):
                self.assertIn(
                    f"evaluation purity: {importer} imports {imported}",
                    check_import_rules.check_rules(
                        {
                            importer: {imported},
                            imported: set(),
                        }
                    ),
                )

    def test_household_model_diagnostic_is_an_evaluation_contract(self) -> None:
        importer = "baby_monitor.household_model_diagnostic"
        imported = "baby_monitor.local_perception"
        self.assertIn(
            f"evaluation purity: {importer} imports {imported}",
            check_import_rules.check_rules(
                {
                    importer: {imported},
                    imported: set(),
                }
            ),
        )

    def test_household_pose_fusion_is_an_evaluation_contract(self) -> None:
        importer = "baby_monitor.household_pose_fusion_diagnostic"
        imported = "baby_monitor.candidate_worker"
        self.assertIn(
            f"evaluation purity: {importer} imports {imported}",
            check_import_rules.check_rules(
                {
                    importer: {imported},
                    imported: set(),
                }
            ),
        )

    def test_household_pose_fusion_holdout_is_an_evaluation_contract(self) -> None:
        importer = "baby_monitor.household_pose_fusion_holdout"
        imported = "baby_monitor.candidate_worker"
        self.assertIn(
            f"evaluation purity: {importer} imports {imported}",
            check_import_rules.check_rules(
                {
                    importer: {imported},
                    imported: set(),
                }
            ),
        )

    def test_household_human_presence_audit_is_an_evaluation_contract(self) -> None:
        importer = "baby_monitor.household_human_presence_audit"
        imported = "baby_monitor.candidate_worker"
        self.assertIn(
            f"evaluation purity: {importer} imports {imported}",
            check_import_rules.check_rules(
                {
                    importer: {imported},
                    imported: set(),
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
