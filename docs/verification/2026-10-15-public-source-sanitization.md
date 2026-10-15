# Public source sanitization — 2026-10-15

## Scope

- Condensed the public README, specification, development plan and licensing/deployment summaries.
- Removed household experiment narratives and research handoffs from the public tree; original source and Git history were backed up privately before history replacement.
- Kept code, synthetic fixtures, versioned contracts and general architecture documentation.
- Removed operator-specific launcher defaults, expanded private-path checks and added a regression test covering portable path forms.
- Updated CI to install the standard-library core offline so cleaned subprocess environments can import it without a temporary PYTHONPATH exception.
- No model, household media, production service, scheduled task, firewall or router setting was changed or started.

## Completed checks

| Check | Actual result |
|---|---|
| Full repository unittest suite | 2,633 tests, 213.910 seconds, OK |
| Focused audit/live-media/release regression | 54 tests, 6.651 seconds, OK |
| Ruff lint | Passed |
| Ruff formatting | Passed after one mechanical test-file formatting update |
| Import rules | Passed: 219 modules, 695 internal imports, two reviewed exceptions |
| CLI/configuration references | Both generated references current |
| Python compilation | Passed |
| Event/signal/decision/notification JSON fixtures | All four validators passed |
| Windows launcher syntax | PowerShell parser passed; UI not started |
| Staged source privacy audit | Passed, no violations; reports only codes/paths/lines |
| Original Git bundle backup | Created and verified privately before rewrite |

## Limits

This is a source/privacy verification, not a model-quality or infant-care acceptance result. Household samples, labels, predictions, credentials, device/room mappings and operational logs are not published. Defaults for optional uploads, notifications, identity and deletion remain gated.

A rewritten public history cannot erase copies or cached references held by other parties. Remote refs and hosted CI are checked separately when publishing; local tests do not attest hosted CI.

## Offline source-package verification

`bash scripts/verify_source_release.sh` exited zero in 233.007 seconds. It built two byte-identical source archives, verified the embedded/standalone source SBOM, extracted a clean checkout and installed the core without PyPI access.

The extracted standard-library environment ran 2,633 tests in 211.168 seconds: OK, with 11 optional-dependency tests explicitly skipped. Lint passed. The package contained 739 staged source files plus the source SBOM. No model was loaded and no network was used by this verification.

The final change after that extracted-suite run only appends these verification results; final staged privacy and deterministic-package checks are repeated before publication.
