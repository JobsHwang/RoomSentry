# Changelog

## Unreleased

### Added

- Optional live detector/VLM adapters with a single shared VLM execution lease.
- Bounded sampling diagnostics, content-bound review evidence and local review workflows.
- Anonymous person tracking, association checkpoints and independent shadow experiments.
- Explicitly gated annotation and fine-tuning dataset preparation tools.
- Actual source-PTS sampling ledgers and coverage-aware reports.

### Fixed

- Preserve evidence gaps instead of treating unknown or stale inputs as clear observations.
- Isolate late asynchronous reviews from changed tracking generations and role scopes.
- Guard invalid motion predictions without stealing existing qualified associations.
- Preserve primary results when optional secondary review fails.
- Make report refresh requests recoverable and idempotent.

### Changed

- Condensed public documentation to product contracts, implementation status and limits.
- Removed household experiment narratives and operator-specific deployment information from the public source tree; local originals remain private.
- Kept all model experiments separate from production safety claims.
- Feature development is paused; services and scheduled work are not implicitly resumed.

### Security

- Added broader private-path checks and ignore rules for runtime/model artifacts.
- Public releases exclude household media, annotations, predictions, local configuration and operational logs.
- Third-party models, runtimes and data remain outside the Apache-2.0 source distribution.

No production infant-care accuracy or unattended-safety acceptance is claimed.
