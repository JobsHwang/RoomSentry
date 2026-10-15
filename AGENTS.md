# AGENTS.md

## Scope

This repository implements a privacy-first, LAN-only baby-monitoring pipeline. Keep camera vendors, detectors, local VLMs, remote providers, and storage targets behind adapters.

## Source of truth

- `SPEC.md` defines the current product scope, logical/physical architecture, contracts, privacy rules, and acceptance criteria.
- `DEVELOPMENT_PLAN.md` defines milestone order, decision gates, verification requirements, and current implementation status.
- Project-owned code and documentation are Apache-2.0. Do not imply that third-party libraries, model weights, datasets, media, or camera bridges are relicensed; follow `THIRD_PARTY_LICENSES.md`.
- Optimize first for a useful, sustainable open-source project. Sponsorship or paid support may happen, but monetization, acquisition, and closed-core packaging are not product requirements.
- Keep `MediaSource` and `SignalSource` separate. RTSP is the preferred live transport, not the internal domain model; HA is an optional signal adapter, not a required platform.

## Non-negotiable safety rules

- Never read, print, commit, or log passwords, camera credentials, tokens, API keys, private NAS data, or `.env` contents.
- Never open public ports, alter router/firewall settings, or expose services beyond the LAN as part of routine development.
- Do not upload frames or clips unless remote review is explicitly enabled by the user and configuration.
- Do not download large model weights or start multiple VLMs without explicit confirmation.
- Do not delete existing media. Retention deletion requires both a persistent opt-in setting and a one-shot destructive CLI confirmation.
- Keep generated filenames free of personal names. Use UTC time, sanitized camera ID, and opaque event ID.
- Treat `observations` as visible/system-detected facts only; put uncertain interpretation in `inferences`.
- Default uncertain results to `unknown`; do not inflate confidence.

## Development workflow

- Target Ubuntu 24.04 WSL2 under `~/projects/baby-monitor`; do not choose `/mnt/c` for final deployment.
- Keep the core runnable with Python 3.11+ standard library. Put computer-vision/model dependencies behind optional extras.
- Run relevant tests after every material change and append actual results under `docs/verification/`.
- A/B runs must load backends sequentially so two VLMs are not resident together.
- Do not hard-code a Xiaomi bridge implementation until its exact upstream URL is supplied and reviewed.
- Preserve user changes and never use destructive Git commands.

## Validation commands

```bash
python -m unittest discover -s tests -t . -v
python -m baby_monitor validate-json tests/fixtures/valid_vlm_event.json
python -m baby_monitor validate-signal-json tests/fixtures/valid_signal_event.json
python -m baby_monitor validate-decision-json tests/fixtures/valid_decision_record.json
```
