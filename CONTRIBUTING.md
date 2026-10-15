# Contributing

Thank you for helping build a privacy-first, LAN-only baby-monitoring project. Contributions are
welcome, but household safety and privacy take priority over convenience.

## Start here

Read these files before changing code:

1. `SPEC.md` for product contracts, architecture, privacy, and acceptance criteria;
2. `DEVELOPMENT_PLAN.md` for milestone order and decision gates;
3. `AGENTS.md` for repository-specific safety and verification rules;
4. `THIRD_PARTY_LICENSES.md` before adding a dependency, model, dataset, camera bridge, or media
   asset.

Keep camera vendors, media transports, signal sources, detectors, local/remote VLMs, notification
targets, and storage destinations behind adapters. RTSP is a preferred transport, not the domain
model. Home Assistant is an optional signal source, not a platform dependency.

## Protect private data

Never put any of the following in an issue, discussion, pull request, test, commit, or CI log:

- camera, NAS, router, Xiaomi, Home Assistant, cloud, or model-provider credentials;
- tokens, cookies, private keys, `.env`, `config/local.toml`, or credential-bearing URLs;
- household IP addresses, usernames, absolute user paths, device IDs, or camera serial numbers;
- family recordings, screenshots, audio, event databases, reports, or model prompts/responses based
  on private media.

Use generated fixtures, loopback addresses, opaque camera/event IDs, and visibly fake credentials.
Credential-shaped negative tests require the explicit
`release-audit: allow=fixture-credential` marker within the three immediately preceding lines of the
fixture.

If private data was submitted, stop sharing it, remove it from the proposed change, rotate any
exposed credential, and follow `SECURITY.md`. Do not rely on merely deleting a later commit.

## Development setup

The supported development target is Ubuntu 24.04 with Python 3.11 or 3.12. The standard-library core
does not require Docker, a camera, GPU, model download, NAS, cloud key, or public port.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/ruff check src tests
.venv/bin/python -m baby_monitor validate-json tests/fixtures/valid_vlm_event.json
.venv/bin/python -m baby_monitor release-audit --root .
```

Do not make an optional computer-vision or model dependency mandatory for core tests. Never add an
implicit model download. VLM A/B runs must load models sequentially.

## Making a change

- Keep each change bounded and preserve existing user work.
- Add or update contract tests before changing a public JSON shape.
- Keep `observations` limited to visible or system-detected facts; put uncertain interpretation in
  `inferences` and default insufficient evidence to `unknown`.
- Add actual verification results under `docs/verification/` after material changes.
- Do not add automatic deletion, external upload, public listening, or system-wide configuration as
  an incidental part of another feature.
- Record the exact upstream URL, revision, license, and redistribution boundary for new third-party
  components. Project-owned code is Apache-2.0; that does not relicense dependencies or weights.

## Commits and DCO

Contributions use the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
Sign commits using your authorized public contributor identity and an email address you are
prepared to make public. Prefer a GitHub no-reply email rather than exposing a private address:

```bash
git commit -s -m "feat: describe the change"
```

The sign-off certifies the contribution under the repository's applicable license and becomes part
of the public Git history. Fix a missing sign-off before review; do not add another person's sign-off
without their authorization.

## Pull-request checklist

Before requesting review:

- run the full standard-library suite, Ruff, required JSON validation, and `release-audit`;
- state which tests were run and their actual results;
- confirm that remote upload, public ports, retention deletion, and implicit model downloads remain
  disabled by default;
- explain any new adapter, dependency, network target, destructive action, or data migration;
- confirm that no private household data or third-party material without redistribution rights is
  included;
- sign every commit under the DCO.

Use `CODE_OF_CONDUCT.md` for community expectations and `SECURITY.md` for vulnerability reports.
