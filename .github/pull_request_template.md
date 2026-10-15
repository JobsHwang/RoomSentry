## Outcome

Describe the user-visible result and the bounded scope of this change.

## Safety and privacy

- [ ] No credentials, private endpoints/paths, device IDs, household media, databases, or reports.
- [ ] No public listening, router/firewall change, remote upload, deletion, or implicit model download.
- [ ] Observations remain facts; uncertain interpretation remains inference or `unknown`.
- [ ] New dependencies/assets include exact source, revision, license, and redistribution notes.

## Verification

List exact commands and actual results, including relevant resource measurements.

- [ ] Full standard-library tests pass.
- [ ] Ruff and required JSON/Schema checks pass.
- [ ] `python -m baby_monitor release-audit --root .` passes.
- [ ] Material results are recorded under `docs/verification/`.

## Contribution

- [ ] Every commit includes a DCO sign-off (`git commit -s`).
- [ ] I have the right to submit this contribution under the applicable repository license.
