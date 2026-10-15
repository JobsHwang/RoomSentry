---
name: Bug report
about: Report a reproducible defect using synthetic or redacted data
title: "[Bug] "
labels: bug
assignees: ""
---

<!--
STOP: do not attach household frames, clips, audio, reports, databases, credentials, tokens,
device IDs, private IPs, absolute user paths, or credential-bearing URLs. Use generated fixtures.
Security vulnerabilities belong in the private process described by SECURITY.md.
-->

## Summary

Describe the defect and its impact in a few sentences.

## Reproduction

Provide the smallest synthetic reproduction, command, and configuration fields that are safe to
publish. Do not paste `.env` or `config/local.toml`.

## Expected and actual behavior

Explain what should happen and what happened. Redact logs to generic error codes.

## Environment

- OS/WSL version:
- Python version:
- Project revision:
- Adapter/backend names and versions (no credentials or private endpoints):

## Verification

List the tests or checks already run and their actual results.
