# Security policy

This project processes highly sensitive household media. Treat every frame, clip, event record,
credential, network address, and device identifier as private unless it is an explicitly generated
fixture.

## Supported versions

Before the first public release, only the current default development branch is supported. After
versioned releases begin, this file must be updated with an explicit support table before any release
is described as security-supported.

## Reporting a vulnerability

Use the repository host's private vulnerability-reporting or Security Advisory feature when it is
available. Do not open a public issue with exploit details, credentials, household topology, logs,
frames, clips, database contents, or absolute paths.

If no private reporting channel is configured yet, open a content-free issue asking the maintainers
to establish private contact. Include no technical details in that issue. Publishing a dedicated
private security contact is a release gate; this placeholder process is not a substitute for one.

In a private report, include only the minimum synthetic reproduction needed to explain:

- affected component and revision;
- local prerequisites and expected/actual behavior;
- privacy, integrity, availability, or safety impact;
- a reproduction using generated data where possible;
- whether a credential or private artifact may already have been exposed.

Maintainers should acknowledge reports privately, reproduce them with synthetic data, coordinate a
fix and disclosure window, and avoid copying private evidence into issues or CI. No fixed response
time is promised before maintainers publish a supported-release policy.

## Security boundaries

The secure default is LAN-only and local-first:

- no public port mapping or router/firewall changes;
- remote review disabled unless both enablement and upload consent are explicit;
- no frame/clip upload by notifications;
- no retention deletion without persistent opt-in and one-shot confirmation;
- no model download or unreviewed remote code at core installation time;
- no credentials, source URIs, local paths, model raw output, or private media in health/audit logs.

A report that a default violates one of these boundaries is security-relevant. Vulnerabilities in a
third-party camera bridge, model runtime, FFmpeg, MediaMTX, driver, or NAS should also be reported to
that upstream project; this repository cannot grant rights or support for third-party components.

## Safety disclaimer

This is an assistive monitoring project, not a medical device, certified alarm, or replacement for
direct supervision. A model miss, false alarm, unavailable stream, delayed queue, or incorrect
inference must never be treated as proof that a child is safe or unsafe.
