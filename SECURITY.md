# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| < 2.0   | :x:                |

Insight137 EAP follows semantic versioning. Security fixes are backported to the current major version only.

## Reporting a Vulnerability

If you discover a security vulnerability in Insight137 EAP, please report it privately so we can address it before public disclosure.

**Do not open a public GitHub issue for security vulnerabilities.**

### How to report

Email **support@insight137.com** with:

- A description of the vulnerability
- Steps to reproduce
- The version of `insight137-eap` affected
- Any proof-of-concept code or output (if applicable)
- Your name and a link to attribute (if you wish to be credited)

### What to expect

- **Acknowledgment within 7 days** — we'll confirm we received your report.
- **Initial assessment within 14 days** — we'll respond with whether we've reproduced the issue, our severity assessment, and an estimated fix timeline.
- **Fix within 90 days** for confirmed vulnerabilities.
- **Coordinated disclosure** — we ask reporters not to publicly disclose the vulnerability until a fix is released or 90 days have passed, whichever comes first.

### Recognition

We don't currently offer a bug bounty. We do credit responsible disclosures in:

- The release notes for the version that includes the fix
- A `SECURITY-CREDITS.md` file in the repository (if you opt in)

## Scope

In scope:
- The `insight137-eap` PyPI package and its source code in this repository
- The mathematical operations and entropy computations exposed by the library

Out of scope:
- The Insight137 web platform (insight137.com) — please report platform vulnerabilities via the same email but note "Platform" in the subject line
- Third-party dependencies — please report directly to the upstream project
- Issues in research papers, datasets, or external publications

## Cryptography

This library does not implement cryptographic primitives or process secrets. If you discover what appears to be a cryptographic issue, it is most likely a defect in a downstream consumer's use of the library rather than the library itself.

## Security Design

The library is designed with the following security principles, which may help researchers scope their reports:

- **No network access:** Pure computation, no HTTP calls or external connections
- **No file I/O:** Does not read or write files (except when run as `__main__` for verification output)
- **No exec/eval:** No dynamic code execution
- **Input validation:** All public functions validate inputs and reject malformed data
- **Immutable outputs:** `PsiProfile` is a frozen dataclass — cannot be modified after creation
- **No secrets:** The library contains no API keys, credentials, or sensitive data

## Hall of Fame

Researchers who have responsibly disclosed vulnerabilities will be listed here once the first credit is earned.
