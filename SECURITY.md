# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | Yes       |
| < 2.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in this library, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email: **roger@insight137.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Assessment:** Within 7 days
- **Fix (if confirmed):** Within 30 days
- **Disclosure:** Coordinated with reporter

## Scope

This policy covers the `insight137_eap.py` library code. It does not cover:
- Third-party dependencies (NumPy) — report those to their maintainers
- The Insight137 web platform (insight137.com) — report to security@insight137.com
- Research datasets — these are not distributed with this library

## Security Design

The library is designed with the following security principles:
- **No network access:** Pure computation, no HTTP calls or external connections
- **No file I/O:** Does not read or write files (except when run as `__main__` for verification output)
- **No exec/eval:** No dynamic code execution
- **Input validation:** All public functions validate inputs and reject malformed data
- **Immutable outputs:** PsiProfile is a frozen dataclass — cannot be modified after creation
- **No secrets:** The library contains no API keys, credentials, or sensitive data
