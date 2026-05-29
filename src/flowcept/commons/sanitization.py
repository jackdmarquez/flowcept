"""Sanitization helpers for provenance capture."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

REDACTED = "REDACTED"

SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "private_key",
    "client_secret",
    "aws_secret_access_key",
    "access_key",
    "mongo_uri",
    "uri",
    "connection_string",
)

_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]+")
_URL_CREDENTIALS_PATTERN = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<creds>[^/@\s]+)@")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _redact_url_credentials(value: str) -> str:
    """Redact credentials embedded in URL-like strings."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        parsed = None

    if parsed and parsed.scheme and parsed.netloc and "@" in parsed.netloc:
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        redacted_netloc = f"{REDACTED}@{host}" if host else REDACTED
        return urlunsplit(
            SplitResult(parsed.scheme, redacted_netloc, parsed.path, parsed.query, parsed.fragment)
        )

    return _URL_CREDENTIALS_PATTERN.sub(rf"\g<scheme>{REDACTED}@", value)


def sanitize_value(value: Any) -> Any:
    """Recursively redact sensitive keys and URI credentials."""
    if isinstance(value, dict):
        return {
            str(key): REDACTED if _is_sensitive_key(str(key)) else sanitize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        if _OPENAI_KEY_PATTERN.search(value):
            return REDACTED
        return _redact_url_credentials(value)
    return value
