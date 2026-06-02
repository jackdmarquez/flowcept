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


def _mongo_safe_key(key: str) -> str:
    """Rename keys that MongoDB update documents may treat as operators/paths."""
    if key.startswith("$"):
        key = f"_dollar_{key[1:]}"
    return key.replace(".", "_")


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
        return urlunsplit(SplitResult(parsed.scheme, redacted_netloc, parsed.path, parsed.query, parsed.fragment))

    return _URL_CREDENTIALS_PATTERN.sub(rf"\g<scheme>{REDACTED}@", value)


def sanitize_value(value: Any, mongo_safe: bool = True) -> Any:
    """Recursively redact sensitive values and optionally make dict keys Mongo-safe."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            string_key = str(key)
            safe_key = _mongo_safe_key(string_key) if mongo_safe else string_key
            sanitized[safe_key] = REDACTED if _is_sensitive_key(string_key) else sanitize_value(item, mongo_safe)
        return sanitized
    if isinstance(value, list):
        return [sanitize_value(item, mongo_safe) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item, mongo_safe) for item in value]
    if isinstance(value, str):
        if _OPENAI_KEY_PATTERN.search(value):
            return REDACTED
        return _redact_url_credentials(value)
    return value
