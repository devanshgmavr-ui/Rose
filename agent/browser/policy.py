"""Browser URL policy and validation.

Stage 2.4.2 - Safe URL navigation.

Provides URL validation, scheme restriction, and audit log
sanitization for browser navigation. Only HTTP/HTTPS schemes
are permitted. Sensitive query parameters are redacted in logs.
"""

import re
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = {"http", "https"}

BLOCKED_SCHEMES = {
    "file",
    "javascript",
    "data",
    "vbscript",
    "about",
    "chrome",
    "edge",
    "view-source",
    "blob",
    "filesystem",
}

SENSITIVE_PARAM_PATTERNS = {
    "token",
    "access_token",
    "api_key",
    "key",
    "password",
    "secret",
    "auth",
    "code",
    "credential",
    "session",
    "sid",
    "jwt",
}


def validate_url(url: str) -> Tuple[bool, List[str]]:
    """Validate a URL for safe navigation.

    Checks:
    - Non-empty
    - Valid URL structure
    - Allowed scheme (http/https)
    - Non-empty hostname

    Args:
        url: The URL to validate.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []

    if not url or not url.strip():
        errors.append("URL is empty")
        return False, errors

    url = url.strip()

    try:
        parsed = urlparse(url)
    except Exception:
        errors.append("URL is malformed")
        return False, errors

    if not parsed.scheme:
        errors.append("URL has no scheme")
    elif parsed.scheme.lower() not in ALLOWED_SCHEMES:
        errors.append(
            f"Unsupported URL scheme: {parsed.scheme}. "
            f"Allowed: {', '.join(sorted(ALLOWED_SCHEMES))}"
        )

    if not parsed.hostname:
        errors.append("URL has no hostname")

    return (len(errors) == 0, errors)


def sanitize_url_for_log(url: str) -> str:
    """Sanitize a URL for safe audit logging.

    Redacts sensitive query parameters while preserving
    the rest of the URL structure.

    Args:
        url: The URL to sanitize.

    Returns:
        Sanitized URL string safe for logging.
    """
    if not url:
        return url

    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        sanitized_params = {}
        changed = False
        for key, values in params.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in SENSITIVE_PARAM_PATTERNS):
                sanitized_params[key] = ["[REDACTED]"]
                changed = True
            else:
                sanitized_params[key] = values

        if changed:
            flat_params = {}
            for key, values in sanitized_params.items():
                flat_params[key] = values[0] if len(values) == 1 else values
            new_query = urlencode(flat_params, doseq=True)
            sanitized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                "",
            ))
            return sanitized

        return url
    except Exception:
        return "[URL_PARSE_ERROR]"
