"""HTTP helpers resilient to broken Python SSL trust stores (common on macOS).

System ``curl`` often verifies correctly when ``requests`` / ``curl_cffi`` do not.
Prefer ``requests`` with certifi; fall back to ``curl`` subprocess, then
``verify=False`` as a last resort (logged once).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Mapping
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

_warned_insecure = False
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _verify_setting() -> bool | str:
    """Respect ``VENDOR_SSL_VERIFY`` (0/false disables verification)."""
    flag = os.environ.get("VENDOR_SSL_VERIFY", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    try:
        import certifi

        return certifi.where()
    except ImportError:  # pragma: no cover
        return True


def http_get_text(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout: float = 45.0,
    headers: Mapping[str, str] | None = None,
    accept: str = "text/html,application/xhtml+xml",
) -> str:
    """GET text/HTML from ``url`` with the same SSL fallbacks as JSON helpers."""
    global _warned_insecure
    params = dict(params or {})
    hdrs = {"Accept": accept, "User-Agent": _DEFAULT_UA}
    if headers:
        hdrs.update(headers)
    verify = _verify_setting()

    try:
        import requests

        resp = requests.get(url, params=params, headers=hdrs, timeout=timeout, verify=verify)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        logger.debug("requests GET text failed (%s); trying curl", exc)

    full = url if not params else f"{url}?{urlencode(params)}"
    curl_cmd = [
        "curl",
        "-sS",
        "-L",
        "--fail-with-body",
        "-H",
        f"Accept: {accept}",
        "-H",
        f"User-Agent: {_DEFAULT_UA}",
        "--max-time",
        str(int(timeout)),
        full,
    ]
    for k, v in hdrs.items():
        if k.lower() in {"accept", "user-agent"}:
            continue
        curl_cmd[5:5] = ["-H", f"{k}: {v}"]
    try:
        proc = subprocess.run(curl_cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        err = (proc.stderr or proc.stdout or "")[:300]
        logger.debug("curl GET text failed rc=%s: %s", proc.returncode, err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("curl subprocess failed: %s", exc)

    if verify is not False:
        if not _warned_insecure:
            logger.warning(
                "TLS verify failed for vendor HTTP; retrying with verify=False. "
                "Set VENDOR_SSL_VERIFY=0 to silence, or fix the local CA store."
            )
            _warned_insecure = True
        import requests

        resp = requests.get(url, params=params, headers=hdrs, timeout=timeout, verify=False)
        resp.raise_for_status()
        return resp.text

    raise RuntimeError(f"HTTP GET failed for {url}")


def http_get_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout: float = 45.0,
    headers: Mapping[str, str] | None = None,
) -> Any:
    """GET JSON from ``url`` with SSL fallbacks suitable for vendor APIs."""
    text = http_get_text(
        url,
        params=params,
        timeout=timeout,
        headers=headers,
        accept="application/json",
    )
    return json.loads(text)
