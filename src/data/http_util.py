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


def http_get_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout: float = 45.0,
    headers: Mapping[str, str] | None = None,
) -> Any:
    """GET JSON from ``url`` with SSL fallbacks suitable for vendor APIs."""
    global _warned_insecure
    params = dict(params or {})
    headers = dict(headers or {})
    verify = _verify_setting()

    try:
        import requests

        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=verify)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("requests GET failed (%s); trying curl", exc)

    # System curl (SecureTransport on macOS) often works when OpenSSL does not.
    full = url if not params else f"{url}?{urlencode(params)}"
    curl_cmd = [
        "curl",
        "-sS",
        "-L",
        "--fail-with-body",
        "-H",
        "Accept: application/json",
        "--max-time",
        str(int(timeout)),
        full,
    ]
    for k, v in headers.items():
        curl_cmd[5:5] = ["-H", f"{k}: {v}"]
    try:
        proc = subprocess.run(curl_cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout:
            return json.loads(proc.stdout)
        err = (proc.stderr or proc.stdout or "")[:300]
        logger.debug("curl GET failed rc=%s: %s", proc.returncode, err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("curl subprocess failed: %s", exc)

    # Last resort: insecure requests (corporate MITM / broken CA chain).
    if verify is not False:
        if not _warned_insecure:
            logger.warning(
                "TLS verify failed for vendor HTTP; retrying with verify=False. "
                "Set VENDOR_SSL_VERIFY=0 to silence, or fix the local CA store."
            )
            _warned_insecure = True
        import requests

        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"HTTP GET failed for {url}")
