"""Service-to-service auth on Cloud Run.

All Med-Ops services are deployed IAM-protected (no allUsers invoker). When running ON
Cloud Run, calls to another https Cloud Run service carry a Google ID token minted for the
attached runtime service account (metadata server). Locally nothing is added, so the
local stack (http://127.0.0.1, Orthanc basic auth) is unchanged.
"""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

from medops.settings import on_cloud_run

_TOKEN_TTL_S = 45 * 60          # Google ID tokens live 60 min
_lock = threading.Lock()
_cache: dict[str, tuple[str, float]] = {}


def _audience(url: str) -> str:
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}"


def id_token_headers(url: str) -> dict:
    if not on_cloud_run() or not url.startswith("https://"):
        return {}
    aud = _audience(url)
    with _lock:
        token, at = _cache.get(aud, (None, 0.0))
        if token is None or time.time() - at > _TOKEN_TTL_S:
            import google.auth.transport.requests
            import google.oauth2.id_token

            token = google.oauth2.id_token.fetch_id_token(
                google.auth.transport.requests.Request(), aud)
            _cache[aud] = (token, time.time())
    return {"Authorization": f"Bearer {token}"}


try:
    from requests.auth import AuthBase
except ImportError:  # pragma: no cover - requests is a hard dependency
    AuthBase = object


class IdTokenAuth(AuthBase):
    """requests auth hook: adds the Cloud Run ID token (no-op off Cloud Run / for http)."""

    def __call__(self, r):
        r.headers.update(id_token_headers(r.url))
        return r
