"""Simple ID/password login for the UI (hackathon rule: test account, no social/email login).

Accounts come from MEDOPS_UI_USERS (on Cloud Run: a Secret Manager secret) as JSON:
    {"judge": "scrypt$<salt hex>$<hash hex>", ...}
Only scrypt hashes are stored; plaintext passwords never are. When no accounts are
configured (local quick start), the UI needs no login.

    python -m medops.auth hash        # prompts for a password, prints the stored form
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
from typing import Optional

_N, _R, _P = 2 ** 14, 8, 1


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    candidate = hash_password(password, bytes.fromhex(salt_hex)).split("$")[2]
    return hmac.compare_digest(candidate, hash_hex)


def load_users(raw: Optional[str] = None) -> dict[str, str]:
    raw = raw if raw is not None else os.environ.get("MEDOPS_UI_USERS", "")
    if not raw.strip():
        return {}
    users = json.loads(raw)
    if not isinstance(users, dict):
        raise ValueError("MEDOPS_UI_USERS must be a JSON object {user: scrypt-hash}")
    return users


def check_login(username: str, password: str, users: dict[str, str]) -> bool:
    stored = users.get(username.strip())
    if stored is None:
        hash_password(password)  # same cost for unknown users (no username probing by timing)
        return False
    return verify_password(password, stored)


if __name__ == "__main__":
    if sys.argv[1:] == ["hash"]:
        pw = getpass.getpass("password: ")
        print(hash_password(pw))
    else:
        print(__doc__)
