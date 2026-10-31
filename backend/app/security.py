"""Passwords, session tokens and encryption of provider API keys."""

import base64
import hashlib
import hmac
import os
import secrets

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.config import get_settings

# scrypt parameters (OWASP: N=2^17, r=8, p=1 is ~128 MB; 2^15 keeps login fast on a laptop).
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 128 * 1024 * 1024


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, maxmem=_SCRYPT_MAXMEM
    )
    salt_b64 = base64.b64encode(salt).decode()
    digest_b64 = base64.b64encode(digest).decode()
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt_b64}${digest_b64}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            maxmem=_SCRYPT_MAXMEM,
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class SecretKeyMissing(RuntimeError):
    pass


def _credential_key() -> bytes:
    secret = get_settings().app_secret_key
    if len(secret) < 32:
        raise SecretKeyMissing(
            "APP_SECRET_KEY is missing or too short in backend/.env (run scripts/setup.sh)."
        )
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"ai-image-studio/credentials/v1"
    ).derive(secret.encode())


def encrypt_secret(plaintext: str, *, associated: str) -> bytes:
    """AES-256-GCM. `associated` (user id + provider) binds the ciphertext to its row."""
    nonce = os.urandom(12)
    sealed = AESGCM(_credential_key()).encrypt(nonce, plaintext.encode(), associated.encode())
    return b"v1" + nonce + sealed


def decrypt_secret(blob: bytes, *, associated: str) -> str:
    if blob[:2] != b"v1":
        raise ValueError("Unknown credential format")
    nonce, sealed = blob[2:14], blob[14:]
    return AESGCM(_credential_key()).decrypt(nonce, sealed, associated.encode()).decode()


def mask_key(last4: str) -> str:
    return "••••••••" + last4
