"""M8 signed stateless MuJoCo session protocol (stdlib only).

No server-side session database is used. A signed token guarantees integrity,
NOT at-most-once consumption: clients must serialize actions. A stolen valid
token can be replayed until expiry. Never claim global single-use semantics.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
import time
import zlib

import numpy as np

FRAME_SIZE = 64 * 64 * 3
STATE_LIMIT = 512  # 51 for the frozen PushT XML; validate the exact size at runtime
MAX_STEPS = 64
LIFETIME_SECONDS = 15 * 60
TOKEN_LIMIT = 58_000


class SessionError(ValueError):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > TOKEN_LIMIT or not value:
        raise SessionError("Invalid token component")
    try:
        return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise SessionError("Invalid base64 token") from exc


def sign(payload: dict, secret: str) -> str:
    if not isinstance(secret, str) or len(secret.encode("utf-8")) < 32:
        raise RuntimeError("SESSION_SECRET must have at least 32 UTF-8 bytes")
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    token = _b64(body) + "." + _b64(mac)
    if len(token) > TOKEN_LIMIT:
        raise SessionError("Session token too large")
    return token


def verify(token: str, secret: str, *, now: int | None = None) -> dict:
    if not isinstance(token, str) or len(token) > TOKEN_LIMIT or token.count(".") != 1:
        raise SessionError("Malformed token")
    encoded, sig = token.split(".")
    body, mac = _unb64(encoded), _unb64(sig)
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise SessionError("Invalid session signature")
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionError("Invalid session payload") from exc
    t = int(time.time()) if now is None else int(now)
    if not isinstance(data, dict) or data.get("v") != 1:
        raise SessionError("Unrecognized session version")
    if not isinstance(data.get("exp"), int) or not t < data["exp"] <= t + LIFETIME_SECONDS:
        raise SessionError("Expired or invalid session")
    steps, seq = data.get("steps"), data.get("seq")
    if type(steps) is not int or type(seq) is not int or not 0 <= steps == seq <= MAX_STEPS:
        raise SessionError("Invalid session sequence")
    sid = data.get("sid")
    if not isinstance(sid, str) or len(sid) != 32 or not all(x in "0123456789abcdef" for x in sid):
        raise SessionError("Invalid session identity")
    return data


def new_payload(state: np.ndarray, previous: np.ndarray, current: np.ndarray,
                fingerprint: str, *, now: int | None = None) -> dict:
    t = int(time.time()) if now is None else int(now)
    return _payload(state, previous, current, fingerprint,
                    sid=secrets.token_hex(16), exp=t + LIFETIME_SECONDS, seq=0)


def next_payload(old: dict, state: np.ndarray, previous: np.ndarray, current: np.ndarray) -> dict:
    if old["steps"] >= MAX_STEPS:
        raise SessionError("Maximum session length reached; reset to continue")
    return _payload(state, previous, current, old["fp"], sid=old["sid"],
                    exp=old["exp"], seq=old["seq"] + 1)


def _payload(state, previous, current, fingerprint, *, sid, exp, seq):
    arr = np.asarray(state, dtype="<f8")
    if arr.ndim != 1 or not 1 <= arr.size <= STATE_LIMIT or not np.isfinite(arr).all():
        raise SessionError("Invalid MuJoCo integration state")
    p = np.asarray(previous)
    c = np.asarray(current)
    if p.dtype != np.uint8 or c.dtype != np.uint8 or p.shape != (64, 64, 3) or c.shape != p.shape:
        raise SessionError("Invalid real RGB window")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise SessionError("Invalid artifact fingerprint")
    return {
        "v": 1, "sid": sid, "exp": exp, "seq": seq, "steps": seq, "fp": fingerprint,
        "physics": _b64(arr.tobytes()), "nstate": arr.size,
        "rgb": _b64(zlib.compress(p.tobytes() + c.tobytes(), level=6)),
    }


def extract(data: dict, expected_fingerprint: str, expected_nstate: int):
    if data.get("fp") != expected_fingerprint:
        raise SessionError("Session is from another build or frozen artifact set")
    nstate = data.get("nstate")
    if type(nstate) is not int or nstate != expected_nstate:
        raise SessionError("MuJoCo state layout mismatch")
    raw = _unb64(data.get("physics"))
    if len(raw) != expected_nstate * 8:
        raise SessionError("Unexpected physics state byte size")
    state = np.frombuffer(raw, dtype="<f8").copy()
    if not np.isfinite(state).all():
        raise SessionError("Nonfinite physics state")
    compressed = _unb64(data.get("rgb"))
    try:
        inflate = zlib.decompressobj()
        frames = inflate.decompress(compressed, FRAME_SIZE * 2 + 1)
        # Do not call flush() on untrusted zlib data: it can bypass the
        # output-size cap. A complete zlib stream already sets eof=True.
        if not inflate.eof or inflate.unused_data or inflate.unconsumed_tail or len(frames) != FRAME_SIZE * 2:
            raise SessionError("Invalid RGB window")
    except zlib.error as exc:
        raise SessionError("Invalid RGB compression") from exc
    rgb = np.frombuffer(frames, dtype=np.uint8).reshape(2, 64, 64, 3).copy()
    return state, rgb[0], rgb[1]
