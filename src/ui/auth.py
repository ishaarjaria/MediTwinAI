"""
Demo access for the MediTwin academic prototype.

THIS IS NOT PRODUCTION AUTHENTICATION. It exists so the product journey
(landing page -> sign in -> workspace) can be demonstrated. Be precise about what it is:

* Accounts live only in this server process's memory. They disappear when the server
  restarts and are never written to disk, a database or a network service.
* Access lasts for one browser session (Streamlit ``session_state``). Refreshing the
  page or opening a new tab starts a new session and asks to sign in again.
* Passwords are hashed with PBKDF2-HMAC-SHA256 (standard library, random salt), so the
  in-memory store never holds a plain-text password. That is hygiene, not security:
  there is no rate limiting, e-mail verification, password reset, TLS or audit trail.
* The app is a single-process teaching demo; anyone who can read the source can bypass
  the gate. Do not enter a real password. "Continue as demo user" needs none.

A production system would delegate to an identity provider (Streamlit ``st.login`` with
OIDC) over HTTPS. That is deliberately out of scope here.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from typing import MutableMapping

ITERATIONS = 120_000
MIN_PASSWORD_LENGTH = 6
SESSION_KEY = "auth_user"

DEMO_NOTICE = (
    "Educational demo access. Accounts exist only in this running session and are not stored. "
    "Please do not enter a real password."
)
DEMO_LIMITS = (
    "This is not production authentication: there is no e-mail verification, password reset, "
    "encryption in transit or audit log."
)

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(ValueError):
    """A sign-in or sign-up problem, with a message that is safe to show to the user."""


@dataclass(frozen=True)
class User:
    name: str
    email: str
    demo: bool = False

    @property
    def initials(self) -> str:
        parts = [p for p in re.split(r"\s+", self.name.strip()) if p]
        letters = "".join(p[0] for p in parts[:2]).upper()
        return letters or "U"

    @property
    def role(self) -> str:
        return "Demo access" if self.demo else "Demo account"


DEMO_USER = User(name="Demo Administrator", email="demo@meditwin.example", demo=True)


# ----------------------------------------------------------------------
# Password hashing (standard library only)
# ----------------------------------------------------------------------
def hash_password(password: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")  # noqa: E731
    return f"pbkdf2_sha256${iterations}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# ----------------------------------------------------------------------
# In-memory account store
# ----------------------------------------------------------------------
class AccountStore:
    """Process-memory registry of demo accounts (never persisted)."""

    def __init__(self) -> None:
        self._accounts: dict[str, tuple[str, str]] = {}  # email -> (name, password hash)

    @staticmethod
    def normalise_email(email: str) -> str:
        return (email or "").strip().lower()

    def __len__(self) -> int:
        return len(self._accounts)

    def create(self, name: str, email: str, password: str) -> User:
        name = " ".join((name or "").split())
        email = self.normalise_email(email)
        if not name:
            raise AuthError("Please enter your name.")
        if len(name) > 80:
            raise AuthError("Please use a shorter name.")
        if not _EMAIL.match(email):
            raise AuthError("Please enter an e-mail address such as name@example.com.")
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise AuthError(f"Please choose a password with at least {MIN_PASSWORD_LENGTH} characters.")
        if email == DEMO_USER.email or email in self._accounts:
            raise AuthError("An account with this e-mail already exists in this session. Try signing in.")
        self._accounts[email] = (name, hash_password(password))
        return User(name=name, email=email)

    def authenticate(self, email: str, password: str) -> User:
        email = self.normalise_email(email)
        record = self._accounts.get(email)
        # Always run one hash so a missing account and a wrong password take similar time.
        stored = record[1] if record else hash_password("x", salt=b"\0" * 16)
        ok = verify_password(password or "", stored)
        if not record or not ok:
            raise AuthError("That e-mail and password do not match an account in this session.")
        return User(name=record[0], email=email)


# ----------------------------------------------------------------------
# Session helpers (state is passed in, so they can be tested without Streamlit)
# ----------------------------------------------------------------------
def current_user(state: MutableMapping) -> User | None:
    user = state.get(SESSION_KEY)
    return user if isinstance(user, User) else None


def sign_in(state: MutableMapping, user: User) -> None:
    state[SESSION_KEY] = user


def sign_out(state: MutableMapping) -> None:
    """End the session: the user and everything the session remembered (view, last results)."""
    for key in list(state.keys()):
        try:
            del state[key]
        except Exception:  # a widget-bound key may refuse deletion; the user entry is what matters
            pass
    state.pop(SESSION_KEY, None)


def is_authenticated(state: MutableMapping) -> bool:
    return current_user(state) is not None


def get_store() -> AccountStore:
    """The process-wide in-memory account store (Streamlit keeps one per server process)."""
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _store() -> AccountStore:
        return AccountStore()

    return _store()
