"""Email/password authentication using the Firebase Auth REST API."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import os
import requests

AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:"
TOKEN_URL = "https://securetoken.googleapis.com/v1/token"


def _firebase_urls():
    """Use production by default; allow the official local Auth emulator for development."""
    emulator = os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "").strip()
    if not emulator:
        return AUTH_URL, TOKEN_URL
    if emulator not in {"127.0.0.1:9099", "localhost:9099"}:
        raise AuthError("The local Firebase Auth emulator must use localhost:9099.")
    return (
        f"http://{emulator}/identitytoolkit.googleapis.com/v1/accounts:",
        f"http://{emulator}/securetoken.googleapis.com/v1/token",
    )


class AuthError(Exception):
    pass


@dataclass
class UserSession:
    email: str
    uid: str
    id_token: str
    refresh_token: str
    expires_at: datetime

    @classmethod
    def from_response(cls, data):
        try:
            return cls(
                email=data["email"], uid=data["localId"], id_token=data["idToken"],
                refresh_token=data["refreshToken"],
                expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=max(0, int(data.get("expiresIn", 3600)) - 60)),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise AuthError("Firebase returned an invalid session response.") from exc


_AUTH_ERRORS = {
    "EMAIL_EXISTS": "This email address already has an account.",
    "INVALID_EMAIL": "Please enter a valid email address.",
    "EMAIL_NOT_FOUND": "The email or password is incorrect.",
    "INVALID_PASSWORD": "The email or password is incorrect.",
    "INVALID_LOGIN_CREDENTIALS": "The email or password is incorrect.",
    "INVALID_CREDENTIAL": "The email or password is incorrect.",
    "USER_DISABLED": "This account is disabled.",
    "WEAK_PASSWORD": "Use a stronger password (at least 6 characters).",
    "OPERATION_NOT_ALLOWED": "Email/password sign-in is not enabled in Firebase.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Try again later.",
    "TOKEN_EXPIRED": "Your session expired. Please sign in again.",
}


def _post(url, body, *, form=False, transport=requests):
    try:
        reply = transport.post(url, data=body if form else None,
                               json=None if form else body, timeout=12)
        try:
            data = reply.json()
        except ValueError:
            raise AuthError("Firebase returned an invalid response.") from None
        if not isinstance(data, dict):
            raise AuthError("Firebase returned an invalid response.")
        if reply.status_code >= 400:
            detail = data.get("error", {})
            detail = detail.get("message", "") if isinstance(detail, dict) else ""
            code = str(detail).split(" : ")[0].split(" ")[0]
            raise AuthError(_AUTH_ERRORS.get(code, "Could not complete authentication. Please try again."))
        return data
    except requests.RequestException as exc:
        raise AuthError("Could not reach Firebase. Check your connection.") from exc


def validate_credentials(email, password, *, signup=False):
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email or ""):
        raise AuthError("Enter a valid email address.")
    if not password:
        raise AuthError("Enter your password.")
    if signup and len(password) < 6:
        raise AuthError("Your password must contain at least 6 characters.")


def register(api_key, email, password, *, transport=requests):
    validate_credentials(email, password, signup=True)
    if not api_key:
        raise AuthError("Firebase is not configured. Add FIREBASE_API_KEY to .env.")
    return UserSession.from_response(_post(
        f"{_firebase_urls()[0]}signUp?key={api_key}",
        {"email": email, "password": password, "returnSecureToken": True}, transport=transport))


def login(api_key, email, password, *, transport=requests):
    validate_credentials(email, password)
    if not api_key:
        raise AuthError("Firebase is not configured. Add FIREBASE_API_KEY to .env.")
    return UserSession.from_response(_post(
        f"{_firebase_urls()[0]}signInWithPassword?key={api_key}",
        {"email": email, "password": password, "returnSecureToken": True}, transport=transport))


def ensure_fresh_session(api_key, session, *, transport=requests):
    if session.expires_at > datetime.now(timezone.utc):
        return session
    if not api_key or not session.refresh_token:
        raise AuthError("Your session expired. Please sign in again.")
    data = _post(f"{_firebase_urls()[1]}?key={api_key}",
                 {"grant_type": "refresh_token", "refresh_token": session.refresh_token},
                 form=True, transport=transport)
    try:
        return UserSession(
            email=session.email, uid=data["user_id"], id_token=data["id_token"],
            refresh_token=data["refresh_token"],
            expires_at=datetime.now(timezone.utc)+timedelta(seconds=max(0, int(data["expires_in"])-60)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise AuthError("Firebase returned an invalid session response.") from exc


def clear_user_state(state):
    # clear only the current Streamlit session; keep the shared guide cache
    for key in list(state.keys()):
        if key not in {"_streamlit_auth_style"}:
            del state[key]
