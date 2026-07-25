"""Google-OAuth session helpers + current-user dependency + Google credentials.

Auth is OPTIONAL: when Google creds aren't configured (``settings.auth_enabled`` is
False) the app stays fully un-gated and every existing feature works as before. When
configured, protected routes require a valid signed session cookie.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from sqlmodel import Session

from . import models
from .config import settings
from .database import get_session

SESSION_COOKIE = "session"
_ALG = "HS256"


def make_session_token(user_id: int) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    payload = {"uid": user_id, "iat": now, "exp": now + dt.timedelta(days=30)}
    return jwt.encode(payload, settings.session_secret, algorithm=_ALG)


def cookie_kwargs() -> dict:
    """Cookie flags: cross-site (SameSite=None;Secure) in prod (https), lax locally."""
    secure = settings.post_login_redirect.startswith("https")
    return {
        "httponly": True,
        "secure": secure,
        "samesite": "none" if secure else "lax",
        "max_age": 30 * 24 * 3600,
        "path": "/",
    }


def _user_from_request(request: Request, session: Session) -> Optional[models.User]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.session_secret, algorithms=[_ALG])
        return session.get(models.User, int(payload["uid"]))
    except Exception:  # noqa: BLE001 - bad/expired token → treat as logged out
        return None


def current_user(
    request: Request, session: Session = Depends(get_session)
) -> Optional[models.User]:
    """The logged-in user, or None. Never raises — routes decide whether to gate."""
    return _user_from_request(request, session)


def require_user(
    request: Request, session: Session = Depends(get_session)
) -> Optional[models.User]:
    """Require login ONLY when auth is enabled; otherwise allow (returns None), so the
    app is unchanged before Google creds exist."""
    if not settings.auth_enabled:
        return None
    user = _user_from_request(request, session)
    if user is None:
        raise HTTPException(status_code=401, detail="Login required")
    return user


def google_credentials(user: Optional[models.User], session: Session):
    """Valid google.oauth2 Credentials for the user (auto-refreshing), or None."""
    if user is None or not user.google_refresh_token:
        return None
    from google.auth.transport.requests import Request as GRequest
    from google.oauth2.credentials import Credentials

    creds = Credentials(
        token=user.google_access_token,
        refresh_token=user.google_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=settings.google_scopes,
    )
    try:
        if not creds.valid:
            creds.refresh(GRequest())
            user.google_access_token = creds.token
            user.google_token_expiry = creds.expiry
            session.add(user)
            session.commit()
    except Exception:  # noqa: BLE001
        return None
    return creds
