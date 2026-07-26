"""Google OAuth endpoints: "Continue with Google" login, callback, logout, and /me.

The user clicks one button; Google's own screen handles email/password OR the
already-signed-in account. We only ever receive the tokens Google returns.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlmodel import Session, select

from .. import models
from ..auth import SESSION_COOKIE, cookie_kwargs, current_user, make_session_token
from ..config import settings
from ..database import get_session

router = APIRouter(prefix="/api/auth", tags=["auth"])

# oauthlib blocks http by default — allow it ONLY for local (http) dev, never in prod
# (https). Also relax token-scope so login completes even if some scopes weren't granted.
if settings.oauth_redirect_uri.startswith("http://"):
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")


def _flow(state: Optional[str] = None):
    from google_auth_oauthlib.flow import Flow

    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [settings.oauth_redirect_uri],
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=settings.google_scopes,
        redirect_uri=settings.oauth_redirect_uri,
        state=state,
    )


@router.get("/config")
def auth_config():
    """Lets the frontend know whether to show the login gate."""
    return {"auth_enabled": settings.auth_enabled}


@router.get("/me")
def me(user: Optional[models.User] = Depends(current_user)):
    if user is None:
        return JSONResponse({"authenticated": False, "auth_enabled": settings.auth_enabled})
    return {
        "authenticated": True,
        "auth_enabled": True,
        "user": {"email": user.email, "name": user.name, "picture": user.picture},
        "calendar_connected": bool(user.google_refresh_token),
    }


@router.get("/google/login")
def google_login():
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="Google auth is not configured")
    flow = _flow()
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    resp = RedirectResponse(auth_url)
    ck = cookie_kwargs()
    resp.set_cookie("oauth_state", state, httponly=True, max_age=600,
                    samesite=ck["samesite"], secure=ck["secure"], path="/")
    # Persist the PKCE code_verifier so the callback's fresh Flow can complete the exchange.
    verifier = getattr(flow, "code_verifier", None)
    if verifier:
        resp.set_cookie("oauth_verifier", verifier, httponly=True, max_age=600,
                        samesite=ck["samesite"], secure=ck["secure"], path="/")
    return resp


@router.get("/google/callback")
def google_callback(request: Request, session: Session = Depends(get_session)):
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="Google auth is not configured")
    flow = _flow(state=request.cookies.get("oauth_state"))
    verifier = request.cookies.get("oauth_verifier")
    if verifier:
        flow.code_verifier = verifier
    try:
        flow.fetch_token(authorization_response=str(request.url))
        creds = flow.credentials
        from google.auth.transport import requests as grequests
        from google.oauth2 import id_token as gid

        info = gid.verify_oauth2_token(creds.id_token, grequests.Request(), settings.google_client_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Google sign-in failed: {exc}")

    sub = info["sub"]
    user = session.exec(select(models.User).where(models.User.google_sub == sub)).first()
    if user is None:
        user = models.User(google_sub=sub, email=info.get("email", ""))
    user.email = info.get("email", user.email)
    user.name = info.get("name")
    user.picture = info.get("picture")
    user.google_access_token = creds.token
    if creds.refresh_token:  # only returned on first consent
        user.google_refresh_token = creds.refresh_token
    user.google_token_expiry = creds.expiry
    user.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(user)
    session.commit()
    session.refresh(user)

    resp = RedirectResponse(settings.post_login_redirect)
    resp.set_cookie(SESSION_COOKIE, make_session_token(user.id), **cookie_kwargs())
    resp.delete_cookie("oauth_state", path="/")
    resp.delete_cookie("oauth_verifier", path="/")
    return resp


@router.post("/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp
