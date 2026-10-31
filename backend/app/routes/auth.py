"""Accounts: sign up, sign in, sign out, profile, password and sessions."""

from __future__ import annotations

import secrets
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db
from app.deps import SESSION_COOKIE, api_error, current_session, current_user
from app.models import AuthSession, User, UserSettings
from app.security import hash_password, hash_token, new_session_token, verify_password
from app.services.serializers import iso

router = APIRouter(prefix="/api", tags=["auth"])

# Failed sign-ins per email: (count, first failure time). Kept in memory on purpose:
# the app runs as one process on the owner's computer.
_failures: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
MAX_FAILURES = 8
FAILURE_WINDOW = 15 * 60


class RegisterBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    language: str | None = Field(default=None, pattern="^(en|sk)$")


class LoginBody(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class ProfileBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    email: EmailStr | None = None


class PasswordBody(BaseModel):
    current_password: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=8, max_length=200)


def user_json(user: User) -> dict:
    return {
        "id": str(user.id),
        "name": user.name,
        "email": user.email,
        "created_at": iso(user.created_at),
    }


async def _start_session(db: AsyncSession, user: User, request: Request, response: Response):
    token = new_session_token()
    days = get_settings().session_days
    db.add(
        AuthSession(
            user_id=user.id,
            token_hash=hash_token(token),
            user_agent=(request.headers.get("user-agent") or "")[:300],
            expires_at=datetime.now(UTC) + timedelta(days=days),
        )
    )
    await db.commit()
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=days * 86400,
        httponly=True,
        samesite="lax",
        secure=False,  # served over http://localhost only
        path="/",
    )


@router.get("/auth/state")
async def auth_state(request: Request, db: AsyncSession = Depends(get_db)):
    has_users = bool(await db.scalar(select(func.count()).select_from(User)))
    user = None
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        session = await db.scalar(
            select(AuthSession).where(AuthSession.token_hash == hash_token(token))
        )
        if session and session.expires_at > datetime.now(UTC):
            row = await db.get(User, session.user_id)
            user = user_json(row) if row else None
    return {"has_users": has_users, "user": user, "auto_login": get_settings().auto_login}


class AutoBody(BaseModel):
    language: str | None = Field(default=None, pattern="^(en|sk)$")


@router.post("/auth/auto")
async def auto_login(
    body: AutoBody, request: Request, response: Response, db: AsyncSession = Depends(get_db)
):
    """No sign-in screen: opens the first account on this Mac, or creates one. The servers
    listen only on 127.0.0.1, foreign Host headers are refused and this POST needs the client
    header, so other websites cannot use it."""
    if not get_settings().auto_login:
        raise api_error(403, "auto_login_off", "Sign in with your email and password.")
    user = await db.scalar(select(User).order_by(User.created_at, User.id).limit(1))
    if user is None:
        user = User(
            email="studio@localhost",
            name="Studio",
            # A random password nobody knows: with auto sign-in none is needed.
            password_hash=hash_password(secrets.token_urlsafe(32)),
        )
        db.add(user)
        await db.flush()
        db.add(UserSettings(user_id=user.id, data={"language": body.language or "en"}))
        await db.commit()
    await _start_session(db, user, request, response)
    return {"user": user_json(user)}


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterBody, request: Request, response: Response, db: AsyncSession = Depends(get_db)
):
    email = body.email.lower()
    if await db.scalar(select(User).where(User.email == email)):
        raise api_error(409, "email_taken", "An account with this email already exists.")
    user = User(email=email, name=body.name.strip(), password_hash=hash_password(body.password))
    db.add(user)
    await db.flush()
    db.add(UserSettings(user_id=user.id, data={"language": body.language or "en"}))
    await db.commit()
    await _start_session(db, user, request, response)
    return {"user": user_json(user)}


@router.post("/auth/login")
async def login(
    body: LoginBody, request: Request, response: Response, db: AsyncSession = Depends(get_db)
):
    email = body.email.lower()
    count, first = _failures[email]
    if count >= MAX_FAILURES and time.monotonic() - first < FAILURE_WINDOW:
        raise api_error(429, "too_many_attempts", "Too many attempts. Try again in 15 minutes.")
    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(body.password, user.password_hash):
        if count == 0 or time.monotonic() - first >= FAILURE_WINDOW:
            _failures[email] = (1, time.monotonic())
        else:
            _failures[email] = (count + 1, first)
        raise api_error(401, "invalid_login", "Wrong email or password.")
    _failures.pop(email, None)
    await _start_session(db, user, request, response)
    return {"user": user_json(user)}


@router.post("/auth/logout")
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        await db.execute(delete(AuthSession).where(AuthSession.token_hash == hash_token(token)))
        await db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/me")
async def me(user: User = Depends(current_user)):
    return {"user": user_json(user)}


@router.patch("/account")
async def update_profile(
    body: ProfileBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    row = await db.get(User, user.id)
    assert row is not None
    if body.email and body.email.lower() != row.email:
        if await db.scalar(select(User).where(User.email == body.email.lower())):
            raise api_error(409, "email_taken", "An account with this email already exists.")
        row.email = body.email.lower()
    if body.name:
        row.name = body.name.strip()
    await db.commit()
    return {"user": user_json(row)}


@router.post("/account/password")
async def change_password(
    body: PasswordBody,
    pair: tuple[User, AuthSession] = Depends(current_session),
    db: AsyncSession = Depends(get_db),
):
    user, session = pair
    row = await db.get(User, user.id)
    assert row is not None
    if not verify_password(body.current_password, row.password_hash):
        raise api_error(400, "wrong_password", "The current password is not correct.")
    row.password_hash = hash_password(body.new_password)
    # Changing the password signs out every other session.
    await db.execute(
        delete(AuthSession).where(AuthSession.user_id == user.id, AuthSession.id != session.id)
    )
    await db.commit()
    return {"ok": True}


@router.get("/account/sessions")
async def list_sessions(
    pair: tuple[User, AuthSession] = Depends(current_session), db: AsyncSession = Depends(get_db)
):
    user, current = pair
    rows = (
        await db.scalars(
            select(AuthSession)
            .where(AuthSession.user_id == user.id)
            .order_by(AuthSession.last_seen_at.desc())
        )
    ).all()
    return {
        "sessions": [
            {
                "id": str(s.id),
                "user_agent": s.user_agent,
                "created_at": iso(s.created_at),
                "last_seen_at": iso(s.last_seen_at),
                "expires_at": iso(s.expires_at),
                "current": s.id == current.id,
            }
            for s in rows
        ]
    }


@router.delete("/account/sessions/{session_id}")
async def revoke_session(
    session_id: uuid.UUID,
    pair: tuple[User, AuthSession] = Depends(current_session),
    db: AsyncSession = Depends(get_db),
):
    user, _ = pair
    await db.execute(
        delete(AuthSession).where(AuthSession.id == session_id, AuthSession.user_id == user.id)
    )
    await db.commit()
    return {"ok": True}


@router.post("/account/sessions/revoke-others")
async def revoke_other_sessions(
    pair: tuple[User, AuthSession] = Depends(current_session), db: AsyncSession = Depends(get_db)
):
    user, current = pair
    await db.execute(
        delete(AuthSession).where(AuthSession.user_id == user.id, AuthSession.id != current.id)
    )
    await db.commit()
    return {"ok": True}
