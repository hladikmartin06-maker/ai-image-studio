"""Request dependencies: the signed-in user and the CSRF check."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import AuthSession, User
from app.security import hash_token

SESSION_COOKIE = "studio_session"
# Every state-changing request must carry this header. Browsers cannot add a custom
# header to a cross-site form post, and the API sends no CORS headers, so other sites
# cannot make the signed-in browser change anything.
CSRF_HEADER = "x-studio-client"
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


async def csrf_guard(request: Request) -> None:
    if request.method in UNSAFE_METHODS and request.headers.get(CSRF_HEADER) != "web":
        raise api_error(status.HTTP_403_FORBIDDEN, "csrf", "Missing client header.")


async def current_session(
    request: Request, db: AsyncSession = Depends(get_db)
) -> tuple[User, AuthSession]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise api_error(status.HTTP_401_UNAUTHORIZED, "unauthenticated", "Please sign in.")
    session = await db.scalar(
        select(AuthSession).where(AuthSession.token_hash == hash_token(token))
    )
    now = datetime.now(UTC)
    if session is None or session.expires_at < now:
        raise api_error(status.HTTP_401_UNAUTHORIZED, "unauthenticated", "Please sign in.")
    user = await db.get(User, session.user_id)
    if user is None:
        raise api_error(status.HTTP_401_UNAUTHORIZED, "unauthenticated", "Please sign in.")
    if now - session.last_seen_at > timedelta(minutes=5):
        session.last_seen_at = now
        await db.commit()
    return user, session


async def current_user(
    pair: tuple[User, AuthSession] = Depends(current_session),
) -> User:
    return pair[0]
