"""AI Image Studio API (FastAPI). Listens on 127.0.0.1 only; the Next.js app proxies /api."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.advanced import engine as advanced_engine
from app.deps import csrf_guard
from app.routes import (
    advanced,
    ai,
    auth,
    comfy,
    faces,
    generations,
    gpu,
    images,
    integrations,
    jobs,
    loras,
    misc,
    personas,
    references,
    secrets,
    snippets,
    uploads,
)
from app.security import SecretKeyMissing
from app.services.gpu_runner import gpu_runner
from app.services.worker import worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs full request URLs at INFO; keep them out of the log (they hold task ids only,
# but the rule is simple: no provider traffic in the logs).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)  # used by the anthropic SDK


@asynccontextmanager
async def lifespan(_: FastAPI):
    await worker.start()
    await gpu_runner.start()
    await advanced_engine.recover()
    try:
        yield
    finally:
        await gpu_runner.stop()
        await worker.stop()


app = FastAPI(
    title="AI Image Studio API",
    version=misc.VERSION,
    lifespan=lifespan,
    dependencies=[Depends(csrf_guard)],
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        body = {"error": detail}
    else:
        body = {"error": {"code": f"http_{exc.status_code}", "message": str(detail)}}
    return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", ())[1:]) or "request"
    message = f"{field}: {first.get('msg', 'invalid value')}"
    return JSONResponse(
        {"error": {"code": "invalid_parameter", "message": message, "field": field}},
        status_code=422,
    )


@app.exception_handler(SecretKeyMissing)
async def secret_missing(_: Request, exc: SecretKeyMissing):
    return JSONResponse(
        {"error": {"code": "server_misconfigured", "message": str(exc)}}, status_code=500
    )


for module in (
    misc,
    auth,
    integrations,
    references,
    personas,
    generations,
    images,
    ai,
    gpu,
    uploads,
    secrets,
    comfy,
    loras,
    jobs,
    snippets,
    faces,
    advanced,
):
    app.include_router(module.router)
