"""Consistent error envelope.

Every error leaves the service as ``{"error": {"type": ..., "message": ...}}``.
Unhandled exceptions are logged in full and reported as a generic message: a
stack trace in a response body is an information leak.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)


def envelope(error_type: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": error_type, "message": message}},
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(request: Request, exc: RateLimitExceeded):
        # Structured JSON rather than slowapi's default HTML-ish body.
        return envelope(
            "rate_limit_exceeded",
            f"rate limit exceeded: {exc.detail}",
            status.HTTP_429_TOO_MANY_REQUESTS,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return envelope(_slug(exc.status_code), str(exc.detail), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return envelope(
            "validation_error",
            "; ".join(
                f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}"
                for err in exc.errors()
            )
            or "request validation failed",
            422,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return envelope(
            "internal_error",
            "an internal error occurred",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _slug(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "rate_limit_exceeded",
        503: "service_unavailable",
    }.get(status_code, "error")
