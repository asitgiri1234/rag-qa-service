"""Aggregated metrics.

Percentiles are computed in plain Python over the JSONL record file -- no pandas,
no prometheus client. The point is a real aggregate the reviewer can read, not a
scrape endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.limits import READ_LIMIT, limiter
from app.config import Settings, get_settings
from app.core import metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


@router.get("/summary", summary="Aggregated query and ingestion metrics")
@limiter.limit(READ_LIMIT)
def summary(request: Request) -> dict:
    return metrics.summarize(_settings(request).metrics_path)
