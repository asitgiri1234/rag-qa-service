"""FastAPI application factory."""

from fastapi import FastAPI

from app import __version__
from app.config import get_settings
from app.models.health import HealthResponse


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title="RAG QA Service",
        version=__version__,
    )
    application.state.settings = settings

    @application.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    # Routers from app.api are registered here as they land.

    return application


app = create_app()
