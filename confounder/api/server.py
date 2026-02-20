"""FastAPI server application factory."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from confounder import __version__
from confounder.api.routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Confounder API",
        description="Detect hidden confounders in observational studies.",
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, str]:
        """Check system health."""
        return {"status": "ok", "version": __version__}

    logger.info("Confounder API created (v%s)", __version__)
    return app


app = create_app()
