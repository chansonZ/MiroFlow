# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""MiroFlow Web API - FastAPI application entry point."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add parent directory for MiroFlow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
dotenv.load_dotenv()

# Configure root logging so INFO messages from our modules are visible in the
# server console.  uvicorn already sets up a handler, so we only set the level
# here without adding extra handlers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("miroflow").setLevel(logging.INFO)
logging.getLogger("web_app").setLevel(logging.INFO)

from .api.dependencies import init_dependencies  # noqa: E402
from .api.routes import configs, health, tasks, uploads  # noqa: E402
from .core.config import config  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    config.sessions_dir.mkdir(parents=True, exist_ok=True)
    config.uploads_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    init_dependencies()
    yield
    # Shutdown - cleanup if needed


app = FastAPI(
    title="MiroFlow API",
    description="REST API for MiroFlow AI Research Agent",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(health.router)
app.include_router(tasks.router)
app.include_router(configs.router)
app.include_router(uploads.router)

# Serve static files (built frontend) if directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists() and any(static_dir.iterdir()):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
async def root():
    """Root endpoint - returns API info or redirects to frontend."""
    return {
        "name": "MiroFlow API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )
