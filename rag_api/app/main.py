"""FastAPI RAG Retrieval API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.routers import query_router, databases_router, health_router
from app.services.database_manager import get_database_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print("Starting RAG API...")
    # Initialize database manager connection
    get_database_manager()
    print("RAG API started successfully")
    
    yield
    
    # Shutdown
    print("Shutting down RAG API...")
    db_manager = get_database_manager()
    db_manager.close()
    print("RAG API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    
    app = FastAPI(
        title=config.api_title,
        version=config.api_version,
        description="RAG Document Retrieval API - Returns relevant documents for queries",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(databases_router)
    app.include_router(query_router)
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

