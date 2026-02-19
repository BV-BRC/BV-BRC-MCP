"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.database_manager import get_database_manager
from app.services.embedding_service import get_embedding_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health status of the API and its dependencies.
    
    Returns:
        HealthResponse with status of API and connected services.
    """
    db_manager = get_database_manager()
    embedding_service = get_embedding_service()
    
    mongodb_connected = db_manager.check_connection()
    embedding_available = embedding_service.check_health()
    
    # Overall status is "healthy" only if all services are up
    if mongodb_connected and embedding_available:
        status = "healthy"
    elif mongodb_connected or embedding_available:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        mongodb_connected=mongodb_connected,
        embedding_service_available=embedding_available,
    )

