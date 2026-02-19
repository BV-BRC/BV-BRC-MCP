"""Database management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import DatabaseInfo, DatabaseListResponse, ErrorResponse
from app.services.database_manager import get_database_manager

router = APIRouter(prefix="/databases", tags=["databases"])


@router.get(
    "",
    response_model=DatabaseListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_databases(
    active_only: bool = Query(default=True, description="Only return active databases"),
) -> DatabaseListResponse:
    """List all available RAG databases.
    
    Args:
        active_only: If True, only return active databases.
        
    Returns:
        DatabaseListResponse with list of available databases.
    """
    try:
        db_manager = get_database_manager()
        configs = db_manager.list_databases(active_only=active_only)
        
        # Return all configurations (same database name can have multiple configs)
        databases = []
        
        for config in configs:
            databases.append(
                DatabaseInfo(
                    name=config.get('name'),
                    program=config.get('program', 'unknown'),
                    active=config.get('active', True),
                    description=config.get('description'),
                    data=config.get('data'),
                )
            )
        
        return DatabaseListResponse(
            databases=databases,
            total=len(databases),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing databases: {str(e)}")


@router.get(
    "/{database_name}",
    response_model=DatabaseInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Database not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_database(database_name: str) -> DatabaseInfo:
    """Get information about a specific RAG database.
    
    Args:
        database_name: Name of the RAG database.
        
    Returns:
        DatabaseInfo with database details.
    """
    try:
        db_manager = get_database_manager()
        config = db_manager.get_database_config(database_name)
        
        if config is None:
            raise HTTPException(status_code=404, detail=f"Database '{database_name}' not found")
        
        return DatabaseInfo(
            name=config.get('name'),
            program=config.get('program', 'unknown'),
            active=config.get('active', True),
            description=config.get('description'),
            data=config.get('data'),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database: {str(e)}")

