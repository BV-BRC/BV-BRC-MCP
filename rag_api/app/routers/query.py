"""Query endpoints for document retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, DocumentResult, ErrorResponse
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "/{database_name}",
    response_model=QueryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Database not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def query_database(database_name: str, request: QueryRequest) -> QueryResponse:
    """Query a RAG database for relevant documents.
    
    Args:
        database_name: Name of the RAG database to query.
        request: Query request containing the search query and parameters.
        
    Returns:
        QueryResponse with retrieved documents and query embedding.
    """
    try:
        rag_service = get_rag_service()
        
        result = rag_service.search(
            db_name=database_name,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        
        documents = [
            DocumentResult(
                content=doc['content'],
                score=doc['score'],
                metadata=doc.get('metadata'),
            )
            for doc in result['documents']
        ]
        
        return QueryResponse(
            query=request.query,
            database=database_name,
            documents=documents,
            embedding=result.get('embedding'),
            total_results=len(documents),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

