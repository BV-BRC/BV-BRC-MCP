"""Pydantic schemas for request/response models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for document retrieval queries."""
    
    query: str = Field(..., description="The search query text")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of documents to retrieve (uses config default if omitted)",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (uses config default if omitted)",
    )


class DocumentResult(BaseModel):
    """A single retrieved document."""
    
    content: str = Field(..., description="Document text content")
    score: float = Field(..., description="Similarity score")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata")


class QueryResponse(BaseModel):
    """Response model for document retrieval queries."""
    
    query: str = Field(..., description="The original query")
    database: str = Field(..., description="The database queried")
    documents: list[DocumentResult] = Field(..., description="Retrieved documents")
    embedding: Optional[list[float]] = Field(default=None, description="Query embedding vector")
    total_results: int = Field(..., description="Number of documents returned")


class DatabaseInfo(BaseModel):
    """Information about a RAG database."""
    
    name: str = Field(..., description="Database name/identifier")
    program: str = Field(..., description="RAG program type (distllm, tfidf, etc.)")
    active: bool = Field(default=True, description="Whether the database is active")
    description: Optional[str] = Field(default=None, description="Database description")
    data: Optional[dict[str, Any]] = Field(default=None, description="Database configuration data")


class DatabaseListResponse(BaseModel):
    """Response model for listing databases."""
    
    databases: list[DatabaseInfo] = Field(..., description="List of available databases")
    total: int = Field(..., description="Total number of databases")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="API status")
    mongodb_connected: bool = Field(..., description="MongoDB connection status")
    embedding_service_available: bool = Field(..., description="Embedding service status")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(default=None, description="Additional error details")

