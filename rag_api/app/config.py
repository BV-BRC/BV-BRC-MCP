"""Configuration management for RAG API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding service."""
    
    url: str = Field(..., description="URL of the embedding endpoint")
    model: str = Field(..., description="Model name for embeddings")
    api_key: str = Field(..., description="API key for authentication")
    request_timeout_seconds: int = Field(default=30, description="Embedding request timeout")
    health_timeout_seconds: int = Field(default=5, description="Embedding health check timeout")


class MongoDBConfig(BaseModel):
    """Configuration for MongoDB connection."""
    
    url: str = Field(..., description="MongoDB connection URL")
    database: str = Field(default="copilot", description="Database name")
    collection: str = Field(default="ragList", description="Collection name for RAG configs")


class RuntimeConfig(BaseModel):
    """Configuration for runtime/server behavior."""

    host: str = Field(default="0.0.0.0", description="Host interface to bind")
    port: int = Field(default=8000, description="Port to bind")
    reload: bool = Field(default=False, description="Enable hot reload")
    log_level: str = Field(default="info", description="Uvicorn log level")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval defaults."""

    default_top_k: int = Field(default=10, ge=1, le=100, description="Default top-k")
    default_score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Default minimum similarity score",
    )


class AppConfig(BaseModel):
    """Main application configuration."""
    
    embedding: EmbeddingConfig
    mongodb: MongoDBConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api_title: str = Field(default="RAG Retrieval API")
    api_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to the config file. If None, looks for config.json
                     in the project root directory.
    
    Returns:
        AppConfig instance with loaded configuration.
    """
    if config_path is None:
        # Look for config.json in the project root (parent of app/)
        config_path = str(Path(__file__).parent.parent / "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Transform flat config to nested structure
    app_config = AppConfig(
        embedding=EmbeddingConfig(
            url=config_data.get("embedding_url"),
            model=config_data.get("embedding_model"),
            api_key=config_data.get("embedding_apiKey"),
            request_timeout_seconds=config_data.get("embedding_request_timeout_seconds", 30),
            health_timeout_seconds=config_data.get("embedding_health_timeout_seconds", 5),
        ),
        mongodb=MongoDBConfig(
            url=config_data.get("mongodb_url"),
            database=config_data.get("mongodb_database", "copilot"),
            collection=config_data.get("mongodb_collection", "ragList"),
        ),
        runtime=RuntimeConfig(
            host=config_data.get("host", "0.0.0.0"),
            port=config_data.get("port", 8000),
            reload=config_data.get("reload", False),
            log_level=config_data.get("log_level", "info"),
        ),
        retrieval=RetrievalConfig(
            default_top_k=config_data.get("default_top_k", 10),
            default_score_threshold=config_data.get("default_score_threshold", 0.0),
        ),
        api_title=config_data.get("api_title", "RAG Retrieval API"),
        api_version=config_data.get("api_version", "1.0.0"),
        debug=config_data.get("debug", False),
    )
    
    return app_config


# Global config instance (loaded on import)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

