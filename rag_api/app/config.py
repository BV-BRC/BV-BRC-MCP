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


class MongoDBConfig(BaseModel):
    """Configuration for MongoDB connection."""
    
    url: str = Field(..., description="MongoDB connection URL")
    database: str = Field(default="copilot", description="Database name")
    collection: str = Field(default="ragList", description="Collection name for RAG configs")


class AppConfig(BaseModel):
    """Main application configuration."""
    
    embedding: EmbeddingConfig
    mongodb: MongoDBConfig
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
        ),
        mongodb=MongoDBConfig(
            url=config_data.get("mongodb_url"),
            database=config_data.get("mongodb_database", "copilot"),
            collection=config_data.get("mongodb_collection", "ragList"),
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

