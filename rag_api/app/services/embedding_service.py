"""Embedding service for generating query embeddings via HTTP endpoint."""

from __future__ import annotations

import numpy as np
import requests

from app.config import get_config


class EmbeddingService:
    """Service for generating embeddings via remote HTTP endpoint."""
    
    def __init__(self):
        """Initialize the embedding service with configuration."""
        config = get_config()
        self.url = config.embedding.url
        self.model = config.embedding.model
        self.api_key = config.embedding.api_key
        self.request_timeout_seconds = config.embedding.request_timeout_seconds
        self.health_timeout_seconds = config.embedding.health_timeout_seconds
    
    def get_embeddings(self, texts: str | list[str]) -> np.ndarray:
        """Get embeddings for the given texts.
        
        Args:
            texts: A single text string or list of text strings to embed.
            
        Returns:
            numpy array of embeddings (shape: [num_texts, embedding_dim])
            
        Raises:
            ValueError: If the embedding API request fails.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout_seconds,
        )
        
        if response.status_code != 200:
            raise ValueError(
                f"Embedding API request failed with status code {response.status_code}: {response.text}"
            )
        
        embeddings_data = response.json()
        embeddings = []
        
        for data in embeddings_data.get("data", []):
            embeddings.append(data.get("embedding", []))
        
        return np.array(embeddings, dtype=np.float32)
    
    def check_health(self) -> bool:
        """Check if the embedding service is available.
        
        Returns:
            True if the service is available, False otherwise.
        """
        try:
            # Try a minimal request to check connectivity
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "input": ["test"]
            }
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.health_timeout_seconds,
            )
            return response.status_code == 200
        except Exception:
            return False


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

