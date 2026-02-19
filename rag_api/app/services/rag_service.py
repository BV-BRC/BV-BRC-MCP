"""Core RAG retrieval service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
from datasets import Dataset

from app.services.embedding_service import get_embedding_service
from app.services.database_manager import get_database_manager


class RAGService:
    """Service for RAG document retrieval."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.embedding_service = get_embedding_service()
        self.database_manager = get_database_manager()
        self._loaded_indexes: dict[str, dict[str, Any]] = {}
    
    def _get_cache_key(self, db_name: str, config: dict[str, Any]) -> str:
        """Generate a unique cache key for a database configuration.
        
        Args:
            db_name: Database name.
            config: Database configuration from MongoDB.
            
        Returns:
            Unique cache key string.
        """
        program = config.get('program', 'unknown')
        data = config.get('data', {})
        faiss_path = data.get('faiss_index_path', '')
        # Use db_name, program, and faiss path to create unique key
        return f"{db_name}_{program}_{faiss_path}"
    
    def _load_faiss_index(self, db_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Load or get cached FAISS index for a database configuration.
        
        Args:
            db_name: Database name.
            config: Database configuration from MongoDB.
            
        Returns:
            Dictionary with 'index' (FAISS index) and 'dataset' (HF Dataset).
        """
        cache_key = self._get_cache_key(db_name, config)
        
        if cache_key in self._loaded_indexes:
            return self._loaded_indexes[cache_key]
        
        data = config.get('data', {})
        dataset_dir = data.get('dataset_dir')
        faiss_index_path = data.get('faiss_index_path')
        
        if not dataset_dir or not faiss_index_path:
            raise ValueError(f"Missing dataset_dir or faiss_index_path in config for {db_name}")
        
        # Load the dataset
        dataset = Dataset.load_from_disk(dataset_dir)
        
        # Load the FAISS index
        faiss_index = faiss.read_index(faiss_index_path)
        
        self._loaded_indexes[cache_key] = {
            'index': faiss_index,
            'dataset': dataset,
        }
        
        return self._loaded_indexes[cache_key]
    
    def _search_single_config(
        self,
        db_name: str,
        config: dict[str, Any],
        query_embedding: np.ndarray,
        top_k: int,
        score_threshold: float,
    ) -> list[dict[str, Any]]:
        """Search a single database configuration.
        
        Args:
            db_name: Database name.
            config: Database configuration.
            query_embedding: Pre-computed query embedding.
            top_k: Number of documents to retrieve per config.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of document dictionaries.
        """
        program = config.get('program', 'unknown')
        
        # Only process distllm configs for now (FAISS-based)
        if program != 'distllm':
            return []
        
        # Load the FAISS index
        index_data = self._load_faiss_index(db_name, config)
        faiss_index = index_data['index']
        dataset = index_data['dataset']
        
        # Search
        scores, indices = faiss_index.search(query_embedding, top_k)
        
        # Build results
        documents = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            if score < score_threshold:
                continue
            
            doc_data = dataset[int(idx)]
            doc = {
                'content': doc_data.get('text', ''),
                'score': float(score),
                'metadata': {
                    'program': program,
                    'config_name': db_name,
                },
            }
            
            # Add any additional metadata fields
            for key in dataset.column_names:
                if key not in ('text', 'embeddings'):
                    doc['metadata'][key] = doc_data.get(key)
            
            documents.append(doc)
        
        return documents
    
    def search(
        self,
        db_name: str,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Search for relevant documents in a RAG database.
        
        Performs RAG on all entries with the given database name and combines results.
        
        Args:
            db_name: Name of the RAG database to search.
            query: The search query text.
            top_k: Number of documents to retrieve per configuration.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            Dictionary containing 'documents' (list of dicts) and 'embedding' (query embedding).
        """
        # Get all database configurations
        configs = self.database_manager.get_all_database_configs(db_name)
        
        if not configs:
            raise ValueError(f"No configuration found for database '{db_name}'")
        
        # Get the query embedding once (shared across all configs)
        query_embedding = self.embedding_service.get_embeddings(query)
        
        # Normalize for inner product search
        faiss.normalize_L2(query_embedding)
        
        # Search all configurations
        all_documents = []
        for config in configs:
            documents = self._search_single_config(
                db_name=db_name,
                config=config,
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            all_documents.extend(documents)
        
        # Sort by score (descending) and limit to top_k total if needed
        all_documents.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'documents': all_documents,
            'embedding': query_embedding[0].tolist(),
        }


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get the singleton RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

