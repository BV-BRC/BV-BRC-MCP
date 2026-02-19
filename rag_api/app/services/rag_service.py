"""Core RAG retrieval service."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
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
        embeddings_path = data.get('embeddings_path', '')
        vectorizer_path = data.get('vectorizer_path', '')
        dataset_dir = data.get('dataset_dir', '')
        # Include all supported storage paths to avoid cache key collisions.
        return f"{db_name}_{program}_{dataset_dir}_{faiss_path}_{embeddings_path}_{vectorizer_path}"
    
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
            print(f"[RAG Service] Using cached index for database '{db_name}' (cache_key: {cache_key})")
            return self._loaded_indexes[cache_key]
        
        data = config.get('data', {})
        dataset_dir = data.get('dataset_dir')
        faiss_index_path = data.get('faiss_index_path')
        
        if not dataset_dir or not faiss_index_path:
            raise ValueError(f"Missing dataset_dir or faiss_index_path in config for {db_name}")
        
        program = config.get('program', 'unknown')
        print(f"[RAG Service] Loading database '{db_name}' (program: {program})...")
        print(f"[RAG Service]   - Dataset directory: {dataset_dir}")
        print(f"[RAG Service]   - FAISS index path: {faiss_index_path}")
        
        # Load the dataset
        print(f"[RAG Service]   - Loading dataset from disk...")
        dataset = Dataset.load_from_disk(dataset_dir)
        print(f"[RAG Service]   - Dataset loaded: {len(dataset)} documents")
        
        # Load the FAISS index
        print(f"[RAG Service]   - Loading FAISS index from disk...")
        faiss_index = faiss.read_index(faiss_index_path)
        print(f"[RAG Service]   - FAISS index loaded: {faiss_index.ntotal} vectors")
        
        self._loaded_indexes[cache_key] = {
            'index': faiss_index,
            'dataset': dataset,
        }
        
        print(f"[RAG Service] Database '{db_name}' loaded successfully")
        
        return self._loaded_indexes[cache_key]

    def _load_tfidf_data(self, db_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Load or get cached TF-IDF vectorizer and embeddings for a database configuration."""
        cache_key = self._get_cache_key(db_name, config)

        if cache_key in self._loaded_indexes:
            print(f"[RAG Service] Using cached TF-IDF data for database '{db_name}' (cache_key: {cache_key})")
            return self._loaded_indexes[cache_key]

        data = config.get('data', {})
        embeddings_path = data.get('embeddings_path')
        vectorizer_path = data.get('vectorizer_path')

        if not embeddings_path or not vectorizer_path:
            raise ValueError(f"Missing embeddings_path or vectorizer_path in config for {db_name}")

        print(f"[RAG Service] Loading TF-IDF database '{db_name}'...")
        print(f"[RAG Service]   - Embeddings path: {embeddings_path}")
        print(f"[RAG Service]   - Vectorizer path: {vectorizer_path}")

        vectorizer_file = Path(vectorizer_path) / "vectorizer_components.arrow"
        if not vectorizer_file.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer file not found: {vectorizer_file}")

        with pa.memory_map(str(vectorizer_file), 'r') as source:
            vectorizer_table = pa.ipc.open_file(source).read_all()

        if 'vocabulary' not in vectorizer_table.column_names or 'idf_values' not in vectorizer_table.column_names:
            raise ValueError("TF-IDF vectorizer file missing 'vocabulary' and/or 'idf_values' columns")

        vocabulary = vectorizer_table.column('vocabulary').to_pylist()
        idf_values = vectorizer_table.column('idf_values').to_pylist()
        vocabulary_index = {term: i for i, term in enumerate(vocabulary)}
        idf_array = np.array(idf_values, dtype=np.float32)

        embedding_files = sorted(Path(embeddings_path).glob("tfidf_embeddings_batch_*.arrow"))
        if not embedding_files:
            raise FileNotFoundError(f"No tfidf_embeddings_batch_*.arrow files found in {embeddings_path}")

        batch_tables = []
        for batch_file in embedding_files:
            print(f"[RAG Service]   - Loading TF-IDF embedding batch: {batch_file.name}")
            with pa.memory_map(str(batch_file), 'r') as source:
                batch_tables.append(pa.ipc.open_file(source).read_all())

        embeddings_table = pa.concat_tables(batch_tables, promote_options="default")
        if 'embedding' not in embeddings_table.column_names:
            raise ValueError("TF-IDF embeddings table missing 'embedding' column")

        embedding_matrix = np.array(embeddings_table.column('embedding').to_pylist(), dtype=np.float32)
        if embedding_matrix.size == 0:
            raise ValueError(f"TF-IDF embedding matrix is empty for {db_name}")

        embedding_dim = embedding_matrix.shape[1]
        tfidf_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embedding_matrix)
        tfidf_index.add(embedding_matrix)

        self._loaded_indexes[cache_key] = {
            'type': 'tfidf',
            'index': tfidf_index,
            'table': embeddings_table,
            'vocabulary_index': vocabulary_index,
            'idf_values': idf_array,
        }

        print(f"[RAG Service] TF-IDF database '{db_name}' loaded successfully ({embeddings_table.num_rows} rows)")
        return self._loaded_indexes[cache_key]

    def _encode_tfidf_query(
        self,
        query: str,
        vocabulary_index: dict[str, int],
        idf_values: np.ndarray,
    ) -> np.ndarray:
        """Encode query text into a TF-IDF vector compatible with stored embeddings."""
        tokens = re.findall(r"(?u)\b\w\w+\b", query.lower())
        token_counts = Counter(tokens)

        query_vector = np.zeros((1, len(vocabulary_index)), dtype=np.float32)
        for token, count in token_counts.items():
            idx = vocabulary_index.get(token)
            if idx is None:
                continue
            query_vector[0, idx] = float(count) * float(idf_values[idx])

        faiss.normalize_L2(query_vector)
        return query_vector
    
    def _search_single_config(
        self,
        db_name: str,
        config: dict[str, Any],
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        score_threshold: float,
    ) -> list[dict[str, Any]]:
        """Search a single database configuration.
        
        Args:
            db_name: Database name.
            config: Database configuration.
            query_text: Query string.
            query_embedding: Pre-computed distllm query embedding.
            top_k: Number of documents to retrieve per config.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of document dictionaries.
        """
        program = config.get('program', 'unknown')
        
        if program == 'distllm':
            # Load the FAISS index
            index_data = self._load_faiss_index(db_name, config)
            faiss_index = index_data['index']
            dataset = index_data['dataset']

            # Search
            print(f"[RAG Service]   - Performing distllm FAISS search (top_k={top_k}, score_threshold={score_threshold})...")
            scores, indices = faiss_index.search(query_embedding, top_k)

            # Build results
            documents = []
            for score, idx in zip(scores[0], indices[0]):
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

        if program == 'tfidf':
            index_data = self._load_tfidf_data(db_name, config)
            tfidf_index = index_data['index']
            table = index_data['table']
            vocabulary_index = index_data['vocabulary_index']
            idf_values = index_data['idf_values']

            query_vector = self._encode_tfidf_query(
                query=query_text,
                vocabulary_index=vocabulary_index,
                idf_values=idf_values,
            )

            print(f"[RAG Service]   - Performing TF-IDF search (top_k={top_k}, score_threshold={score_threshold})...")
            scores, indices = tfidf_index.search(query_vector, top_k)

            documents = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if score < score_threshold:
                    continue

                row_idx = int(idx)
                content = table.column('text')[row_idx].as_py() if 'text' in table.column_names else ''
                doc = {
                    'content': content,
                    'score': float(score),
                    'metadata': {
                        'program': program,
                        'config_name': db_name,
                    },
                }

                for key in table.column_names:
                    if key not in ('text', 'embedding'):
                        doc['metadata'][key] = table.column(key)[row_idx].as_py()

                documents.append(doc)

            return documents

        print(f"[RAG Service]   - Skipping unsupported program: {program}")
        return []

    def preload_active_indexes(self) -> dict[str, int]:
        """Preload indexes for active distllm and TF-IDF configurations.

        Returns:
            Summary dictionary with counts for loaded/skipped/failed configurations.
        """
        print("[RAG Service] Starting preload of active indexes...")
        configs = self.database_manager.list_databases(active_only=True)
        print(f"[RAG Service] Found {len(configs)} active configuration(s)")

        loaded = 0
        skipped = 0
        failed = 0

        for config in configs:
            db_name = config.get("name")
            program = config.get("program", "unknown")

            if not db_name:
                print("[RAG Service]   - Skipping config with missing database name")
                skipped += 1
                continue

            if program not in {"distllm", "tfidf"}:
                print(f"[RAG Service]   - Skipping unsupported config '{db_name}' (program: {program})")
                skipped += 1
                continue

            try:
                if program == "distllm":
                    self._load_faiss_index(db_name, config)
                else:
                    self._load_tfidf_data(db_name, config)
                loaded += 1
            except Exception as exc:
                print(f"[RAG Service]   - Failed to preload '{db_name}': {exc}")
                failed += 1

        print(
            f"[RAG Service] Preload complete: loaded={loaded}, skipped={skipped}, failed={failed}"
        )
        return {"loaded": loaded, "skipped": skipped, "failed": failed}
    
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
        print(f"[RAG Service] Starting search in database '{db_name}' with query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        # Get all database configurations
        configs = self.database_manager.get_all_database_configs(db_name)
        
        if not configs:
            raise ValueError(f"No configuration found for database '{db_name}'")
        
        print(f"[RAG Service] Found {len(configs)} configuration(s) for database '{db_name}'")
        
        # Get the query embedding once (shared across all configs)
        print(f"[RAG Service] Generating query embedding...")
        query_embedding = self.embedding_service.get_embeddings(query)
        
        # Normalize for inner product search
        faiss.normalize_L2(query_embedding)
        print(f"[RAG Service] Query embedding generated (dimension: {query_embedding.shape[1]})")
        
        # Search all configurations
        all_documents = []
        for i, config in enumerate(configs, 1):
            program = config.get('program', 'unknown')
            print(f"[RAG Service] Searching configuration {i}/{len(configs)} (program: {program})...")
            documents = self._search_single_config(
                db_name=db_name,
                config=config,
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            print(f"[RAG Service] Configuration {i}/{len(configs)} returned {len(documents)} document(s)")
            all_documents.extend(documents)
        
        # Sort by score (descending) and limit to top_k total if needed
        all_documents.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"[RAG Service] Search complete: {len(all_documents)} total document(s) found")
        
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

