"""Database manager for RAG database registry using MongoDB."""

from __future__ import annotations

from typing import Any, Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

from app.config import get_config


class DatabaseManager:
    """Manager for RAG database configurations stored in MongoDB."""
    
    def __init__(self):
        """Initialize the database manager with MongoDB connection."""
        config = get_config()
        self.mongo_url = config.mongodb.url
        self.db_name = config.mongodb.database
        self.collection_name = config.mongodb.collection
        self.client: Optional[MongoClient] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_url)
            # Test the connection
            self.client.admin.command('ping')
            print("Connected to MongoDB successfully")
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            self.client = None
        except Exception as e:
            print(f"Error establishing MongoDB connection: {e}")
            self.client = None
    
    @property
    def db(self):
        """Get the database instance."""
        if self.client is None:
            raise Exception("MongoDB connection not established")
        return self.client[self.db_name]
    
    @property
    def collection(self):
        """Get the RAG collection."""
        return self.db[self.collection_name]
    
    def get_database_config(self, db_name: str) -> Optional[dict[str, Any]]:
        """Get configuration for a specific RAG database.
        
        Args:
            db_name: Name of the RAG database.
            
        Returns:
            Dictionary containing the database configuration, or None if not found.
        """
        try:
            result = self.collection.find_one({'name': db_name})
            if result and '_id' in result:
                result['_id'] = str(result['_id'])
            return result
        except PyMongoError as e:
            raise Exception(f"Database query error: {e}")
    
    def get_all_database_configs(self, db_name: str) -> list[dict[str, Any]]:
        """Get all configurations for a specific RAG database name.
        
        Some databases may have multiple configurations (e.g., tfidf + distllm).
        
        Args:
            db_name: Name of the RAG database.
            
        Returns:
            List of database configurations.
        """
        try:
            results = list(self.collection.find({'name': db_name}))
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            return results
        except PyMongoError as e:
            raise Exception(f"Database query error: {e}")
    
    def list_databases(self, active_only: bool = True) -> list[dict[str, Any]]:
        """List all available RAG databases.
        
        Args:
            active_only: If True, only return active databases.
            
        Returns:
            List of database configurations.
        """
        try:
            query = {'active': True} if active_only else {}
            results = list(self.collection.find(query).sort('priority', 1))
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            return results
        except PyMongoError as e:
            raise Exception(f"Database query error: {e}")
    
    def check_connection(self) -> bool:
        """Check if MongoDB connection is healthy.
        
        Returns:
            True if connected, False otherwise.
        """
        try:
            if self.client is None:
                return False
            self.client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None


# Singleton instance
_database_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Get the singleton database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager

