# RAG API

A FastAPI-based Retrieval-Augmented Generation (RAG) API for document retrieval using vector embeddings.

## Overview

This API provides endpoints for querying document databases using semantic search. It uses vector embeddings and FAISS for similarity search to retrieve relevant documents based on user queries.

## Features

- Document querying with semantic search
- Database management endpoints
- Vector similarity search using FAISS
- MongoDB integration for document storage
- Embedding service integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the application by creating a `config.json` file in the project root with the following structure:
```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "reload": false,
  "log_level": "info",
  "embedding_url": "your_embedding_service_url",
  "embedding_model": "your_model_name",
  "embedding_apiKey": "your_api_key",
  "embedding_request_timeout_seconds": 30,
  "embedding_health_timeout_seconds": 5,
  "mongodb_url": "your_mongodb_connection_string",
  "mongodb_database": "copilot",
  "mongodb_collection": "ragList",
  "default_top_k": 10,
  "default_score_threshold": 0.0
}
```

3. Run the application:
```bash
./startup.sh
```

## API Endpoints

- `/health` - Health check endpoint
- `/databases` - Database management endpoints
- `/query/{database_name}` - Query a RAG database for relevant documents

## Technologies

- FastAPI
- MongoDB
- FAISS
- Pydantic
- Uvicorn

