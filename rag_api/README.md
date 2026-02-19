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
  "embedding_url": "your_embedding_service_url",
  "embedding_model": "your_model_name",
  "embedding_apiKey": "your_api_key",
  "mongodb_url": "your_mongodb_connection_string",
  "mongodb_database": "copilot",
  "mongodb_collection": "ragList"
}
```

3. Run the application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
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

