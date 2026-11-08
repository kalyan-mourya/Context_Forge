# Context Forge

Context Forge is a retrieval-augmented generation (RAG) application for building a document-grounded conversational assistant. The current project is structured as a clean, extensible Python FastAPI app with LLM integration, vector retrieval, and a browser UI.

## Overview

This project follows a standard RAG pattern:

- load configuration and environment settings
- initialize the embedding model and LLM
- build or load a FAISS vector database
- retrieve relevant documents for a query
- combine retrieved context with chat history and prompt instructions
- return a final grounded answer through the API

The codebase is intentionally organized so that the core architecture remains stable while the dataset, prompts, and retrieval sources can be changed for different use cases.

## Current architecture

- `app/main.py` — FastAPI app, startup lifecycle, config loading, and route handlers
- `app/assistant.py` — chat manager and retrieval-based answer orchestration
- `app/settings.py` — application paths and configuration constants
- `app/models/schemas.py` — request/response validation models
- `app/generation/prompt_templates.py` — prompt template loader
- `app/generation/prompt_templates.yaml` — prompt definitions used by the assistant
- `app/src/` — model, embedding, and vector database factories
- `app/data/` — source documents, vector store, and dataset artifacts
- `app/index.html` — UI entry file served by FastAPI

## Runtime behavior

The app starts by loading environment variables and checking for NVIDIA credentials.

- If the NVIDIA API key is configured, the app initializes the embedding model, LLM, and vector database.
- If not configured, it falls back to demo mode so the app can still start and serve requests.

This keeps the service available while still preserving the full RAG path when credentials exist.

## API

### Health check

```bash
GET /
```

Example response:

```json
{"health": "ok"}
```

### Chat endpoint

```bash
POST /chat
```

Request body:

```json
{
  "query": "Explain the lecture topic briefly",
  "last_3_turn": []
}
```

Response:

```json
{
  "response": "..."
}
```

### UI

```bash
GET /ui
```

This serves the browser-based interface for local interaction.

## Project configuration

The project uses configuration from the `app/config` directory and environment variables such as:

```env
NVIDIA_LLM_API=your_key_here
NVIDIA_API_KEY=your_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

The app accepts either `NVIDIA_LLM_API` or `NVIDIA_API_KEY`.

## Running locally

1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start the FastAPI app

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

4. Open the app

```text
http://127.0.0.1:8000/ui
```

## Testing

```bash
python -m pytest test_app.py -q
```

## Design note

The core design supports reuse across domains. The model, prompt set, and dataset sources can be updated without rewriting the service architecture. In other words, the same app pattern can be adapted to different knowledge domains by swapping the retrieval corpus and prompt instructions.

## Summary

Context Forge is a clean, modular RAG application with a FastAPI backend, retrieval pipeline, LLM integration, prompt configuration, and a browser UI. The project is structured for extension and domain adaptation while maintaining a simple and understandable codebase.

  































