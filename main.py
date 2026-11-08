from __future__ import annotations

import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .assistant import ChatManager
from .models.schemas import ChatInput, ChatOutput
from .settings import DOCS_PATH, EMBD_MODEL_DIR, MODEL_CONFIG_PATH, VECTOR_STORE_PATH
from .src.embedder.embedder_factory import EMBFactory
from .src.llm.llm_factory import CHATLLMFactory
from .src.vector_database.vector_db_factory import VECTORDBFactory


load_dotenv()
logger = logging.getLogger("uvicorn")

llm: Any = None
embedder_pipe: Any = None
vector_db: Any = None
manager: ChatManager | None = ChatManager(None, None)

NVIDIA_NVC_API_KEY = os.getenv("NVIDIA_LLM_API") or os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")


def load_config(config_path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError("Model config must be a dictionary.")

    return config


async def initialize_runtime() -> ChatManager:
    global llm, embedder_pipe, vector_db, manager

    if not NVIDIA_NVC_API_KEY:
        logger.warning("NVIDIA_LLM_API is not set. Starting in demo mode.")
        manager = ChatManager(None, None)
        return manager

    try:
        start_time = time.perf_counter()
        model_config = load_config()

        chat_llm_args = model_config["chat_llm_args"]
        chat_llm_args["api_key"] = NVIDIA_NVC_API_KEY
        chat_llm_args["base_url"] = NVIDIA_BASE_URL

        db_args = model_config["db_args"]
        db_args["vector_store_path"] = VECTOR_STORE_PATH
        db_args["docs_path"] = DOCS_PATH

        embedder_args = model_config["embedder_args"]
        embedder_args["model_path"] = EMBD_MODEL_DIR

        llm_pipe = CHATLLMFactory.create_chat_model_pipeline(chat_llm_args["type"], **chat_llm_args)
        embedder_pipe = EMBFactory.create_embedder_model_pipeline(embedder_args["type"], **embedder_args)

        db_args["embedding_model"] = await embedder_pipe.load_model()
        vector_db_pipe = VECTORDBFactory.create_vector_db_pipeline(db_args["type"], **db_args)

        llm = await llm_pipe.load_model()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("⚡ Latency (llm initialization): %.2f ms", elapsed_ms)

        start_time = time.perf_counter()
        vector_db = await vector_db_pipe.load_faiss_db()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("⚡ Latency (vector initialization): %.2f ms", elapsed_ms)

        manager = ChatManager(llm, vector_db.as_retriever())
        _ = vector_db.as_retriever().invoke("startup test")
        logger.info("✅ All components initialized successfully.")
        return manager
    except Exception:
        logger.warning("AI backend unavailable during startup: %s", traceback.format_exc())
        logger.warning("Falling back to demo mode so the app can still boot.")
        manager = ChatManager(None, None)
        return manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = await initialize_runtime()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Context Forge", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def health_check() -> dict[str, str]:
        return {"health": "ok"}

    @app.get("/ui")
    def chat_ui() -> FileResponse:
        return FileResponse(Path(__file__).resolve().parent / "index.html")

    @app.post("/chat", response_model=ChatOutput)
    async def get_response(request: ChatInput) -> dict[str, str]:
        if manager is None:
            raise HTTPException(status_code=503, detail="Chat manager is not initialized.")

        try:
            logger.info("📥 Received query: %s", request.query)
            logger.info("📚 Context: %s", request.last_3_turn)

            start_time = time.perf_counter()
            response = await manager.run(request.query, request.last_3_turn)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info("⚡ Latency (model response): %.2f ms", elapsed_ms)
            logger.info("📤 Returning response: %s", response)
            return {"response": response}
        except Exception as exc:
            logger.error("❌ Error processing request: %s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()

