from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.utils.logging import logger

app = FastAPI(
    title="LLM Hallucination Detection and RAG Validation System",
    description="A modular system for RAG retrieval, LLM generation, and hallucination detection.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health_check() -> dict:
    logger.info("Health check requested")
    return {"status": "ok"}
