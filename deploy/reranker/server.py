"""BGE-reranker-v2-m3 HTTP server — OpenAI-compatible /v1/rerank endpoint.

Exposes:
  POST /v1/rerank        — rerank a query against a list of documents
  GET  /v1/models         — list available models
  GET  /health            — health check

API is compatible with the Cohere / OpenAI reranking format:
  Request:  {"query": str, "documents": [str], "top_n": int}
  Response: {"results": [{"index": int, "relevance_score": float}]}
"""

import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── FlagEmbedding ──────────────────────────────────────────────────────────────

from FlagEmbedding import FlagReranker

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("MODEL_ID", "BAAI/bge-reranker-v2-m3")
PORT = int(os.environ.get("PORT", "80"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
USE_FP16 = os.environ.get("USE_FP16", "true").lower() == "true"

# ── Model singleton ────────────────────────────────────────────────────────────

_reranker: FlagReranker | None = None


def _load_model() -> FlagReranker:
    global _reranker
    print(f"Loading reranker model: {MODEL_ID} (fp16={USE_FP16}, max_length={MAX_LENGTH})")
    t0 = time.monotonic()
    _reranker = FlagReranker(
        MODEL_ID,
        use_fp16=USE_FP16,
        max_length=MAX_LENGTH,
    )
    elapsed = time.monotonic() - t0
    print(f"Model loaded in {elapsed:.1f}s")
    return _reranker


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="BGE Reranker", version="1.0.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = None
    return_documents: bool = False


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    results: list[RerankResult]
    query: str
    model: str


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "BAAI"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/v1/rerank")
def rerank(req: RerankRequest) -> RerankResponse:
    if not _reranker:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not req.documents:
        return RerankResponse(results=[], query=req.query, model=MODEL_ID)

    # FlagEmbedding expects list of [query, document] pairs
    pairs = [[req.query, doc] for doc in req.documents]
    scores = _reranker.compute_score(pairs, normalize=True)

    # Build indexed results
    indexed = list(enumerate(scores))
    # Sort by score descending
    indexed.sort(key=lambda x: x[1], reverse=True)

    top_n = req.top_n if req.top_n is not None else len(req.documents)
    top_n = min(top_n, len(req.documents))

    results = []
    for idx, score in indexed[:top_n]:
        result = RerankResult(
            index=idx,
            relevance_score=score,
            document=req.documents[idx] if req.return_documents else None,
        )
        results.append(result)

    return RerankResponse(results=results, query=req.query, model=MODEL_ID)


@app.get("/v1/models")
def list_models() -> ModelsResponse:
    return ModelsResponse(
        data=[
            ModelInfo(id=MODEL_ID),
        ]
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, workers=1)
