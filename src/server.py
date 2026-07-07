"""FastAPI server exposing the tiered embedding cache over HTTP.

Endpoints
---------
GET  /embedding/{key} : return the embedding for `key` (cache or compute).
GET  /stats           : return cache statistics.
POST /warm            : preload a list of keys into the cache.
GET  /healthz         : liveness probe.

The cache is not thread-safe, so all cache access is serialized with an asyncio
lock. Embedding computation is offloaded to a thread so it does not block the
event loop.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from .embedder import DEFAULT_DIM, embed
from .tiered_cache import TieredCache

load_dotenv()

DIM = int(os.getenv("EMBED_DIM", DEFAULT_DIM))
HOT_CAPACITY = int(os.getenv("HOT_CAPACITY", "256"))
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join("data", "cold_tier"))


def create_app(
    hot_capacity: int = HOT_CAPACITY,
    cache_dir: str = CACHE_DIR,
    dim: int = DIM,
) -> FastAPI:
    """Build a FastAPI app with its own cache instance.

    Passing explicit args makes the app easy to construct in tests with a
    temporary cache directory.
    """
    app = FastAPI(title="Tiered Embedding Cache Server")
    cache = TieredCache(hot_capacity=hot_capacity, cache_dir=cache_dir, dim=dim)
    lock = asyncio.Lock()

    app.state.cache = cache
    app.state.dim = dim

    async def _get_embedding(key: str):
        # Compute outside the lock to keep the critical section short; the
        # embedder is pure and deterministic so double-compute is harmless.
        async with lock:
            cached = cache.get(key)
        if cached is not None:
            return cached
        vec = await asyncio.to_thread(embed, key, dim)
        async with lock:
            # Another request may have filled it in the meantime; get() again
            # keeps stats honest but we still store our value if absent.
            existing = cache.get(key)
            if existing is not None:
                return existing
            cache.put(key, vec)
        return vec

    class WarmRequest(BaseModel):
        keys: list[str]

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/embedding/{key}")
    async def get_embedding(key: str):
        vec = await _get_embedding(key)
        return {"key": key, "dim": len(vec), "embedding": vec.tolist()}

    @app.get("/stats")
    async def get_stats():
        async with lock:
            return cache.stats_dict()

    @app.post("/warm")
    async def warm(req: WarmRequest):
        for key in req.keys:
            await _get_embedding(key)
        async with lock:
            stats = cache.stats_dict()
        return {"warmed": len(req.keys), "stats": stats}

    return app


# Module-level app for `uvicorn src.server:app`.
app = create_app()
