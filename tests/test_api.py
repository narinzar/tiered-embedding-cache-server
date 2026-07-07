"""API tests using FastAPI's TestClient.

These build an app with a temporary cache directory so nothing leaks between
runs, then check response shapes and that stats change as expected.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.server import create_app

DIM = 32


@pytest.fixture()
def client(tmp_path):
    app = create_app(hot_capacity=4, cache_dir=str(tmp_path / "cold"), dim=DIM)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_embedding_shape(client):
    r = client.get("/embedding/hello")
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "hello"
    assert body["dim"] == DIM
    assert isinstance(body["embedding"], list)
    assert len(body["embedding"]) == DIM
    # Deterministic embedder: same key -> identical vector.
    r2 = client.get("/embedding/hello")
    assert r2.json()["embedding"] == body["embedding"]


def test_embedding_is_normalized(client):
    r = client.get("/embedding/some-non-empty-key")
    emb = r.json()["embedding"]
    norm = sum(x * x for x in emb) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_stats_change_on_access(client):
    s0 = client.get("/stats").json()
    assert s0["lookups"] == 0

    # First access: a miss + compute + store.
    client.get("/embedding/alpha")
    s1 = client.get("/stats").json()
    assert s1["misses"] >= 1

    # Second access to the same key: a hit.
    client.get("/embedding/alpha")
    s2 = client.get("/stats").json()
    assert s2["hits"] >= 1
    assert s2["lookups"] == s2["hits"] + s2["misses"]


def test_warm_endpoint(client):
    keys = [f"warm-{i}" for i in range(10)]
    r = client.post("/warm", json={"keys": keys})
    assert r.status_code == 200
    body = r.json()
    assert body["warmed"] == len(keys)
    assert "stats" in body

    # Hot capacity is 4, so some warmed keys spill to the cold tier.
    stats = client.get("/stats").json()
    assert stats["hot_size"] <= 4
    assert stats["distinct_keys"] >= len(keys)
