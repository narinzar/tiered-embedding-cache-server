"""Synthetic benchmark for the tiered embedding cache.

Generates a Zipfian access pattern over N distinct keys and replays it through
the cache at several hot-tier capacities, measuring hit rate and latency. For
each capacity it runs both the frequency-aware policy (this cache) and a
pure-LRU baseline so the effect of promotion/demotion is visible.

Nothing here fabricates numbers: it measures whatever the machine produces.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .embedder import DEFAULT_DIM, embed
from .tiered_cache import TieredCache


def zipf_access_sequence(
    n_keys: int,
    n_accesses: int,
    skew: float = 1.2,
    seed: int = 0,
) -> list[str]:
    """Return a list of key names sampled from a Zipfian distribution.

    Rank-1 keys are hit far more often than tail keys, which is the pattern a
    small hot tier is meant to capture.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_keys + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, skew)
    weights /= weights.sum()
    idx = rng.choice(n_keys, size=n_accesses, p=weights)
    return [f"key-{i:06d}" for i in idx]


class _PureLRU:
    """Minimal in-memory LRU baseline (no cold tier, no promotion)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: str, compute) -> np.ndarray:
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        self.misses += 1
        vec = compute()
        self._d[key] = vec
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)
        return vec

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


@dataclass
class BenchResult:
    capacity: int
    policy: str
    hit_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    lookups: int


def _percentile_ms(latencies_s: list[float], pct: float) -> float:
    if not latencies_s:
        return 0.0
    return float(np.percentile(np.array(latencies_s), pct) * 1000.0)


def run_bench(
    n_keys: int = 5000,
    n_accesses: int = 50000,
    skew: float = 1.2,
    capacities: list[int] | None = None,
    dim: int = DEFAULT_DIM,
    cache_dir_base: str = "outputs/bench_cache",
    seed: int = 0,
) -> dict:
    """Run the benchmark and return a JSON-serializable results dict."""
    if capacities is None:
        capacities = [50, 100, 250, 500, 1000]

    seq = zipf_access_sequence(n_keys, n_accesses, skew=skew, seed=seed)

    def compute_for(key: str):
        return embed(key, dim)

    results: list[BenchResult] = []

    for cap in tqdm(capacities, desc="capacities"):
        # Frequency-aware tiered cache.
        import os
        import shutil

        cdir = os.path.join(cache_dir_base, f"cap_{cap}")
        if os.path.isdir(cdir):
            shutil.rmtree(cdir)
        cache = TieredCache(hot_capacity=cap, cache_dir=cdir, dim=dim)

        lat: list[float] = []
        for key in tqdm(seq, desc=f"tiered cap={cap}", leave=False):
            t0 = time.perf_counter()
            cache.get_or_compute(key, lambda k=key: compute_for(k))
            lat.append(time.perf_counter() - t0)
        s = cache.stats_dict()
        results.append(
            BenchResult(
                capacity=cap,
                policy="tiered_freq",
                hit_rate=s["hit_rate"],
                mean_latency_ms=float(np.mean(lat) * 1000.0),
                p95_latency_ms=_percentile_ms(lat, 95),
                lookups=s["lookups"],
            )
        )

        # Pure-LRU baseline at the same capacity.
        lru = _PureLRU(capacity=cap)
        lat_lru: list[float] = []
        for key in tqdm(seq, desc=f"lru cap={cap}", leave=False):
            t0 = time.perf_counter()
            lru.get_or_compute(key, lambda k=key: compute_for(k))
            lat_lru.append(time.perf_counter() - t0)
        results.append(
            BenchResult(
                capacity=cap,
                policy="pure_lru",
                hit_rate=lru.hit_rate(),
                mean_latency_ms=float(np.mean(lat_lru) * 1000.0),
                p95_latency_ms=_percentile_ms(lat_lru, 95),
                lookups=lru.hits + lru.misses,
            )
        )

    return {
        "config": {
            "n_keys": n_keys,
            "n_accesses": n_accesses,
            "skew": skew,
            "dim": dim,
            "capacities": capacities,
            "seed": seed,
        },
        "results": [r.__dict__ for r in results],
    }
