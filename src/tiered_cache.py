"""Two-tier embedding cache with a frequency-aware promotion policy.

Tiers
-----
Hot tier : an in-memory OrderedDict acting as an LRU with a fixed capacity.
Cold tier: an on-disk store of numpy .npy files under a cache directory.

Lookup
------
`get(key)` checks the hot tier first, then the cold tier. Every access updates
a per-key access count and last-access timestamp. On a cold hit the promotion
policy runs.

Promotion / demotion policy
---------------------------
Recency alone (plain LRU) throws away a key just because it has not been touched
recently, even if it is accessed often overall. Here a cold key promotes into
the hot tier when its measured access frequency exceeds the frequency of the
hot tier's least-frequently-used victim. When that happens the victim is
demoted to the cold tier and the cold key takes its place. If the hot tier is
not full, a cold hit promotes for free (no victim needed). This keeps genuinely
hot-by-frequency keys resident even if their accesses are bursty rather than
strictly recent.

The class is deliberately not thread-safe on its own; the server serializes
access with an asyncio lock.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CacheStats:
    """Running counters for cache behavior."""

    hits: int = 0
    misses: int = 0
    hot_hits: int = 0
    cold_hits: int = 0
    promotions: int = 0
    demotions: int = 0

    def as_dict(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hot_hits": self.hot_hits,
            "cold_hits": self.cold_hits,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "lookups": total,
            "hit_rate": hit_rate,
        }


@dataclass
class _Meta:
    """Per-key access bookkeeping (kept for every key ever seen)."""

    count: int = 0
    last_access: float = field(default_factory=time.time)


class TieredCache:
    """LRU hot tier + on-disk cold tier with frequency-based promotion."""

    def __init__(self, hot_capacity: int, cache_dir: str, dim: int | None = None):
        if hot_capacity < 0:
            raise ValueError("hot_capacity must be >= 0")
        self.hot_capacity = hot_capacity
        self.cache_dir = cache_dir
        self.dim = dim
        os.makedirs(self.cache_dir, exist_ok=True)

        # Hot tier: key -> vector, ordered from least- to most-recently used.
        self._hot: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # Set of keys currently living on disk.
        self._cold: set[str] = set()
        # Per-key access metadata.
        self._meta: dict[str, _Meta] = {}

        self.stats = CacheStats()

    # ---- disk helpers -------------------------------------------------
    def _path(self, key: str) -> str:
        # Hash the key to a filesystem-safe stable filename.
        import hashlib

        h = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    def _disk_write(self, key: str, vec: np.ndarray) -> None:
        np.save(self._path(key), vec)

    def _disk_read(self, key: str) -> np.ndarray:
        return np.load(self._path(key))

    def _disk_delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    # ---- metadata -----------------------------------------------------
    def _touch(self, key: str) -> _Meta:
        meta = self._meta.get(key)
        if meta is None:
            meta = _Meta(count=0, last_access=time.time())
            self._meta[key] = meta
        meta.count += 1
        meta.last_access = time.time()
        return meta

    def _min_freq_hot_key(self) -> str | None:
        """Return the hot key with the smallest access count (LFU victim).

        Ties are broken by least-recently-used order, which is the natural
        iteration order of the OrderedDict (front = oldest).
        """
        victim = None
        victim_count = None
        for k in self._hot:  # oldest first
            c = self._meta[k].count
            if victim_count is None or c < victim_count:
                victim = k
                victim_count = c
        return victim

    # ---- eviction / demotion -----------------------------------------
    def _demote_to_cold(self, key: str) -> None:
        """Move a key from the hot tier to the cold tier."""
        vec = self._hot.pop(key)
        self._disk_write(key, vec)
        self._cold.add(key)
        self.stats.demotions += 1

    def _evict_if_needed(self) -> None:
        """If the hot tier is over capacity, demote the LRU key to cold.

        This is the plain-LRU eviction used when inserting a freshly computed
        value. Frequency-based swaps are handled separately in `get`.
        """
        while len(self._hot) > self.hot_capacity:
            # OrderedDict is LRU-ordered: first item is least-recently used.
            lru_key = next(iter(self._hot))
            self._demote_to_cold(lru_key)

    def _insert_hot(self, key: str, vec: np.ndarray) -> None:
        self._hot[key] = vec
        self._hot.move_to_end(key)  # most-recently used
        self._cold.discard(key)
        self._evict_if_needed()

    # ---- public API ---------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self._hot or key in self._cold

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached vector for `key`, or None on a miss.

        Updates stats and applies the promotion policy on a cold hit.
        """
        # Hot hit.
        if key in self._hot:
            self._touch(key)
            self._hot.move_to_end(key)  # refresh recency
            self.stats.hits += 1
            self.stats.hot_hits += 1
            return self._hot[key]

        # Cold hit.
        if key in self._cold:
            meta = self._touch(key)
            vec = self._disk_read(key)
            self.stats.hits += 1
            self.stats.cold_hits += 1
            self._maybe_promote(key, vec, meta.count)
            return vec

        # Miss.
        self.stats.misses += 1
        return None

    def _maybe_promote(self, key: str, vec: np.ndarray, key_count: int) -> None:
        """Decide whether a cold `key` should move into the hot tier."""
        if self.hot_capacity == 0:
            return  # nowhere to promote to

        # Free space: promote directly.
        if len(self._hot) < self.hot_capacity:
            self._cold.discard(key)
            self._disk_delete(key)
            self._insert_hot(key, vec)
            self.stats.promotions += 1
            return

        # Full hot tier: compare against the least-frequent hot victim.
        victim = self._min_freq_hot_key()
        if victim is None:
            return
        victim_count = self._meta[victim].count

        # Strictly-greater frequency wins the slot. Equal frequency keeps the
        # incumbent to avoid churn on ties.
        if key_count > victim_count:
            self._demote_to_cold(victim)  # victim -> cold
            self._cold.discard(key)
            self._disk_delete(key)
            self._insert_hot(key, vec)  # key -> hot
            self.stats.promotions += 1

    def put(self, key: str, vec: np.ndarray) -> None:
        """Insert a freshly computed value. Goes into the hot tier (LRU)."""
        self._touch(key)
        self._insert_hot(key, vec)

    def get_or_compute(self, key: str, compute) -> np.ndarray:
        """Return cached vector or compute, store, and return it.

        `compute` is a zero-arg callable returning a numpy array. This is the
        method the server and benchmark use.
        """
        vec = self.get(key)
        if vec is not None:
            return vec
        vec = compute()
        self.put(key, vec)
        return vec

    # ---- introspection ------------------------------------------------
    def hot_keys(self) -> list[str]:
        """Hot keys in LRU order (oldest first)."""
        return list(self._hot.keys())

    def cold_keys(self) -> list[str]:
        return sorted(self._cold)

    def stats_dict(self) -> dict:
        d = self.stats.as_dict()
        d.update(
            {
                "hot_size": len(self._hot),
                "cold_size": len(self._cold),
                "hot_capacity": self.hot_capacity,
                "distinct_keys": len(self._meta),
            }
        )
        return d
