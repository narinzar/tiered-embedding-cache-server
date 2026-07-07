"""Tests for the two-tier cache: LRU eviction, cold round-trip, promotion,
demotion, and stats accounting.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tiered_cache import TieredCache


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def test_put_get_hot_hit(tmp_path):
    cache = TieredCache(hot_capacity=4, cache_dir=str(tmp_path))
    v = _vec(1)
    cache.put("a", v)
    got = cache.get("a")
    assert got is not None
    np.testing.assert_array_equal(got, v)
    s = cache.stats_dict()
    assert s["hits"] == 1
    assert s["hot_hits"] == 1
    assert s["misses"] == 0


def test_miss_counts(tmp_path):
    cache = TieredCache(hot_capacity=4, cache_dir=str(tmp_path))
    assert cache.get("nope") is None
    assert cache.stats_dict()["misses"] == 1
    assert cache.stats_dict()["hits"] == 0


def test_lru_eviction_to_cold(tmp_path):
    # Capacity 2: inserting a 3rd key demotes the LRU key to the cold tier.
    cache = TieredCache(hot_capacity=2, cache_dir=str(tmp_path))
    cache.put("a", _vec(1))
    cache.put("b", _vec(2))
    cache.put("c", _vec(3))  # "a" is LRU -> demoted to cold

    assert "a" not in cache.hot_keys()
    assert "a" in cache.cold_keys()
    assert set(cache.hot_keys()) == {"b", "c"}
    assert cache.stats_dict()["demotions"] == 1


def test_lru_recency_refresh(tmp_path):
    # Touching "a" should protect it from being the eviction victim.
    cache = TieredCache(hot_capacity=2, cache_dir=str(tmp_path))
    cache.put("a", _vec(1))
    cache.put("b", _vec(2))
    assert cache.get("a") is not None  # "a" now most-recently used
    cache.put("c", _vec(3))  # "b" is now LRU -> demoted
    assert "b" in cache.cold_keys()
    assert "a" in cache.hot_keys()


def test_cold_tier_round_trip(tmp_path):
    # A demoted key's vector must survive the on-disk round trip unchanged.
    cache = TieredCache(hot_capacity=1, cache_dir=str(tmp_path))
    va = _vec(11)
    cache.put("a", va)
    cache.put("b", _vec(12))  # demotes "a" to disk
    assert "a" in cache.cold_keys()

    got = cache.get("a")  # cold hit, reads from disk
    assert got is not None
    np.testing.assert_array_equal(got, va)
    assert cache.stats_dict()["cold_hits"] == 1


def test_promotion_of_frequent_cold_key(tmp_path):
    # A cold key accessed more often than the hot victim should promote,
    # demoting the victim.
    cache = TieredCache(hot_capacity=1, cache_dir=str(tmp_path))

    # Make "hot" the resident with a low access count.
    cache.put("hot", _vec(1))  # count(hot) = 1
    cache.put("cold", _vec(2))  # capacity 1 -> "hot" demoted to cold, "cold" hot

    # Now "cold" is actually in the hot tier and "hot" is on disk. Rebuild a
    # clearer scenario: access the on-disk key repeatedly so its frequency
    # overtakes the resident's.
    resident = cache.hot_keys()[0]
    disk_key = cache.cold_keys()[0]

    resident_count = cache._meta[resident].count
    # Access the disk key enough times to exceed the resident's count.
    promotions_before = cache.stats_dict()["promotions"]
    demotions_before = cache.stats_dict()["demotions"]

    got = None
    for _ in range(resident_count + 2):
        got = cache.get(disk_key)
    assert got is not None

    # The disk key should now be resident in the hot tier.
    assert disk_key in cache.hot_keys()
    # The old resident should have been demoted to cold.
    assert resident in cache.cold_keys()
    assert cache.stats_dict()["promotions"] > promotions_before
    assert cache.stats_dict()["demotions"] > demotions_before


def test_promotion_uses_free_slot(tmp_path):
    # When the hot tier is below capacity, a cold hit promotes without a victim
    # (no demotion needed).
    cache = TieredCache(hot_capacity=3, cache_dir=str(tmp_path))
    cache.put("a", _vec(1))
    cache.put("b", _vec(2))
    cache.put("c", _vec(3))
    cache.put("d", _vec(4))  # hot full at 3, LRU "a" demoted to cold
    assert "a" in cache.cold_keys()

    # Open a hot slot without touching "a" so the free-slot branch is exercised.
    cache._demote_to_cold("b")  # hot now has 2 of 3 slots used
    promotions_before = cache.stats_dict()["promotions"]

    got = cache.get("a")  # cold hit with a free slot -> direct promotion
    assert got is not None
    assert "a" in cache.hot_keys()
    assert cache.stats_dict()["promotions"] == promotions_before + 1


def test_no_promotion_when_capacity_zero(tmp_path):
    cache = TieredCache(hot_capacity=0, cache_dir=str(tmp_path))
    v = _vec(1)
    cache.put("a", v)  # cannot stay hot; ends up demoted to cold
    assert "a" in cache.cold_keys()
    for _ in range(5):
        got = cache.get("a")
        assert got is not None
    # With no hot capacity there can be no promotions.
    assert cache.stats_dict()["promotions"] == 0


def test_get_or_compute(tmp_path):
    cache = TieredCache(hot_capacity=2, cache_dir=str(tmp_path))
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return _vec(99)

    a = cache.get_or_compute("a", compute)
    b = cache.get_or_compute("a", compute)  # served from cache
    np.testing.assert_array_equal(a, b)
    assert calls["n"] == 1
    s = cache.stats_dict()
    assert s["hits"] == 1  # second call was a hot hit
    assert s["misses"] == 1  # first call missed then computed


def test_stats_accounting_totals(tmp_path):
    cache = TieredCache(hot_capacity=1, cache_dir=str(tmp_path))
    cache.put("a", _vec(1))
    cache.put("b", _vec(2))  # demotes a
    cache.get("a")  # cold hit
    cache.get("b")  # hot hit (or possibly swapped) -> still a hit
    cache.get("missing")  # miss

    s = cache.stats_dict()
    assert s["lookups"] == s["hits"] + s["misses"]
    assert s["hits"] == s["hot_hits"] + s["cold_hits"]
    assert s["misses"] == 1


def test_invalid_capacity(tmp_path):
    with pytest.raises(ValueError):
        TieredCache(hot_capacity=-1, cache_dir=str(tmp_path))
