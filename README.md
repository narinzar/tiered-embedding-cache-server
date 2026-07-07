# tiered-embedding-cache-server

An LRU + disk-tiered cache for text embeddings, exposed over a small FastAPI
HTTP API, with a frequency-aware promotion/demotion policy and a Zipfian
benchmark for hit rate and latency.

## Problem

Embedding lookups are expensive to recompute and large embedding sets do not
fit in memory. A single in-memory LRU either wastes RAM or thrashes on a working
set larger than its capacity. Real access patterns are skewed: a small set of
keys is hit constantly while a long tail is hit rarely. The non-trivial part is
keeping the genuinely hot-by-frequency keys resident even when their accesses
are bursty rather than strictly recent, while still bounding memory and paging
the cold tail to disk without losing correctness on the round trip.

## Approach

- Two tiers behind one `get`: an in-memory `OrderedDict` LRU hot tier with a
  fixed capacity, and an on-disk cold tier of `.npy` files under a cache dir.
- Per-key bookkeeping of access count and last-access time, kept for every key
  ever seen (not just resident keys).
- Frequency-aware promotion: on a cold hit, if the cold key's access count
  exceeds the least-frequently-used hot victim's count, the cold key promotes
  into the hot tier and the victim demotes to disk. Ties keep the incumbent to
  avoid churn. A free hot slot promotes for free.
- Deterministic offline embedder using the signed hashing trick plus L2
  normalization, so the whole thing runs with no model download. It is a
  stand-in and can be swapped for sentence-transformers or an API embedder
  without touching the cache or server.
- FastAPI server serializes cache access with an asyncio lock and offloads
  embedding computation to a thread so the event loop stays responsive.
- Benchmark replays a Zipfian access sequence through the cache at several hot
  capacities and compares the frequency-aware policy against a pure-LRU
  baseline at the same capacity.

## Setup

```bash
# create a virtual environment (either tool)
uv venv --python 3.12 .venv
# or: python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env                  # no secrets required; placeholder only
```

This project is CPU-only and needs no torch. If you swap the embedder for a real
GPU model, install torch from the CUDA 12.8 index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## How to run

Run the server:

```bash
python scripts/run_server.py
# override capacity/port via env, e.g.:
# HOT_CAPACITY=512 PORT=8001 python scripts/run_server.py
```

Then query it:

```bash
curl http://127.0.0.1:8000/embedding/hello
curl http://127.0.0.1:8000/stats
curl -X POST http://127.0.0.1:8000/warm \
  -H "Content-Type: application/json" \
  -d '{"keys": ["a", "b", "c"]}'
```

Run the benchmark (writes `outputs/bench.json` and the plot):

```bash
python scripts/run_bench.py
# custom pattern:
# python scripts/run_bench.py --n-keys 8000 --n-accesses 80000 --skew 1.1 \
#   --capacities 50 100 250 500 1000
```

Run the tests:

```bash
pytest -q
```

## Results

Reproduce with:

```bash
python scripts/run_bench.py
```

This produces `outputs/bench.json` and `outputs/hit_rate_vs_capacity.png`.

Expected qualitative behavior:

- Hit rate should rise with hot-tier capacity for both policies, then flatten as
  the hot tier grows large enough to hold the skewed working set.
- Because the access pattern is Zipfian, a small hot tier should already capture
  most hits (the rank-1 keys dominate), so the curve should climb steeply at
  small capacities and level off.
- The frequency-aware tiered policy should reach an equal-or-higher hit rate
  than pure LRU at the same capacity under skewed access, since it keeps
  often-hit keys resident instead of evicting them on a recency dip. The gap is
  largest at small-to-mid capacities and shrinks as capacity approaches the
  number of distinct keys.
- Mean and p95 latency are dominated by cold-tier disk reads and misses (which
  recompute the embedding), so latency should fall as hit rate rises.

| capacity | policy      | hit_rate | mean_ms | p95_ms |
|----------|-------------|----------|---------|--------|
| 50       | tiered_freq | TBD (run) | TBD (run) | TBD (run) |
| 50       | pure_lru    | TBD (run) | TBD (run) | TBD (run) |
| 500      | tiered_freq | TBD (run) | TBD (run) | TBD (run) |
| 500      | pure_lru    | TBD (run) | TBD (run) | TBD (run) |

Numbers below are produced by running the commands above; this repo ships the
code, run it to populate them.

## What I'd do next at larger scale

Replace the per-key `.npy` cold tier with a single memory-mapped store or an
embedded key-value DB (LMDB/RocksDB) so millions of keys do not become millions
of small files, and batch cold reads. Shard the hot tier per key-hash with
per-shard locks (or move to a lock-free structure) to cut contention under
concurrent load, and expose the promotion threshold as a tunable so operators
can trade churn against hit rate for their own access skew.
