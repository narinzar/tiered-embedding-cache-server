# outputs/

Benchmark artifacts land here and are gitignored.

- `bench.json` : raw benchmark results (config + per-capacity, per-policy hit
  rate and latency).
- `hit_rate_vs_capacity.png` : plot of hit rate versus hot-tier capacity for the
  frequency-aware tiered cache and the pure-LRU baseline.
- `bench_cache/` : scratch cold-tier directories used during benchmarking.

Produced by `python scripts/run_bench.py`. Safe to delete.
