"""Run the cache benchmark, write outputs/bench.json, and plot hit rate.

Usage:
    python scripts/run_bench.py
    python scripts/run_bench.py --n-keys 8000 --n-accesses 80000 --skew 1.1

The plot (outputs/hit_rate_vs_capacity.png) shows hit rate versus hot-tier
capacity for the frequency-aware tiered cache and a pure-LRU baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as `python scripts/run_bench.py` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt  # noqa: E402

from src.bench import run_bench  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tiered cache benchmark.")
    p.add_argument("--n-keys", type=int, default=5000)
    p.add_argument("--n-accesses", type=int, default=50000)
    p.add_argument("--skew", type=float, default=1.2)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=[50, 100, 250, 500, 1000],
    )
    p.add_argument("--outputs", type=str, default="outputs")
    return p.parse_args()


def _plot(report: dict, out_png: str) -> None:
    rows = report["results"]
    policies = sorted({r["policy"] for r in rows})
    plt.figure(figsize=(7, 5))
    for policy in policies:
        pts = sorted(
            [(r["capacity"], r["hit_rate"]) for r in rows if r["policy"] == policy]
        )
        xs = [c for c, _ in pts]
        ys = [h for _, h in pts]
        plt.plot(xs, ys, marker="o", label=policy)
    plt.xlabel("hot-tier capacity (keys)")
    plt.ylabel("hit rate")
    plt.title("Hit rate vs hot-tier capacity (Zipfian access)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.outputs, exist_ok=True)

    report = run_bench(
        n_keys=args.n_keys,
        n_accesses=args.n_accesses,
        skew=args.skew,
        capacities=args.capacities,
        dim=args.dim,
        cache_dir_base=os.path.join(args.outputs, "bench_cache"),
        seed=args.seed,
    )

    out_json = os.path.join(args.outputs, "bench.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_png = os.path.join(args.outputs, "hit_rate_vs_capacity.png")
    _plot(report, out_png)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    print("Summary (capacity | policy | hit_rate | mean_ms | p95_ms):")
    for r in report["results"]:
        print(
            f"  {r['capacity']:>5} | {r['policy']:<11} | "
            f"{r['hit_rate']:.3f} | {r['mean_latency_ms']:.4f} | "
            f"{r['p95_latency_ms']:.4f}"
        )


if __name__ == "__main__":
    main()
