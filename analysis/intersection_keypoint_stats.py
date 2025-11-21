#!/usr/bin/env python3
"""
Inspect whether intersection keypoint sets differ in scale/response compared to their parent sets.

Default mode compares the common SIFT/KeyNet sets:
  - sift_8000 vs keynet_sift_8k_b (SIFT side of intersection)
  - keynet_8000 vs keynet_sift_8k_a (KeyNet side of intersection)
  - sift_scale_matched_6px vs sift_keynet_scale_matched_intersection_a (scale-matched SIFT side)
  - keynet_scale_matched_6px vs sift_keynet_scale_matched_intersection_b (scale-matched KeyNet side)

Outputs distribution stats (count, mean, std, median, p90/p95/p99) for size and response,
plus simple effect sizes (Cohen's d) vs the paired baseline.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SETS = [
    "sift_8000",
    "keynet_8000",
    "sift_scale_matched_6px",
    "keynet_scale_matched_6px",
    "keynet_sift_8k_a",
    "keynet_sift_8k_b",
    "sift_keynet_scale_matched_intersection_a",
    "sift_keynet_scale_matched_intersection_b",
]

# Baseline mapping: set -> baseline to compute deltas/effect sizes
PAIR_BASELINES = {
    "keynet_sift_8k_a": "keynet_8000",
    "keynet_sift_8k_b": "sift_8000",
    "sift_keynet_scale_matched_intersection_a": "sift_scale_matched_6px",
    "sift_keynet_scale_matched_intersection_b": "keynet_scale_matched_6px",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare keypoint set statistics for intersection vs baseline sets.")
    parser.add_argument(
        "--db",
        default="build/experiments.db",
        help="Path to experiments.db (default: build/experiments.db)",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=DEFAULT_SETS,
        help="Keypoint set names to analyze (default: common SIFT/KeyNet sets)",
    )
    return parser.parse_args()


def fetch_set_ids(conn: sqlite3.Connection, names: List[str]) -> Dict[str, int]:
    placeholders = ",".join(["?"] * len(names))
    rows = conn.execute(f"SELECT name, id FROM keypoint_sets WHERE name IN ({placeholders})", names).fetchall()
    name_to_id = {row[0]: row[1] for row in rows}
    missing = [n for n in names if n not in name_to_id]
    if missing:
        raise SystemExit(f"Missing keypoint sets in DB: {missing}")
    return name_to_id


def load_values(conn: sqlite3.Connection, set_id: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_sql_query(
        "SELECT size, response FROM locked_keypoints WHERE keypoint_set_id = ?",
        conn,
        params=(set_id,),
    )
    return df["size"].to_numpy(), df["response"].to_numpy()


def describe(values: np.ndarray) -> Dict[str, float]:
    return {
        "count": values.size,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def cohen_d(sample: np.ndarray, baseline: np.ndarray) -> float:
    n1, n2 = sample.size, baseline.size
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(sample, ddof=1), np.var(baseline, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    return (np.mean(sample) - np.mean(baseline)) / np.sqrt(pooled)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        name_to_id = fetch_set_ids(conn, args.sets)
        stats = {}
        raw = {}
        for name, set_id in name_to_id.items():
            size_vals, resp_vals = load_values(conn, set_id)
            stats[name] = {
                "size": describe(size_vals),
                "response": describe(resp_vals),
            }
            raw[name] = {"size": size_vals, "response": resp_vals}
    finally:
        conn.close()

    print("=== Keypoint Set Statistics ===")
    header = (
        f"{'set':35}  {'n':>9}  {'mean_size':>10}  {'p50':>8}  {'p90':>8}  {'p95':>8}  {'p99':>8}  "
        f"{'mean_resp':>11}  {'resp_p90':>9}  {'resp_p95':>9}  {'resp_p99':>9}"
    )
    print(header)
    print("-" * len(header))
    for name in args.sets:
        s = stats[name]["size"]
        r = stats[name]["response"]
        print(
            f"{name:35}  {s['count']:9d}  {s['mean']:10.4f}  {s['p50']:8.2f}  {s['p90']:8.2f}  {s['p95']:8.2f}  {s['p99']:8.2f}  "
            f"{r['mean']:11.2f}  {r['p90']:9.2f}  {r['p95']:9.2f}  {r['p99']:9.2f}"
        )
    print()

    print("=== Effect sizes vs baseline (Cohen's d for means) ===")
    header = f"{'set':35}  {'baseline':35}  {'d_size':>8}  {'d_resp':>8}"
    print(header)
    print("-" * len(header))
    for name, baseline in PAIR_BASELINES.items():
        if name not in raw or baseline not in raw:
            continue
        d_size = cohen_d(raw[name]["size"], raw[baseline]["size"])
        d_resp = cohen_d(raw[name]["response"], raw[baseline]["response"])
        print(f"{name:35}  {baseline:35}  {d_size:8.2f}  {d_resp:8.2f}")


if __name__ == "__main__":
    main()
