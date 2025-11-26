#!/usr/bin/env python3
"""
Summarize intersection vs. scale-controlled experiment results.

This script aggregates mAP metrics and keypoint set statistics for a given
experiment name (as stored in experiments.parameters under experiment_name=...).
It is intended to quickly compare phases like the fair_fusion_sift_hardnet_comprehensive
run and inspect how keypoint scale/response distributions differ between sets.

Usage:
  python analysis/intersection_effects_summary.py --experiment-name fair_fusion_sift_hardnet_comprehensive
  python analysis/intersection_effects_summary.py --experiment-name composite_sift_hardnet_intersection_controll
"""

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize intersection experiments from experiments.db")
    parser.add_argument(
        "--db",
        default="build/experiments.db",
        help="Path to experiments.db (default: build/experiments.db)",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="experiment_name value to match in experiments.parameters (e.g., fair_fusion_sift_hardnet_comprehensive)",
    )
    return parser.parse_args()


def fetch_experiments(conn: sqlite3.Connection, experiment_name: str) -> List[Dict]:
    """Load experiments and their results for a given experiment_name."""
    query = """
    SELECT
        e.id,
        e.descriptor_type,
        e.keypoint_set_id,
        e.parameters,
        r.true_map_micro,
        r.viewpoint_map,
        r.illumination_map,
        r.total_matches,
        r.total_keypoints
    FROM experiments e
    JOIN results r ON e.id = r.experiment_id
    WHERE e.parameters LIKE ?
    ORDER BY e.id;
    """
    rows = conn.execute(query, (f"%experiment_name={experiment_name}%",)).fetchall()
    experiments = []
    for row in rows:
        (
            exp_id,
            descriptor_type,
            keypoint_set_id,
            parameters,
            true_map_micro,
            viewpoint_map,
            illumination_map,
            total_matches,
            total_keypoints,
        ) = row
        experiments.append(
            {
                "id": exp_id,
                "descriptor_type": descriptor_type,
                "keypoint_set_id": keypoint_set_id,
                "parameters": parameters,
                "map": true_map_micro,
                "vp_map": viewpoint_map,
                "il_map": illumination_map,
                "total_matches": total_matches,
                "total_keypoints": total_keypoints,
            }
        )
    return experiments


def fetch_keypoint_sets(conn: sqlite3.Connection, keypoint_set_ids: List[int]) -> Dict[int, Dict]:
    """Load keypoint set metadata and aggregate scale/response stats."""
    placeholders = ",".join(["?"] * len(keypoint_set_ids))
    meta_query = f"""
    SELECT id, name, total_keypoints, avg_keypoints_per_image,
           intersection_method, tolerance_px
    FROM keypoint_sets
    WHERE id IN ({placeholders})
    """
    meta_rows = conn.execute(meta_query, keypoint_set_ids).fetchall()

    # Aggregate size/response from locked_keypoints (cheap aggregate, no rows returned)
    stats_query = f"""
    SELECT keypoint_set_id, COUNT(*) AS n, AVG(size) AS mean_size, AVG(response) AS mean_resp
    FROM locked_keypoints
    WHERE keypoint_set_id IN ({placeholders})
    GROUP BY keypoint_set_id
    """
    stats_rows = conn.execute(stats_query, keypoint_set_ids).fetchall()
    stats_map = {row[0]: {"count": row[1], "mean_size": row[2], "mean_resp": row[3]} for row in stats_rows}

    result = {}
    for row in meta_rows:
        set_id, name, total_kp, avg_kp_per_image, intersection_method, tolerance_px = row
        result[set_id] = {
            "name": name,
            "total_kp": total_kp,
            "avg_kp_per_image": avg_kp_per_image,
            "intersection_method": intersection_method,
            "tolerance_px": tolerance_px,
            "agg_count": stats_map.get(set_id, {}).get("count"),
            "mean_size": stats_map.get(set_id, {}).get("mean_size"),
            "mean_resp": stats_map.get(set_id, {}).get("mean_resp"),
        }
    return result


def format_float(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def print_experiment_table(experiments: List[Dict], kp_sets: Dict[int, Dict]) -> None:
    print("=== Experiments ===")
    header = f"{'id':>4}  {'descriptor':<35}  {'mAP':>6}  {'VP':>6}  {'IL':>6}  {'kp_set':<35}"
    print(header)
    print("-" * len(header))
    for exp in experiments:
        kp_meta = kp_sets.get(exp["keypoint_set_id"], {})
        kp_name = kp_meta.get("name", f"id={exp['keypoint_set_id']}")
        line = (
            f"{exp['id']:>4}  "
            f"{exp['descriptor_type']:<35}  "
            f"{format_float(exp['map']):>6}  "
            f"{format_float(exp['vp_map']):>6}  "
            f"{format_float(exp['il_map']):>6}  "
            f"{kp_name:<35}"
        )
        print(line)
    print()


def print_keypoint_table(kp_sets: Dict[int, Dict]) -> None:
    print("=== Keypoint Sets ===")
    header = (
        f"{'id':>4}  {'name':<40}  {'count':>10}  {'mean_size':>9}  {'mean_resp':>11}  "
        f"{'method':<12}  {'tol_px':>6}"
    )
    print(header)
    print("-" * len(header))
    for set_id, meta in sorted(kp_sets.items()):
        line = (
            f"{set_id:>4}  "
            f"{meta.get('name',''):40.40}  "
            f"{(meta.get('agg_count') or meta.get('total_kp') or 0):>10}  "
            f"{format_float(meta.get('mean_size')):>9}  "
            f"{format_float(meta.get('mean_resp')):>11}  "
            f"{(meta.get('intersection_method') or '-'):12.12}  "
            f"{(meta.get('tolerance_px') or 0):>6}"
        )
        print(line)
    print()


def main():
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        experiments = fetch_experiments(conn, args.experiment_name)
        if not experiments:
            raise SystemExit(f"No experiments found with experiment_name={args.experiment_name}")
        kp_ids = sorted({exp["keypoint_set_id"] for exp in experiments})
        kp_sets = fetch_keypoint_sets(conn, kp_ids)

        print_experiment_table(experiments, kp_sets)
        print_keypoint_table(kp_sets)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
