#!/usr/bin/env python3
"""
Keypoint Properties Analysis Script

Compares keypoint properties (scale, response, spatial distribution)
across different keypoint sets to understand intersection mechanism.

Usage:
    cd /home/frank/repos/DescriptorWorkbench
    python3 analysis/scripts/analyze_keypoint_properties.py

Prerequisites:
    - experiments.db in build/ directory with locked_keypoints table populated
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "build" / "experiments.db"

KEYPOINT_SETS = [
    'sift_8000',
    'sift_scale_matched_6px',
    'sift_surf_scale_matched_intersection_a',
    'sift_scale_only_13px',
    'sift_top_scale_13px'
]


def analyze_response_comparison():
    """Compare keypoint response values across sets."""
    print("\n" + "=" * 70)
    print("KEYPOINT RESPONSE COMPARISON")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT
            s.name as keypoint_set,
            COUNT(*) as count,
            ROUND(AVG(k.size), 2) as avg_size,
            ROUND(AVG(k.response), 5) as avg_response,
            ROUND(MIN(k.response), 5) as min_resp,
            ROUND(MAX(k.response), 5) as max_resp
        FROM locked_keypoints k
        JOIN keypoint_sets s ON k.keypoint_set_id = s.id
        WHERE s.name IN ({','.join('?' for _ in KEYPOINT_SETS)})
        GROUP BY s.name
        ORDER BY avg_response DESC
    """, KEYPOINT_SETS)

    print(f"\n{'Keypoint Set':<45} {'Count':>10} {'Avg Size':>10} {'Avg Resp':>12} {'Min':>10} {'Max':>10}")
    print("-" * 100)

    for row in cursor.fetchall():
        name, count, avg_size, avg_resp, min_resp, max_resp = row
        print(f"{name:<45} {count:>10,} {avg_size:>10.2f} {avg_resp:>12.5f} {min_resp:>10.5f} {max_resp:>10.5f}")

    conn.close()


def analyze_scale_distribution():
    """Compare scale distribution buckets across sets."""
    print("\n" + "=" * 70)
    print("SCALE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT
            s.name,
            SUM(CASE WHEN k.size < 6 THEN 1 ELSE 0 END) as tiny,
            SUM(CASE WHEN k.size >= 6 AND k.size < 10 THEN 1 ELSE 0 END) as small,
            SUM(CASE WHEN k.size >= 10 AND k.size < 15 THEN 1 ELSE 0 END) as medium,
            SUM(CASE WHEN k.size >= 15 AND k.size < 25 THEN 1 ELSE 0 END) as large,
            SUM(CASE WHEN k.size >= 25 THEN 1 ELSE 0 END) as very_large,
            COUNT(*) as total
        FROM locked_keypoints k
        JOIN keypoint_sets s ON k.keypoint_set_id = s.id
        WHERE s.name IN ({','.join('?' for _ in KEYPOINT_SETS)})
        GROUP BY s.name
    """, KEYPOINT_SETS)

    print(f"\n{'Keypoint Set':<45} {'<6px':>8} {'6-10':>8} {'10-15':>8} {'15-25':>8} {'>25px':>8} {'Total':>10}")
    print("-" * 100)

    for row in cursor.fetchall():
        name, tiny, small, medium, large, very_large, total = row
        print(f"{name:<45} {tiny:>8} {small:>8} {medium:>8} {large:>8} {very_large:>8} {total:>10,}")

    # Show percentages
    print("\n--- Percentages ---")
    cursor.execute(f"""
        SELECT
            s.name,
            ROUND(100.0 * SUM(CASE WHEN k.size < 6 THEN 1 ELSE 0 END) / COUNT(*), 1) as tiny_pct,
            ROUND(100.0 * SUM(CASE WHEN k.size >= 6 AND k.size < 10 THEN 1 ELSE 0 END) / COUNT(*), 1) as small_pct,
            ROUND(100.0 * SUM(CASE WHEN k.size >= 10 AND k.size < 15 THEN 1 ELSE 0 END) / COUNT(*), 1) as med_pct,
            ROUND(100.0 * SUM(CASE WHEN k.size >= 15 AND k.size < 25 THEN 1 ELSE 0 END) / COUNT(*), 1) as large_pct,
            ROUND(100.0 * SUM(CASE WHEN k.size >= 25 THEN 1 ELSE 0 END) / COUNT(*), 1) as vlarge_pct
        FROM locked_keypoints k
        JOIN keypoint_sets s ON k.keypoint_set_id = s.id
        WHERE s.name IN ({','.join('?' for _ in KEYPOINT_SETS)})
        GROUP BY s.name
    """, KEYPOINT_SETS)

    print(f"\n{'Keypoint Set':<45} {'<6px':>8} {'6-10':>8} {'10-15':>8} {'15-25':>8} {'>25px':>8}")
    print("-" * 90)

    for row in cursor.fetchall():
        name, tiny, small, medium, large, very_large = row
        print(f"{name:<45} {tiny:>7}% {small:>7}% {medium:>7}% {large:>7}% {very_large:>7}%")

    conn.close()


def analyze_spatial_distribution():
    """Compare spatial distribution by quadrant."""
    print("\n" + "=" * 70)
    print("SPATIAL DISTRIBUTION ANALYSIS (Quadrants)")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Assuming 800x600 images, divide into quadrants
    cursor.execute(f"""
        SELECT
            s.name,
            SUM(CASE WHEN k.x < 400 AND k.y < 300 THEN 1 ELSE 0 END) as top_left,
            SUM(CASE WHEN k.x >= 400 AND k.y < 300 THEN 1 ELSE 0 END) as top_right,
            SUM(CASE WHEN k.x < 400 AND k.y >= 300 THEN 1 ELSE 0 END) as bottom_left,
            SUM(CASE WHEN k.x >= 400 AND k.y >= 300 THEN 1 ELSE 0 END) as bottom_right,
            COUNT(*) as total
        FROM locked_keypoints k
        JOIN keypoint_sets s ON k.keypoint_set_id = s.id
        WHERE s.name IN ('sift_surf_scale_matched_intersection_a', 'sift_top_scale_13px', 'sift_scale_only_13px')
        GROUP BY s.name
    """)

    print(f"\n{'Keypoint Set':<45} {'TL':>8} {'TR':>8} {'BL':>8} {'BR':>8} {'Total':>10}")
    print("-" * 90)

    for row in cursor.fetchall():
        name, tl, tr, bl, br, total = row
        print(f"{name:<45} {tl:>8,} {tr:>8,} {bl:>8,} {br:>8,} {total:>10,}")

    conn.close()


def analyze_specific_scene(scene_name):
    """Analyze keypoint properties for a specific scene."""
    print(f"\n" + "=" * 70)
    print(f"SCENE ANALYSIS: {scene_name}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            s.name as kp_set,
            COUNT(*) as kp_count,
            ROUND(AVG(k.size), 2) as avg_size,
            ROUND(AVG(k.response), 5) as avg_resp
        FROM locked_keypoints k
        JOIN keypoint_sets s ON k.keypoint_set_id = s.id
        WHERE k.scene_name = ?
          AND s.name IN ('sift_surf_scale_matched_intersection_a', 'sift_top_scale_13px', 'sift_scale_only_13px')
        GROUP BY s.name
    """, (scene_name,))

    print(f"\n{'Keypoint Set':<45} {'Count':>10} {'Avg Size':>10} {'Avg Resp':>12}")
    print("-" * 80)

    for row in cursor.fetchall():
        name, count, avg_size, avg_resp = row
        print(f"{name:<45} {count:>10,} {avg_size:>10.2f} {avg_resp:>12.5f}")

    conn.close()


def main():
    print("=" * 70)
    print("KEYPOINT PROPERTIES ANALYSIS")
    print("=" * 70)
    print(f"\nDatabase: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    analyze_response_comparison()
    analyze_scale_distribution()
    analyze_spatial_distribution()

    # Analyze extreme cases
    analyze_specific_scene('v_colors')  # Biggest intersection win
    analyze_specific_scene('i_brooklyn')  # Biggest intersection loss

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
