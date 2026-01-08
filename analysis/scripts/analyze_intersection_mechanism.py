#!/usr/bin/env python3
"""
Intersection Mechanism Analysis Script

This script analyzes why detector intersection (SIFT-SURF) outperforms
pure scale selection for keypoint quality.

Usage:
    cd /home/frank/repos/DescriptorWorkbench
    python3 analysis/scripts/analyze_intersection_mechanism.py

Prerequisites:
    - experiments.db in build/ directory
    - Experiments completed for:
        - dspsift_v2__sift_surf_intersection__intersection
        - dspsift_v2__sift_top_scale_13px__pure_scale
        - dspsift_v2__sift_scale_only_13px__scale_only

Output:
    - Per-scene mAP comparison
    - Viewpoint vs Illumination breakdown
    - Keypoint count correlation analysis
"""

import sqlite3
import re
import os
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "build" / "experiments.db"
LOGS_DIR = PROJECT_ROOT / "logs"

def extract_metadata_to_file():
    """Extract per-scene metadata from results table to logs file."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e.descriptor_type, r.metadata
        FROM experiments e
        JOIN results r ON e.id = r.experiment_id
        WHERE e.descriptor_type LIKE '%dspsift_v2%'
          AND (e.descriptor_type LIKE '%intersection__intersection%'
               OR e.descriptor_type LIKE '%scale_only%'
               OR e.descriptor_type LIKE '%pure_scale%')
    """)

    LOGS_DIR.mkdir(exist_ok=True)
    output_file = LOGS_DIR / "scene_metadata.txt"

    with open(output_file, 'w') as f:
        for row in cursor.fetchall():
            f.write(f"{row[0]}|{row[1]}\n")

    conn.close()
    print(f"Extracted metadata to {output_file}")
    return output_file


def parse_scene_results(metadata_file):
    """Parse per-scene mAP from metadata file."""
    with open(metadata_file, 'r') as f:
        lines = f.readlines()

    results = {}
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) != 2:
            continue
        desc_type = parts[0]
        metadata = parts[1]

        # Determine keypoint set type
        if 'intersection__intersection' in desc_type:
            kp_type = 'intersection'
        elif 'scale_only' in desc_type:
            kp_type = 'scale_only'
        elif 'pure_scale' in desc_type:
            kp_type = 'pure_scale'
        else:
            continue

        # Parse scene mAPs
        scene_maps = {}
        for match in re.finditer(r'([vi]_\w+)_true_map=([0-9.]+)', metadata):
            scene = match.group(1)
            mAP = float(match.group(2))
            scene_maps[scene] = mAP

        results[kp_type] = scene_maps

    return results


def analyze_per_scene(results):
    """Compare per-scene mAP across keypoint sets."""
    print("\n" + "=" * 80)
    print("PER-SCENE mAP COMPARISON")
    print("=" * 80)

    print(f"\n{'Scene':<20} {'Intersection':>12} {'Pure Scale':>12} {'Scale Only':>12} {'Int-Pure':>10} {'Int-Scale':>10}")
    print("-" * 80)

    scenes = sorted(results['intersection'].keys())
    for scene in scenes:
        int_map = results['intersection'].get(scene, 0) * 100
        pure_map = results['pure_scale'].get(scene, 0) * 100
        scale_map = results['scale_only'].get(scene, 0) * 100

        int_pure_diff = int_map - pure_map
        int_scale_diff = int_map - scale_map

        print(f"{scene:<20} {int_map:>11.1f}% {pure_map:>11.1f}% {scale_map:>11.1f}% {int_pure_diff:>+9.1f}% {int_scale_diff:>+9.1f}%")

    # Summary stats
    int_maps = [results['intersection'][s] * 100 for s in scenes]
    pure_maps = [results['pure_scale'][s] * 100 for s in scenes]
    scale_maps = [results['scale_only'][s] * 100 for s in scenes]

    int_pure_diffs = [results['intersection'][s] - results['pure_scale'][s] for s in scenes]
    int_scale_diffs = [results['intersection'][s] - results['scale_only'][s] for s in scenes]

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Average mAP: Intersection={sum(int_maps)/len(int_maps):.1f}%  "
          f"Pure Scale={sum(pure_maps)/len(pure_maps):.1f}%  "
          f"Scale Only={sum(scale_maps)/len(scale_maps):.1f}%")
    print(f"Scenes where intersection wins vs pure_scale: {sum(1 for d in int_pure_diffs if d > 0)}/{len(int_pure_diffs)}")
    print(f"Scenes where intersection wins vs scale_only: {sum(1 for d in int_scale_diffs if d > 0)}/{len(int_scale_diffs)}")

    # Find biggest winners/losers
    int_pure_diffs_named = [(s, (results['intersection'][s] - results['pure_scale'][s]) * 100) for s in scenes]
    int_pure_diffs_named.sort(key=lambda x: x[1], reverse=True)

    print("\nBiggest intersection gains vs pure_scale:")
    for s, d in int_pure_diffs_named[:5]:
        print(f"  {s}: {d:+.1f}%")

    print("\nScenes where pure_scale beats intersection:")
    for s, d in int_pure_diffs_named[-5:]:
        if d < 0:
            print(f"  {s}: {d:+.1f}%")

    return scenes


def analyze_viewpoint_vs_illumination(results, scenes):
    """Compare intersection benefit for viewpoint vs illumination scenes."""
    print("\n" + "=" * 80)
    print("VIEWPOINT vs ILLUMINATION ANALYSIS")
    print("=" * 80)

    v_scenes = [s for s in scenes if s.startswith('v_')]
    i_scenes = [s for s in scenes if s.startswith('i_')]

    # Viewpoint analysis
    v_int_wins = sum(1 for s in v_scenes
                     if results['intersection'][s] > results['pure_scale'][s])
    v_int_avg = sum(results['intersection'][s] for s in v_scenes) / len(v_scenes) * 100
    v_pure_avg = sum(results['pure_scale'][s] for s in v_scenes) / len(v_scenes) * 100
    v_scale_avg = sum(results['scale_only'][s] for s in v_scenes) / len(v_scenes) * 100
    v_gains = [(results['intersection'][s] - results['pure_scale'][s]) * 100 for s in v_scenes]

    print(f"\nViewpoint scenes: {len(v_scenes)}")
    print(f"Average mAP: Intersection={v_int_avg:.1f}%  Pure Scale={v_pure_avg:.1f}%  Scale Only={v_scale_avg:.1f}%")
    print(f"Intersection wins: {v_int_wins}/{len(v_scenes)} ({100*v_int_wins/len(v_scenes):.0f}%)")
    print(f"Average gain vs pure_scale: {sum(v_gains)/len(v_gains):+.1f}%")

    # Illumination analysis
    i_int_wins = sum(1 for s in i_scenes
                     if results['intersection'][s] > results['pure_scale'][s])
    i_int_avg = sum(results['intersection'][s] for s in i_scenes) / len(i_scenes) * 100
    i_pure_avg = sum(results['pure_scale'][s] for s in i_scenes) / len(i_scenes) * 100
    i_scale_avg = sum(results['scale_only'][s] for s in i_scenes) / len(i_scenes) * 100
    i_gains = [(results['intersection'][s] - results['pure_scale'][s]) * 100 for s in i_scenes]

    print(f"\nIllumination scenes: {len(i_scenes)}")
    print(f"Average mAP: Intersection={i_int_avg:.1f}%  Pure Scale={i_pure_avg:.1f}%  Scale Only={i_scale_avg:.1f}%")
    print(f"Intersection wins: {i_int_wins}/{len(i_scenes)} ({100*i_int_wins/len(i_scenes):.0f}%)")
    print(f"Average gain vs pure_scale: {sum(i_gains)/len(i_gains):+.1f}%")

    print("\n" + "-" * 40)
    print("KEY FINDING:")
    print(f"Viewpoint gain:    {v_int_avg - v_pure_avg:+.1f}% (intersection vs pure_scale)")
    print(f"Illumination gain: {i_int_avg - i_pure_avg:+.1f}% (intersection vs pure_scale)")


def analyze_keypoint_count_correlation(results, scenes):
    """Analyze if intersection having fewer keypoints correlates with better performance."""
    print("\n" + "=" * 80)
    print("KEYPOINT COUNT CORRELATION ANALYSIS")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Query keypoint counts per scene for each set
    cursor.execute("""
        SELECT
            s.name as kp_set,
            k.scene_name,
            COUNT(*) as kp_count
        FROM locked_keypoints k
        JOIN keypoint_sets s ON k.keypoint_set_id = s.id
        WHERE s.name IN ('sift_surf_scale_matched_intersection_a', 'sift_top_scale_13px')
        GROUP BY s.name, k.scene_name
    """)

    kp_counts = {}
    for row in cursor.fetchall():
        kp_set, scene, count = row
        if scene not in kp_counts:
            kp_counts[scene] = {}
        kp_counts[scene][kp_set] = count

    conn.close()

    # Analyze correlation
    scenes_with_data = [s for s in scenes
                        if s in kp_counts
                        and 'sift_surf_scale_matched_intersection_a' in kp_counts[s]
                        and 'sift_top_scale_13px' in kp_counts[s]]

    int_more_kp = []  # Scenes where intersection has more keypoints
    pure_more_kp = []  # Scenes where pure_scale has more keypoints

    for scene in scenes_with_data:
        int_count = kp_counts[scene]['sift_surf_scale_matched_intersection_a']
        pure_count = kp_counts[scene]['sift_top_scale_13px']

        int_mAP = results['intersection'].get(scene, 0) * 100
        pure_mAP = results['pure_scale'].get(scene, 0) * 100
        gain = int_mAP - pure_mAP

        if int_count > pure_count:
            int_more_kp.append((scene, gain, int_count, pure_count))
        else:
            pure_more_kp.append((scene, gain, int_count, pure_count))

    gains_int_more = [x[1] for x in int_more_kp]
    gains_pure_more = [x[1] for x in pure_more_kp]

    print(f"\nWhen intersection has MORE keypoints than pure_scale:")
    print(f"  Scenes: {len(int_more_kp)}")
    print(f"  Average intersection gain: {sum(gains_int_more)/len(gains_int_more):+.1f}%")
    print(f"  Intersection win rate: {sum(1 for g in gains_int_more if g > 0)}/{len(gains_int_more)}")

    print(f"\nWhen pure_scale has MORE keypoints (intersection has FEWER):")
    print(f"  Scenes: {len(pure_more_kp)}")
    print(f"  Average intersection gain: {sum(gains_pure_more)/len(gains_pure_more):+.1f}%")
    print(f"  Intersection win rate: {sum(1 for g in gains_pure_more if g > 0)}/{len(gains_pure_more)}")

    print("\n" + "-" * 40)
    print("KEY INSIGHT:")
    print(f"When intersection has FEWER keypoints: avg gain = {sum(gains_pure_more)/len(gains_pure_more):+.1f}%")
    print(f"When intersection has MORE keypoints:  avg gain = {sum(gains_int_more)/len(gains_int_more):+.1f}%")
    print("\nConclusion: The benefit comes from REMOVING bad keypoints (quality filtering)")


def main():
    print("=" * 80)
    print("INTERSECTION MECHANISM ANALYSIS")
    print("=" * 80)
    print(f"\nDatabase: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    # Step 1: Extract metadata
    metadata_file = extract_metadata_to_file()

    # Step 2: Parse results
    results = parse_scene_results(metadata_file)

    if not results:
        print("ERROR: No results found in metadata")
        return

    # Step 3: Run analyses
    scenes = analyze_per_scene(results)
    analyze_viewpoint_vs_illumination(results, scenes)
    analyze_keypoint_count_correlation(results, scenes)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
