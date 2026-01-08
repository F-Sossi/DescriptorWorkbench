#!/usr/bin/env python3
"""
Repeatability Analysis Script

Computes keypoint repeatability across image transformations to understand
why detector intersection provides better matching performance.

Usage:
    cd /home/frank/repos/DescriptorWorkbench
    python3 analysis/scripts/analyze_repeatability.py

Output:
    - logs/repeatability_results.csv
    - logs/repeatability_summary.txt
    - Console output with analysis

Author: Investigation 2 - Repeatability Tracking
Date: December 2025
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import os

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "build" / "experiments.db"
DATA_PATH = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Keypoint sets to analyze
KEYPOINT_SETS = [
    'sift_surf_scale_matched_intersection_a',
    'sift_top_scale_13px',
    'sift_scale_only_13px'
]

# Repeatability tolerance in pixels
TOLERANCE = 3.0


def load_homography(scene_path, pair):
    """
    Load homography matrix from HPatches scene.

    Args:
        scene_path: Path to scene folder
        pair: Target image number (2-6)

    Returns:
        3x3 numpy array homography matrix
    """
    h_file = scene_path / f'H_1_{pair}'
    if not h_file.exists():
        return None

    H = np.loadtxt(h_file).reshape(3, 3)
    return H


def load_keypoints_from_db(db_path, keypoint_set_name, scene_name, image_name):
    """
    Load keypoints from database for a specific image.

    Returns:
        List of dicts with x, y, size, angle, response
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT k.x, k.y, k.size, k.angle, k.response
        FROM locked_keypoints k
        JOIN keypoint_sets s ON k.keypoint_set_id = s.id
        WHERE s.name = ?
          AND k.scene_name = ?
          AND k.image_name = ?
    """, (keypoint_set_name, scene_name, image_name))

    keypoints = []
    for row in cursor.fetchall():
        keypoints.append({
            'x': row[0],
            'y': row[1],
            'size': row[2],
            'angle': row[3],
            'response': row[4]
        })

    conn.close()
    return keypoints


def get_all_scenes(data_path):
    """Get list of all HPatches scenes."""
    scenes = []
    for item in os.listdir(data_path):
        scene_path = data_path / item
        if scene_path.is_dir() and (item.startswith('v_') or item.startswith('i_')):
            # Verify it has homography files
            if (scene_path / 'H_1_2').exists():
                scenes.append(item)
    return sorted(scenes)


def project_keypoint(kp, H):
    """
    Project keypoint location using homography.

    Args:
        kp: Dict with x, y
        H: 3x3 homography matrix

    Returns:
        Tuple (x, y) of projected location
    """
    pt = np.array([kp['x'], kp['y'], 1.0])
    pt_proj = H @ pt
    pt_proj = pt_proj[:2] / pt_proj[2]
    return pt_proj[0], pt_proj[1]


def is_in_bounds(x, y, width=800, height=600, margin=10):
    """Check if point is within image bounds with margin."""
    return margin <= x < width - margin and margin <= y < height - margin


def compute_repeatability(kp_ref, kp_target, H, tolerance=3.0):
    """
    Compute repeatability between reference and target keypoints.

    Args:
        kp_ref: List of keypoints in reference image
        kp_target: List of keypoints in target image
        H: Homography from reference to target
        tolerance: Pixel distance tolerance

    Returns:
        Dict with repeatability metrics
    """
    if not kp_ref or not kp_target:
        return {
            'repeatability': 0.0,
            'num_valid': 0,
            'num_repeated': 0,
            'mean_error': float('nan'),
            'median_error': float('nan')
        }

    # Build KD-tree-like structure for target keypoints (simple version)
    target_pts = np.array([[kp['x'], kp['y']] for kp in kp_target])

    valid = 0
    repeated = 0
    errors = []

    for kp in kp_ref:
        # Project to target image
        try:
            proj_x, proj_y = project_keypoint(kp, H)
        except:
            continue

        # Check if in bounds
        if not is_in_bounds(proj_x, proj_y):
            continue

        valid += 1

        # Find nearest keypoint in target
        proj_pt = np.array([proj_x, proj_y])
        distances = np.linalg.norm(target_pts - proj_pt, axis=1)
        min_dist = np.min(distances)
        errors.append(min_dist)

        if min_dist < tolerance:
            repeated += 1

    repeatability = repeated / valid if valid > 0 else 0.0
    mean_error = np.mean(errors) if errors else float('nan')
    median_error = np.median(errors) if errors else float('nan')

    return {
        'repeatability': repeatability,
        'num_valid': valid,
        'num_repeated': repeated,
        'mean_error': mean_error,
        'median_error': median_error
    }


def analyze_keypoint_set(db_path, data_path, keypoint_set_name, scenes):
    """
    Analyze repeatability for a keypoint set across all scenes.

    Returns:
        DataFrame with per-scene, per-pair results
    """
    results = []

    for scene in scenes:
        scene_path = data_path / scene

        # Load reference keypoints (image 1)
        kp_ref = load_keypoints_from_db(db_path, keypoint_set_name, scene, '1.ppm')

        if not kp_ref:
            continue

        for pair in range(2, 7):
            # Load homography
            H = load_homography(scene_path, pair)
            if H is None:
                continue

            # Load target keypoints
            kp_target = load_keypoints_from_db(db_path, keypoint_set_name, scene, f'{pair}.ppm')

            # Compute repeatability
            metrics = compute_repeatability(kp_ref, kp_target, H, TOLERANCE)

            results.append({
                'keypoint_set': keypoint_set_name,
                'scene': scene,
                'scene_type': 'viewpoint' if scene.startswith('v_') else 'illumination',
                'pair': pair,
                'repeatability': metrics['repeatability'],
                'num_valid': metrics['num_valid'],
                'num_repeated': metrics['num_repeated'],
                'mean_error': metrics['mean_error'],
                'median_error': metrics['median_error'],
                'num_ref_keypoints': len(kp_ref)
            })

    return pd.DataFrame(results)


def compare_keypoint_sets(results_dict):
    """
    Compare repeatability across keypoint sets.

    Args:
        results_dict: Dict mapping keypoint_set_name -> DataFrame

    Returns:
        Summary statistics
    """
    summary = []

    for kp_set, df in results_dict.items():
        # Short name for display
        if 'intersection' in kp_set:
            short_name = 'intersection'
        elif 'top_scale' in kp_set:
            short_name = 'pure_scale'
        elif 'scale_only' in kp_set:
            short_name = 'scale_only'
        else:
            short_name = kp_set

        # Overall statistics
        summary.append({
            'keypoint_set': short_name,
            'full_name': kp_set,
            'mean_repeatability': df['repeatability'].mean(),
            'median_repeatability': df['repeatability'].median(),
            'std_repeatability': df['repeatability'].std(),
            'mean_error': df['mean_error'].mean(),
            'num_scenes': df['scene'].nunique(),
            'total_pairs': len(df)
        })

    return pd.DataFrame(summary)


def analyze_by_severity(results_dict):
    """
    Analyze repeatability by transformation severity (pair number).
    """
    severity_data = []

    for kp_set, df in results_dict.items():
        if 'intersection' in kp_set:
            short_name = 'intersection'
        elif 'top_scale' in kp_set:
            short_name = 'pure_scale'
        else:
            short_name = 'scale_only'

        for pair in range(2, 7):
            pair_df = df[df['pair'] == pair]
            severity_data.append({
                'keypoint_set': short_name,
                'pair': pair,
                'mean_repeatability': pair_df['repeatability'].mean(),
                'std_repeatability': pair_df['repeatability'].std()
            })

    return pd.DataFrame(severity_data)


def analyze_by_scene_type(results_dict):
    """
    Analyze repeatability by scene type (viewpoint vs illumination).
    """
    type_data = []

    for kp_set, df in results_dict.items():
        if 'intersection' in kp_set:
            short_name = 'intersection'
        elif 'top_scale' in kp_set:
            short_name = 'pure_scale'
        else:
            short_name = 'scale_only'

        for scene_type in ['viewpoint', 'illumination']:
            type_df = df[df['scene_type'] == scene_type]
            type_data.append({
                'keypoint_set': short_name,
                'scene_type': scene_type,
                'mean_repeatability': type_df['repeatability'].mean(),
                'std_repeatability': type_df['repeatability'].std(),
                'num_scenes': type_df['scene'].nunique()
            })

    return pd.DataFrame(type_data)


def main():
    print("=" * 70)
    print("REPEATABILITY ANALYSIS")
    print("=" * 70)
    print(f"\nDatabase: {DB_PATH}")
    print(f"Data path: {DATA_PATH}")
    print(f"Tolerance: {TOLERANCE}px")

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    # Get all scenes
    scenes = get_all_scenes(DATA_PATH)
    print(f"\nFound {len(scenes)} scenes")

    # Analyze each keypoint set
    results_dict = {}
    for kp_set in KEYPOINT_SETS:
        print(f"\nAnalyzing: {kp_set}")
        df = analyze_keypoint_set(DB_PATH, DATA_PATH, kp_set, scenes)
        results_dict[kp_set] = df
        print(f"  Processed {len(df)} scene-pair combinations")

    # Create output directory
    LOGS_DIR.mkdir(exist_ok=True)

    # Save detailed results
    all_results = pd.concat(results_dict.values(), ignore_index=True)
    all_results.to_csv(LOGS_DIR / 'repeatability_results.csv', index=False)
    print(f"\nSaved detailed results to logs/repeatability_results.csv")

    # Compute and display summary
    print("\n" + "=" * 70)
    print("SUMMARY: OVERALL REPEATABILITY")
    print("=" * 70)

    summary = compare_keypoint_sets(results_dict)
    print(f"\n{'Keypoint Set':<15} {'Mean Rep':>12} {'Median Rep':>12} {'Std':>10} {'Mean Err':>10}")
    print("-" * 60)
    for _, row in summary.iterrows():
        print(f"{row['keypoint_set']:<15} {row['mean_repeatability']*100:>11.2f}% "
              f"{row['median_repeatability']*100:>11.2f}% {row['std_repeatability']*100:>9.2f}% "
              f"{row['mean_error']:>9.2f}px")

    # By severity
    print("\n" + "=" * 70)
    print("REPEATABILITY BY TRANSFORMATION SEVERITY")
    print("=" * 70)

    severity = analyze_by_severity(results_dict)
    severity_pivot = severity.pivot(index='pair', columns='keypoint_set', values='mean_repeatability')
    print(f"\n{'Pair':<6}", end='')
    for col in ['intersection', 'pure_scale', 'scale_only']:
        if col in severity_pivot.columns:
            print(f"{col:>15}", end='')
    print()
    print("-" * 52)
    for pair in range(2, 7):
        print(f"{pair:<6}", end='')
        for col in ['intersection', 'pure_scale', 'scale_only']:
            if col in severity_pivot.columns:
                val = severity_pivot.loc[pair, col] * 100
                print(f"{val:>14.1f}%", end='')
        print()

    # By scene type
    print("\n" + "=" * 70)
    print("REPEATABILITY BY SCENE TYPE")
    print("=" * 70)

    by_type = analyze_by_scene_type(results_dict)
    print(f"\n{'Set':<15} {'Type':<12} {'Mean Rep':>12}")
    print("-" * 42)
    for _, row in by_type.iterrows():
        print(f"{row['keypoint_set']:<15} {row['scene_type']:<12} {row['mean_repeatability']*100:>11.2f}%")

    # Compute intersection advantage
    print("\n" + "=" * 70)
    print("INTERSECTION ADVANTAGE")
    print("=" * 70)

    int_df = results_dict['sift_surf_scale_matched_intersection_a']
    pure_df = results_dict['sift_top_scale_13px']

    # Merge on scene and pair
    merged = int_df.merge(
        pure_df,
        on=['scene', 'pair'],
        suffixes=('_int', '_pure')
    )

    merged['rep_diff'] = merged['repeatability_int'] - merged['repeatability_pure']

    print(f"\nMean repeatability difference: {merged['rep_diff'].mean()*100:+.2f}%")
    print(f"Intersection wins: {(merged['rep_diff'] > 0).sum()}/{len(merged)} "
          f"({100*(merged['rep_diff'] > 0).mean():.1f}%)")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(merged['repeatability_int'], merged['repeatability_pure'])
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.6f}")

    # Save summary
    with open(LOGS_DIR / 'repeatability_summary.txt', 'w') as f:
        f.write("REPEATABILITY ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tolerance: {TOLERANCE}px\n")
        f.write(f"Scenes analyzed: {len(scenes)}\n\n")

        f.write("Overall Repeatability:\n")
        for _, row in summary.iterrows():
            f.write(f"  {row['keypoint_set']}: {row['mean_repeatability']*100:.2f}%\n")

        f.write(f"\nIntersection advantage: {merged['rep_diff'].mean()*100:+.2f}%\n")
        f.write(f"Intersection wins: {(merged['rep_diff'] > 0).sum()}/{len(merged)}\n")
        f.write(f"Paired t-test p-value: {p_value:.6f}\n")

    print(f"\nSaved summary to logs/repeatability_summary.txt")
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
