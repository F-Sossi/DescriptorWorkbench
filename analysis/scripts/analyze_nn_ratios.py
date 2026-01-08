#!/usr/bin/env python3
"""
NN Ratio Analysis Script

Analyzes descriptor distinctiveness by computing NN1/NN2 ratios for each
keypoint set to understand why intersection achieves higher mAP.

Usage:
    cd /home/frank/repos/DescriptorWorkbench
    python3 analysis/scripts/analyze_nn_ratios.py

Prerequisites:
    - Run nn_ratio_study.yaml experiment with save_descriptors: true

Output:
    - logs/nn_ratio_results.csv
    - logs/nn_ratio_summary.txt
    - Console output with analysis

Author: Investigation 1 - NN Ratio Analysis
Date: December 2025
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import struct

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "build" / "experiments.db"
DATA_PATH = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Experiments to analyze
EXPERIMENTS = {
    'intersection': 'dspsift_v2__intersection__nn_study',
    'pure_scale': 'dspsift_v2__pure_scale__nn_study',
    'scale_only': 'dspsift_v2__scale_only__nn_study'
}


def load_homography(scene_path, pair):
    """Load homography matrix from HPatches scene."""
    h_file = scene_path / f'H_1_{pair}'
    if not h_file.exists():
        return None
    return np.loadtxt(h_file).reshape(3, 3)


def blob_to_descriptor(blob, dimension=128):
    """Convert BLOB to numpy array."""
    # Assuming float32 storage
    return np.frombuffer(blob, dtype=np.float32)


def load_descriptors_for_image(db_path, experiment_name, scene_name, image_name):
    """
    Load descriptors for a specific image.

    Returns:
        List of dicts with x, y, descriptor
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT d.keypoint_x, d.keypoint_y, d.descriptor_vector, d.descriptor_dimension
        FROM descriptors d
        JOIN experiments e ON d.experiment_id = e.id
        WHERE e.descriptor_type = ?
          AND d.scene_name = ?
          AND d.image_name = ?
    """, (experiment_name, scene_name, image_name))

    results = []
    for row in cursor.fetchall():
        x, y, blob, dim = row
        desc = blob_to_descriptor(blob, dim)
        results.append({
            'x': x,
            'y': y,
            'descriptor': desc
        })

    conn.close()
    return results


def compute_nn_ratios(desc_query, desc_target, H=None, correct_threshold=3.0):
    """
    Compute NN1/NN2 ratios for query descriptors against target.

    Args:
        desc_query: List of dicts with x, y, descriptor
        desc_target: List of dicts with x, y, descriptor
        H: Optional homography for ground truth checking
        correct_threshold: Pixel threshold for correct match

    Returns:
        List of dicts with ratio, is_correct, nn1_dist, nn2_dist
    """
    if len(desc_query) < 2 or len(desc_target) < 2:
        return []

    # Build descriptor matrices
    query_descs = np.array([d['descriptor'] for d in desc_query])
    target_descs = np.array([d['descriptor'] for d in desc_target])

    # Compute pairwise L2 distances
    # Using efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    query_sq = np.sum(query_descs**2, axis=1, keepdims=True)
    target_sq = np.sum(target_descs**2, axis=1, keepdims=True)
    distances = np.sqrt(np.maximum(
        query_sq + target_sq.T - 2 * query_descs @ target_descs.T,
        0
    ))

    results = []
    for i, q in enumerate(desc_query):
        dists = distances[i]
        sorted_indices = np.argsort(dists)

        nn1_idx = sorted_indices[0]
        nn2_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]

        nn1_dist = dists[nn1_idx]
        nn2_dist = dists[nn2_idx]

        # Compute ratio (avoid division by zero)
        ratio = nn1_dist / nn2_dist if nn2_dist > 1e-10 else 1.0

        # Check if match is correct using homography
        is_correct = None
        if H is not None:
            # Project query point to target
            pt = np.array([q['x'], q['y'], 1.0])
            pt_proj = H @ pt
            pt_proj = pt_proj[:2] / pt_proj[2]

            # Check distance to matched point
            matched = desc_target[nn1_idx]
            match_dist = np.sqrt((matched['x'] - pt_proj[0])**2 +
                                 (matched['y'] - pt_proj[1])**2)
            is_correct = match_dist < correct_threshold

        results.append({
            'ratio': ratio,
            'nn1_dist': nn1_dist,
            'nn2_dist': nn2_dist,
            'is_correct': is_correct
        })

    return results


def get_all_scenes(data_path):
    """Get list of all HPatches scenes."""
    scenes = []
    for item in data_path.iterdir():
        if item.is_dir() and (item.name.startswith('v_') or item.name.startswith('i_')):
            if (item / 'H_1_2').exists():
                scenes.append(item.name)
    return sorted(scenes)


def analyze_experiment(db_path, data_path, experiment_name, short_name, scenes, sample_pairs=5):
    """
    Analyze NN ratios for an experiment.

    Args:
        sample_pairs: Number of image pairs per scene to analyze (for speed)
    """
    all_ratios = []

    for scene in scenes:
        scene_path = data_path / scene

        # Load reference descriptors
        desc_ref = load_descriptors_for_image(db_path, experiment_name, scene, '1.ppm')

        if not desc_ref:
            continue

        # Analyze pairs
        for pair in range(2, min(2 + sample_pairs, 7)):
            H = load_homography(scene_path, pair)
            if H is None:
                continue

            desc_target = load_descriptors_for_image(
                db_path, experiment_name, scene, f'{pair}.ppm'
            )

            if not desc_target:
                continue

            # Compute ratios
            ratios = compute_nn_ratios(desc_ref, desc_target, H)

            for r in ratios:
                r['scene'] = scene
                r['pair'] = pair
                r['keypoint_set'] = short_name
                r['scene_type'] = 'viewpoint' if scene.startswith('v_') else 'illumination'

            all_ratios.extend(ratios)

    return pd.DataFrame(all_ratios)


def main():
    print("=" * 70)
    print("NN RATIO ANALYSIS")
    print("=" * 70)
    print(f"\nDatabase: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    # Check descriptor counts
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.descriptor_type, COUNT(*)
        FROM descriptors d
        JOIN experiments e ON d.experiment_id = e.id
        GROUP BY e.descriptor_type
    """)
    print("\nDescriptor counts:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")
    conn.close()

    # Get scenes
    scenes = get_all_scenes(DATA_PATH)
    print(f"\nFound {len(scenes)} scenes")

    # Analyze each experiment
    results_dict = {}
    for short_name, exp_name in EXPERIMENTS.items():
        print(f"\nAnalyzing: {short_name}")
        df = analyze_experiment(DB_PATH, DATA_PATH, exp_name, short_name, scenes)
        results_dict[short_name] = df
        print(f"  Computed {len(df):,} NN ratios")

    # Combine results
    all_results = pd.concat(results_dict.values(), ignore_index=True)

    # Save detailed results
    LOGS_DIR.mkdir(exist_ok=True)
    all_results.to_csv(LOGS_DIR / 'nn_ratio_results.csv', index=False)
    print(f"\nSaved detailed results to logs/nn_ratio_results.csv")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: NN RATIO DISTRIBUTION")
    print("=" * 70)

    print(f"\n{'Keypoint Set':<15} {'Mean Ratio':>12} {'Median':>10} {'Std':>10} {'<0.8':>10} {'<0.7':>10}")
    print("-" * 70)

    summary_data = []
    for name in ['intersection', 'pure_scale', 'scale_only']:
        df = results_dict[name]
        ratios = df['ratio']

        mean_r = ratios.mean()
        median_r = ratios.median()
        std_r = ratios.std()
        pct_08 = (ratios < 0.8).mean() * 100
        pct_07 = (ratios < 0.7).mean() * 100

        print(f"{name:<15} {mean_r:>12.4f} {median_r:>10.4f} {std_r:>10.4f} {pct_08:>9.1f}% {pct_07:>9.1f}%")

        summary_data.append({
            'keypoint_set': name,
            'mean_ratio': mean_r,
            'median_ratio': median_r,
            'std_ratio': std_r,
            'pct_below_0.8': pct_08,
            'pct_below_0.7': pct_07
        })

    # Analyze correct vs incorrect matches
    print("\n" + "=" * 70)
    print("NN RATIO BY MATCH CORRECTNESS")
    print("=" * 70)

    print(f"\n{'Keypoint Set':<15} {'Correct Mean':>14} {'Incorrect Mean':>16} {'Difference':>12}")
    print("-" * 60)

    for name in ['intersection', 'pure_scale', 'scale_only']:
        df = results_dict[name]
        correct = df[df['is_correct'] == True]['ratio']
        incorrect = df[df['is_correct'] == False]['ratio']

        if len(correct) > 0 and len(incorrect) > 0:
            correct_mean = correct.mean()
            incorrect_mean = incorrect.mean()
            diff = incorrect_mean - correct_mean
            print(f"{name:<15} {correct_mean:>14.4f} {incorrect_mean:>16.4f} {diff:>+12.4f}")

    # By scene type
    print("\n" + "=" * 70)
    print("NN RATIO BY SCENE TYPE")
    print("=" * 70)

    print(f"\n{'Set':<15} {'Viewpoint':>12} {'Illumination':>14}")
    print("-" * 45)

    for name in ['intersection', 'pure_scale', 'scale_only']:
        df = results_dict[name]
        v_mean = df[df['scene_type'] == 'viewpoint']['ratio'].mean()
        i_mean = df[df['scene_type'] == 'illumination']['ratio'].mean()
        print(f"{name:<15} {v_mean:>12.4f} {i_mean:>14.4f}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON: INTERSECTION vs PURE_SCALE")
    print("=" * 70)

    int_ratios = results_dict['intersection']['ratio']
    pure_ratios = results_dict['pure_scale']['ratio']

    from scipy import stats

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(int_ratios, pure_ratios, alternative='less')
    print(f"\nMann-Whitney U test (intersection < pure_scale):")
    print(f"  U-statistic: {u_stat:,.0f}")
    print(f"  p-value: {u_pvalue:.6f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((int_ratios.std()**2 + pure_ratios.std()**2) / 2)
    cohens_d = (int_ratios.mean() - pure_ratios.mean()) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

    # Save summary
    with open(LOGS_DIR / 'nn_ratio_summary.txt', 'w') as f:
        f.write("NN RATIO ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        for data in summary_data:
            f.write(f"{data['keypoint_set']}:\n")
            f.write(f"  Mean ratio: {data['mean_ratio']:.4f}\n")
            f.write(f"  Median ratio: {data['median_ratio']:.4f}\n")
            f.write(f"  % below 0.8: {data['pct_below_0.8']:.1f}%\n")
            f.write(f"  % below 0.7: {data['pct_below_0.7']:.1f}%\n\n")

        f.write(f"Mann-Whitney p-value: {u_pvalue:.6f}\n")
        f.write(f"Cohen's d: {cohens_d:.4f}\n")

    print(f"\nSaved summary to logs/nn_ratio_summary.txt")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    int_mean = results_dict['intersection']['ratio'].mean()
    pure_mean = results_dict['pure_scale']['ratio'].mean()

    if int_mean < pure_mean:
        print(f"""
    Intersection has LOWER mean NN ratio ({int_mean:.4f} vs {pure_mean:.4f})
    This means intersection descriptors are MORE DISTINCTIVE!

    Lower ratio = NN1 much closer than NN2 = more confident matches
    """)
    else:
        print(f"""
    Intersection has HIGHER mean NN ratio ({int_mean:.4f} vs {pure_mean:.4f})
    This means intersection descriptors are LESS distinctive.

    The mechanism must be something other than descriptor distinctiveness.
    """)

    print("=" * 70)


if __name__ == "__main__":
    main()
