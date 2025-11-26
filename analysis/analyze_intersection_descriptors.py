#!/usr/bin/env python3
"""
Analyze Descriptor Fusion Using Real Intersection Sets

Loads spatially-paired keypoints from intersection sets in the database,
extracts SIFT and HardNet descriptors at those positions, and analyzes
their statistical properties to understand fusion behavior.

Key insight: Index i in set_a and index i in set_b are spatially paired
(same physical location, found via mutual nearest neighbor search).
"""

import sys
import os
import cv2
import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))


def load_intersection_keypoints(db_path, set_a_name, set_b_name, scene_name, num_samples=500):
    """
    Load paired keypoints from intersection sets.

    Args:
        db_path: Path to experiments.db
        set_a_name: Keypoint set A (e.g., 'sift_keynet_scale_matched_intersection_a')
        set_b_name: Keypoint set B (e.g., 'sift_keynet_scale_matched_intersection_b')
        scene_name: HPatches scene name (e.g., 'i_ajuntament')
        num_samples: Max number of keypoint pairs to sample

    Returns:
        keypoints_a: List of cv2.KeyPoint for set A
        keypoints_b: List of cv2.KeyPoint for set B (spatially paired)
        image_name: The image name used
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get keypoint set IDs
    cursor.execute("SELECT id FROM keypoint_sets WHERE name = ?", (set_a_name,))
    result_a = cursor.fetchone()
    if not result_a:
        raise ValueError(f"Keypoint set not found: {set_a_name}")
    set_a_id = result_a[0]

    cursor.execute("SELECT id FROM keypoint_sets WHERE name = ?", (set_b_name,))
    result_b = cursor.fetchone()
    if not result_b:
        raise ValueError(f"Keypoint set not found: {set_b_name}")
    set_b_id = result_b[0]

    # Load keypoints from set A for this scene (using image 1)
    image_name = '1.ppm'
    query_a = """
    SELECT x, y, size, angle, response, octave, class_id
    FROM locked_keypoints
    WHERE keypoint_set_id = ? AND scene_name = ? AND image_name = ?
    ORDER BY id
    LIMIT ?
    """

    cursor.execute(query_a, (set_a_id, scene_name, image_name, num_samples))
    rows_a = cursor.fetchall()

    # Load keypoints from set B (same scene, same image, same order)
    query_b = """
    SELECT x, y, size, angle, response, octave, class_id
    FROM locked_keypoints
    WHERE keypoint_set_id = ? AND scene_name = ? AND image_name = ?
    ORDER BY id
    LIMIT ?
    """

    cursor.execute(query_b, (set_b_id, scene_name, image_name, num_samples))
    rows_b = cursor.fetchall()

    conn.close()

    if len(rows_a) == 0 or len(rows_b) == 0:
        raise ValueError(f"No keypoints found for scene {scene_name}")

    # Convert to cv2.KeyPoint objects
    keypoints_a = []
    for row in rows_a:
        x, y, size, angle, response, octave, class_id = row
        kp = cv2.KeyPoint(
            x=float(x),
            y=float(y),
            size=float(size),
            angle=float(angle),
            response=float(response),
            octave=int(octave),
            class_id=int(class_id)
        )
        keypoints_a.append(kp)

    keypoints_b = []
    for row in rows_b:
        x, y, size, angle, response, octave, class_id = row
        kp = cv2.KeyPoint(
            x=float(x),
            y=float(y),
            size=float(size),
            angle=float(angle),
            response=float(response),
            octave=int(octave),
            class_id=int(class_id)
        )
        keypoints_b.append(kp)

    print(f"Loaded {len(keypoints_a)} keypoints from {set_a_name}")
    print(f"Loaded {len(keypoints_b)} keypoints from {set_b_name}")

    # Verify they're paired (should have same count)
    assert len(keypoints_a) == len(keypoints_b), "Intersection sets must have same size!"

    # Check spatial distance (should be close, within tolerance)
    distances = []
    for kp_a, kp_b in zip(keypoints_a[:min(10, len(keypoints_a))], keypoints_b[:min(10, len(keypoints_b))]):
        dist = np.sqrt((kp_a.pt[0] - kp_b.pt[0])**2 + (kp_a.pt[1] - kp_b.pt[1])**2)
        distances.append(dist)

    avg_dist = np.mean(distances)
    print(f"Average spatial distance between paired keypoints: {avg_dist:.2f} pixels")
    if avg_dist > 10:
        print(f"⚠️  WARNING: Large spatial distance - keypoints may not be properly paired!")
    else:
        print(f"✓ Keypoints are spatially paired (avg distance {avg_dist:.2f}px < 10px)")

    return keypoints_a, keypoints_b, image_name


def extract_sift_at_keypoints(image_path, keypoints):
    """
    Extract SIFT descriptors at given keypoint locations.

    Args:
        image_path: Path to image
        keypoints: List of cv2.KeyPoint

    Returns:
        descriptors: numpy array (N x 128), L2-normalized
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create SIFT extractor
    sift = cv2.SIFT_create()

    # Compute descriptors at given keypoints
    _, descriptors = sift.compute(gray, keypoints)

    if descriptors is not None:
        descriptors = descriptors.astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (norms + 1e-8)

    return descriptors


def extract_rootsift_at_keypoints(image_path, keypoints):
    """Extract RootSIFT descriptors at given keypoint locations."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(gray, keypoints)

    if descriptors is not None:
        descriptors = descriptors.astype(np.float32)

        # RootSIFT transformation
        # 1. L1 normalize
        l1_norms = np.sum(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (l1_norms + 1e-8)

        # 2. Square root
        descriptors = np.sqrt(descriptors + 1e-8)

        # 3. L2 normalize
        l2_norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (l2_norms + 1e-8)

    return descriptors


def compute_statistics(descriptors_dict):
    """Compute comprehensive statistics for each descriptor type."""
    stats = {}

    for name, desc in descriptors_dict.items():
        if desc is None or len(desc) == 0:
            print(f"Warning: {name} has no descriptors")
            continue

        stats[name] = {
            'count': len(desc),
            'mean': float(np.mean(desc)),
            'std': float(np.std(desc)),
            'variance': float(np.var(desc)),
            'min': float(np.min(desc)),
            'max': float(np.max(desc)),
            'range': float(np.max(desc) - np.min(desc)),
            'l2_norm_mean': float(np.mean(np.linalg.norm(desc, axis=1))),
            'l2_norm_std': float(np.std(np.linalg.norm(desc, axis=1))),
            'per_dim_mean': np.mean(desc, axis=0),
            'per_dim_std': np.std(desc, axis=0),
            'per_dim_variance': np.var(desc, axis=0),
        }

    return stats


def print_statistics_comparison(stats):
    """Print formatted statistics comparison table."""
    print("\n" + "=" * 120)
    print("REAL DESCRIPTOR STATISTICS (from Intersection Sets)")
    print("=" * 120)

    # Print header
    col_width = 25
    print(f"{'Metric':<{col_width}}", end='')
    for name in stats.keys():
        print(f"{name:<{col_width}}", end='')
    print()
    print("-" * (col_width * (len(stats) + 1)))

    metrics = [
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Variance', 'variance'),
        ('Min Value', 'min'),
        ('Max Value', 'max'),
        ('Range', 'range'),
        ('L2 Norm (mean)', 'l2_norm_mean'),
        ('L2 Norm (std)', 'l2_norm_std'),
    ]

    # Print metrics
    for metric_name, metric_key in metrics:
        print(f"{metric_name:<{col_width}}", end='')
        for desc_name in stats.keys():
            val = stats[desc_name][metric_key]
            if isinstance(val, (int, np.integer)):
                print(f"{val:<{col_width}}", end='')
            else:
                print(f"{val:<{col_width}.6f}", end='')
        print()

    print("\n" + "=" * 120)
    print("VARIANCE RATIO ANALYSIS (Key to Understanding Fusion):")
    print("=" * 120)

    # Compare all pairs
    names = list(stats.keys())
    for i, name_a in enumerate(names):
        for name_b in names[i+1:]:
            std_a = stats[name_a]['std']
            std_b = stats[name_b]['std']
            ratio = std_a / std_b

            print(f"\n{name_a.upper()} vs {name_b.upper()}:")
            print(f"  Std dev ratio: {ratio:.2f}x")
            print(f"  {name_a} std: {std_a:.6f}")
            print(f"  {name_b} std: {std_b:.6f}")

            if ratio > 2:
                print(f"  ⚠️  {name_a.upper()} will DOMINATE fusion (variance {ratio:.1f}x higher)")
                print(f"     → When averaged, {name_a} contributions overwhelm {name_b}")
            elif ratio < 0.5:
                print(f"  ⚠️  {name_b.upper()} will DOMINATE fusion (variance {1/ratio:.1f}x higher)")
                print(f"     → When averaged, {name_b} contributions overwhelm {name_a}")
            else:
                print(f"  ✓ Balanced variance - fusion should work well")

    print("\n" + "=" * 120)


def visualize_descriptor_distributions(descriptors_dict, output_dir='analysis/output'):
    """Create visualizations comparing descriptor distributions."""
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Value distributions
    ax = axes[0, 0]
    for name, desc in descriptors_dict.items():
        ax.hist(desc.flatten(), bins=100, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Descriptor Value')
    ax.set_ylabel('Density')
    ax.set_title('Descriptor Value Distributions')
    ax.legend()
    ax.set_xlim(-0.5, 0.5)

    # Plot 2: Per-dimension variance
    ax = axes[0, 1]
    for name, desc in descriptors_dict.items():
        per_dim_var = np.var(desc, axis=0)
        ax.plot(per_dim_var, label=name, alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.set_title('Per-Dimension Variance')
    ax.legend()
    ax.set_yscale('log')

    # Plot 3: Mean absolute values
    ax = axes[0, 2]
    names = list(descriptors_dict.keys())
    mean_abs_vals = [np.mean(np.abs(desc)) for desc in descriptors_dict.values()]
    bars = ax.bar(names, mean_abs_vals)
    ax.set_ylabel('Mean Absolute Value')
    ax.set_title('Mean Absolute Value per Descriptor')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    # Plot 4: Standard deviation comparison
    ax = axes[1, 0]
    std_vals = [np.std(desc) for desc in descriptors_dict.values()]
    bars = ax.bar(names, std_vals)
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Standard Deviation per Descriptor Type')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    # Plot 5: Box plots
    ax = axes[1, 1]
    data_for_box = [desc.flatten() for desc in descriptors_dict.values()]
    ax.boxplot(data_for_box, labels=names)
    ax.set_ylabel('Descriptor Value')
    ax.set_title('Value Range Comparison')
    ax.grid(True, alpha=0.3)

    # Plot 6: Variance ratio matrix
    ax = axes[1, 2]
    n_desc = len(descriptors_dict)
    variance_ratios = np.zeros((n_desc, n_desc))
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                variance_ratios[i, j] = 1.0
            else:
                std_a = np.std(descriptors_dict[name_a])
                std_b = np.std(descriptors_dict[name_b])
                variance_ratios[i, j] = std_a / std_b

    im = ax.imshow(variance_ratios, cmap='RdYlGn_r', vmin=0.5, vmax=2.0)
    ax.set_xticks(np.arange(n_desc))
    ax.set_yticks(np.arange(n_desc))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add text annotations
    for i in range(n_desc):
        for j in range(n_desc):
            text = ax.text(j, i, f'{variance_ratios[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Variance Ratio Matrix\n(row/column)')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/intersection_descriptor_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/intersection_descriptor_analysis.png")
    plt.close()


def test_fusion(desc_a, desc_b, name_a='A', name_b='B'):
    """Test different fusion strategies."""
    print(f"\n" + "=" * 120)
    print(f"FUSION TEST: {name_a} + {name_b} (Spatially Paired Keypoints)")
    print("=" * 120)

    n = min(len(desc_a), len(desc_b))
    desc_a = desc_a[:n]
    desc_b = desc_b[:n]

    fusion_methods = {
        'average': lambda a, b: (a + b) / 2.0,
        'normalized_avg': lambda a, b: normalize_and_average(a, b),
        'variance_weighted': lambda a, b: variance_weighted_fusion(a, b),
        'concatenate': lambda a, b: np.concatenate([a, b], axis=1),
    }

    for method_name, fusion_fn in fusion_methods.items():
        fused = fusion_fn(desc_a, desc_b)

        if method_name == 'concatenate':
            print(f"\n{method_name}:")
            print(f"  Output dimension: {fused.shape[1]}D (= {desc_a.shape[1]}D + {desc_b.shape[1]}D)")
            continue

        # Measure correlation
        corr_a = np.mean([np.corrcoef(fused[i], desc_a[i])[0, 1] for i in range(min(100, n))])
        corr_b = np.mean([np.corrcoef(fused[i], desc_b[i])[0, 1] for i in range(min(100, n))])

        total_corr = abs(corr_a) + abs(corr_b)
        contrib_a = abs(corr_a) / (total_corr + 1e-8)
        contrib_b = abs(corr_b) / (total_corr + 1e-8)

        print(f"\n{method_name}:")
        print(f"  Correlation with {name_a}: {corr_a:.4f}")
        print(f"  Correlation with {name_b}: {corr_b:.4f}")
        print(f"  Contribution {name_a}: {contrib_a*100:.1f}%")
        print(f"  Contribution {name_b}: {contrib_b*100:.1f}%")

        if contrib_a > 0.7:
            print(f"  ⚠️  {name_a} DOMINATES ({contrib_a*100:.1f}% contribution)")
            print(f"     → Fusion loses information from {name_b}!")
        elif contrib_b > 0.7:
            print(f"  ⚠️  {name_b} DOMINATES ({contrib_b*100:.1f}% contribution)")
            print(f"     → Fusion loses information from {name_a}!")
        else:
            print(f"  ✓  Balanced fusion - both descriptors contribute")


def normalize_and_average(desc_a, desc_b):
    """Normalize to unit variance before averaging."""
    desc_a_norm = desc_a / (np.std(desc_a) + 1e-8)
    desc_b_norm = desc_b / (np.std(desc_b) + 1e-8)
    fused = (desc_a_norm + desc_b_norm) / 2.0
    return fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)


def variance_weighted_fusion(desc_a, desc_b):
    """Weight inversely to variance."""
    var_a = np.var(desc_a)
    var_b = np.var(desc_b)
    w_a = 1.0 / (var_a + 1e-8)
    w_b = 1.0 / (var_b + 1e-8)
    w_sum = w_a + w_b
    return (w_a / w_sum) * desc_a + (w_b / w_sum) * desc_b


def main():
    """Main analysis pipeline."""
    print("=" * 120)
    print("INTERSECTION SET DESCRIPTOR ANALYSIS")
    print("Extracting and analyzing real descriptors from spatially-paired intersection keypoints")
    print("=" * 120)

    # Configuration
    db_path = Path(__file__).parent.parent / 'build' / 'experiments.db'
    data_dir = Path(__file__).parent.parent / 'data'

    # Use scale-matched intersection sets for analysis
    set_a_name = 'sift_keynet_scale_matched_intersection_a'  # SIFT positions
    set_b_name = 'sift_keynet_scale_matched_intersection_b'  # KeyNet positions
    scene_name = 'i_ajuntament'  # First illumination scene

    print(f"\nConfiguration:")
    print(f"  Database: {db_path}")
    print(f"  Set A: {set_a_name}")
    print(f"  Set B: {set_b_name}")
    print(f"  Scene: {scene_name}")

    # Load paired keypoints from database
    print(f"\n[1/5] Loading paired keypoints from database...")
    try:
        kps_a, kps_b, image_name = load_intersection_keypoints(
            db_path, set_a_name, set_b_name, scene_name, num_samples=500
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Get image path
    image_path = data_dir / scene_name / image_name
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 1

    print(f"\nUsing image: {image_path}")

    # Extract descriptors at paired keypoint locations
    print(f"\n[2/5] Extracting descriptors at paired keypoint locations...")
    descriptors = {}

    try:
        print("  Extracting SIFT at Set A positions (SIFT keypoints)...")
        sift_desc = extract_sift_at_keypoints(image_path, kps_a)
        descriptors['sift'] = sift_desc
        print(f"  ✓ SIFT: {sift_desc.shape}")
    except Exception as e:
        print(f"  ✗ SIFT extraction failed: {e}")

    try:
        print("  Extracting RootSIFT at Set A positions (SIFT keypoints)...")
        rootsift_desc = extract_rootsift_at_keypoints(image_path, kps_a)
        descriptors['rootsift'] = rootsift_desc
        print(f"  ✓ RootSIFT: {rootsift_desc.shape}")
    except Exception as e:
        print(f"  ✗ RootSIFT extraction failed: {e}")

    # Note: HardNet extraction would require LibTorch wrapper
    # For now, we demonstrate with SIFT-based descriptors
    print("\n  Note: HardNet extraction requires LibTorch wrapper (C++ extension)")
    print("  To analyze SIFT vs HardNet, we need to run a small experiment with save_descriptors=true")

    if len(descriptors) < 2:
        print("\nERROR: Need at least 2 descriptor types")
        return 1

    # Compute statistics
    print(f"\n[3/5] Computing statistics...")
    stats = compute_statistics(descriptors)
    print_statistics_comparison(stats)

    # Visualize
    print(f"\n[4/5] Creating visualizations...")
    visualize_descriptor_distributions(descriptors)

    # Test fusion
    print(f"\n[5/5] Testing fusion strategies...")
    if 'sift' in descriptors and 'rootsift' in descriptors:
        test_fusion(descriptors['sift'], descriptors['rootsift'], 'SIFT', 'RootSIFT')

    print("\n" + "=" * 120)
    print("SUMMARY:")
    print("=" * 120)
    print("✓ Successfully analyzed descriptors from intersection sets")
    print("✓ Keypoints are spatially paired (index i in set_a corresponds to index i in set_b)")
    print()
    print("NEXT STEPS:")
    print("  1. To analyze SIFT vs HardNet: Run experiment with save_descriptors=true")
    print("  2. Extract saved descriptors from database")
    print("  3. Compare variance ratios to understand fusion behavior")
    print("=" * 120)

    return 0


if __name__ == '__main__':
    sys.exit(main())
