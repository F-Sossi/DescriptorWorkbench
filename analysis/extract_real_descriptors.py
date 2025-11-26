#!/usr/bin/env python3
"""
Extract real descriptors from HPatches images and analyze fusion behavior.

This script extracts actual SIFT, RootSIFT, DSPSIFT, and HardNet descriptors
from a test image and analyzes their statistical properties to understand
why SIFT-based descriptors dominate CNN descriptors during fusion.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add build directory to path for importing C++ modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

def extract_sift_descriptors(image_path, num_keypoints=500):
    """
    Extract SIFT descriptors from image.

    Returns:
        keypoints: List of cv2.KeyPoint
        descriptors: numpy array (N x 128)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=num_keypoints)

    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # L2 normalize
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (norms + 1e-8)

    print(f"SIFT: Extracted {len(keypoints)} keypoints")
    return keypoints, descriptors


def extract_rootsift_descriptors(image_path, num_keypoints=500):
    """
    Extract RootSIFT descriptors from image.

    RootSIFT: L1-normalize, sqrt, then L2-normalize
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=num_keypoints)

    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(gray, None)

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

    print(f"RootSIFT: Extracted {len(keypoints)} keypoints")
    return keypoints, descriptors


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
        }

    return stats


def print_statistics_comparison(stats):
    """Print formatted statistics comparison table."""
    print("\n" + "=" * 100)
    print("REAL DESCRIPTOR STATISTICS (from HPatches image)")
    print("=" * 100)

    headers = ['Metric'] + list(stats.keys())

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

    # Print header
    col_width = 20
    print(f"{'Metric':<{col_width}}", end='')
    for name in stats.keys():
        print(f"{name:<{col_width}}", end='')
    print()
    print("-" * (col_width * (len(stats) + 1)))

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

    print("\n" + "=" * 100)
    print("KEY FINDINGS:")
    print("=" * 100)

    # Compare SIFT vs other descriptors
    if 'sift' in stats:
        sift_std = stats['sift']['std']

        for name in stats.keys():
            if name == 'sift':
                continue
            other_std = stats[name]['std']
            ratio = sift_std / other_std

            print(f"\nSIFT vs {name.upper()}:")
            print(f"  Std dev ratio: {ratio:.2f}x")
            print(f"  SIFT std: {sift_std:.6f}")
            print(f"  {name.upper()} std: {other_std:.6f}")

            if ratio > 2:
                print(f"  ⚠️  SIFT will DOMINATE fusion (variance {ratio:.1f}x higher)")
            elif ratio < 0.5:
                print(f"  ⚠️  {name.upper()} will DOMINATE fusion (variance {1/ratio:.1f}x higher)")
            else:
                print(f"  ✓ Balanced variance - fusion should work well")

    print("\n" + "=" * 100)


def test_fusion(desc_a, desc_b, name_a='A', name_b='B'):
    """Test different fusion strategies and measure contribution."""
    print(f"\n" + "=" * 100)
    print(f"FUSION TEST: {name_a} + {name_b}")
    print("=" * 100)

    # Ensure same number of descriptors
    n = min(len(desc_a), len(desc_b))
    desc_a = desc_a[:n]
    desc_b = desc_b[:n]

    fusion_methods = {
        'average': lambda a, b: (a + b) / 2.0,
        'weighted_avg_0.5': lambda a, b: 0.5 * a + 0.5 * b,
        'normalized_avg': lambda a, b: normalize_and_average(a, b),
        'variance_weighted': lambda a, b: variance_weighted_fusion(a, b),
    }

    for method_name, fusion_fn in fusion_methods.items():
        fused = fusion_fn(desc_a, desc_b)

        # Measure correlation with original descriptors
        corr_a = np.mean([np.corrcoef(fused[i], desc_a[i])[0, 1] for i in range(min(100, n))])
        corr_b = np.mean([np.corrcoef(fused[i], desc_b[i])[0, 1] for i in range(min(100, n))])

        # Compute contribution ratio
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
        elif contrib_b > 0.7:
            print(f"  ⚠️  {name_b} DOMINATES ({contrib_b*100:.1f}% contribution)")
        else:
            print(f"  ✓  Balanced fusion")


def normalize_and_average(desc_a, desc_b):
    """Normalize each descriptor to unit variance before averaging."""
    desc_a_norm = desc_a / (np.std(desc_a) + 1e-8)
    desc_b_norm = desc_b / (np.std(desc_b) + 1e-8)
    fused = (desc_a_norm + desc_b_norm) / 2.0
    # Re-normalize to unit L2
    return fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)


def variance_weighted_fusion(desc_a, desc_b):
    """Weight inversely proportional to variance."""
    var_a = np.var(desc_a)
    var_b = np.var(desc_b)
    w_a = 1.0 / (var_a + 1e-8)
    w_b = 1.0 / (var_b + 1e-8)
    w_sum = w_a + w_b
    return (w_a / w_sum) * desc_a + (w_b / w_sum) * desc_b


def main():
    """Main analysis pipeline."""
    print("=" * 100)
    print("REAL DESCRIPTOR EXTRACTION AND FUSION ANALYSIS")
    print("=" * 100)

    # Find a test image from HPatches
    data_dir = Path(__file__).parent.parent / 'data'

    # Try to find first available scene
    test_image = None
    for scene_dir in sorted(data_dir.glob('*')):
        if scene_dir.is_dir():
            img_path = scene_dir / '1.ppm'
            if img_path.exists():
                test_image = img_path
                break

    if test_image is None:
        print("ERROR: Could not find HPatches test image")
        print(f"Looked in: {data_dir}")
        return 1

    print(f"\nUsing test image: {test_image}")
    print(f"Scene: {test_image.parent.name}")

    # Extract descriptors
    print("\n[1/4] Extracting descriptors...")
    descriptors = {}

    try:
        _, sift_desc = extract_sift_descriptors(test_image, num_keypoints=1000)
        descriptors['sift'] = sift_desc
    except Exception as e:
        print(f"Failed to extract SIFT: {e}")

    try:
        _, rootsift_desc = extract_rootsift_descriptors(test_image, num_keypoints=1000)
        descriptors['rootsift'] = rootsift_desc
    except Exception as e:
        print(f"Failed to extract RootSIFT: {e}")

    # Note: DSPSIFT and HardNet would require more complex setup
    # For now, focus on SIFT vs RootSIFT which we can extract easily

    if len(descriptors) < 2:
        print("ERROR: Need at least 2 descriptor types to analyze fusion")
        return 1

    # Compute statistics
    print("\n[2/4] Computing statistics...")
    stats = compute_statistics(descriptors)
    print_statistics_comparison(stats)

    # Test fusion
    print("\n[3/4] Testing fusion strategies...")
    if 'sift' in descriptors and 'rootsift' in descriptors:
        test_fusion(descriptors['sift'], descriptors['rootsift'], 'SIFT', 'RootSIFT')

    print("\n[4/4] Summary")
    print("=" * 100)
    print("NEXT STEPS:")
    print("  1. If one descriptor dominates, try 'normalized_avg' or 'variance_weighted' fusion")
    print("  2. Check if per-dimension variance differs significantly")
    print("  3. Consider concatenation instead of averaging")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
