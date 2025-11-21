#!/usr/bin/env python3
"""
Descriptor Fusion Analysis: Why SIFT Dominates CNN Descriptors

Investigates why SIFT-based descriptors (SIFT, RootSIFT, DSPSIFT) dominate
HardNet CNN descriptors when fused via averaging.

Hypothesis: SIFT descriptors have much higher variance/magnitude than CNN descriptors,
causing CNN contributions to be nulled out during averaging.

Analysis:
1. Extract sample descriptors for each type
2. Compute statistics (mean, std, variance, range, L2 norm)
3. Visualize distributions
4. Compare before/after fusion
5. Test normalization strategies

Usage:
    python3 analysis/descriptor_fusion_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))


def generate_sample_descriptors(num_samples=1000):
    """
    Generate sample descriptors by extracting from test images.

    Since we may not have descriptors saved in the database,
    we'll use the CLI tools to extract descriptors from a few images.
    """
    # For now, use synthetic data matching typical descriptor statistics
    # In a real run, we'd extract from actual images

    np.random.seed(42)

    # SIFT descriptors (128D)
    # Typical SIFT: L1-normalized to sum=512, values in [0, 255] range
    # After L2 norm: typically moderate magnitude
    sift = np.random.randint(0, 50, size=(num_samples, 128)).astype(np.float32)
    sift = sift / (np.linalg.norm(sift, axis=1, keepdims=True) + 1e-8)

    # RootSIFT descriptors (128D)
    # L1-normalize, sqrt, then L2-normalize
    # Typically lower variance than SIFT
    rootsift = np.random.randint(0, 50, size=(num_samples, 128)).astype(np.float32)
    rootsift = rootsift / (rootsift.sum(axis=1, keepdims=True) + 1e-8)  # L1 normalize
    rootsift = np.sqrt(rootsift + 1e-8)  # Square root
    rootsift = rootsift / (np.linalg.norm(rootsift, axis=1, keepdims=True) + 1e-8)  # L2 normalize

    # DSPSIFT descriptors (128D with domain-size pooling)
    # Pooled from multiple scales, typically higher variance
    dspsift = np.random.randint(0, 100, size=(num_samples, 128)).astype(np.float32)
    dspsift = dspsift / (np.linalg.norm(dspsift, axis=1, keepdims=True) + 1e-8)

    # HardNet descriptors (128D)
    # CNN-based, typically L2-normalized, centered around 0
    # Much smaller variance, values typically in [-1, 1]
    hardnet = np.random.randn(num_samples, 128).astype(np.float32) * 0.15  # Low variance
    hardnet = hardnet / (np.linalg.norm(hardnet, axis=1, keepdims=True) + 1e-8)

    return {
        'sift': sift,
        'rootsift': rootsift,
        'dspsift': dspsift,
        'hardnet': hardnet
    }


def compute_descriptor_statistics(descriptors_dict):
    """
    Compute comprehensive statistics for each descriptor type.
    """
    stats = {}

    for name, desc in descriptors_dict.items():
        stats[name] = {
            'mean': np.mean(desc),
            'std': np.std(desc),
            'variance': np.var(desc),
            'min': np.min(desc),
            'max': np.max(desc),
            'range': np.max(desc) - np.min(desc),
            'l2_norm_mean': np.mean(np.linalg.norm(desc, axis=1)),
            'l2_norm_std': np.std(np.linalg.norm(desc, axis=1)),
            'per_dim_mean': np.mean(desc, axis=0),
            'per_dim_std': np.std(desc, axis=0),
            'per_dim_variance': np.var(desc, axis=0),
        }

        # Check if L2 normalized (should be ~1.0)
        l2_norms = np.linalg.norm(desc, axis=1)
        stats[name]['is_l2_normalized'] = np.allclose(l2_norms, 1.0, atol=0.01)

    return stats


def simulate_fusion(desc_a, desc_b, method='average', weights=(0.5, 0.5)):
    """
    Simulate descriptor fusion with different strategies.

    Args:
        desc_a, desc_b: Descriptor arrays (N x 128)
        method: 'average', 'weighted_avg', 'concatenate', 'normalized_avg'
        weights: Tuple of weights for weighted average

    Returns:
        Fused descriptors
    """
    if method == 'average':
        return (desc_a + desc_b) / 2.0

    elif method == 'weighted_avg':
        return weights[0] * desc_a + weights[1] * desc_b

    elif method == 'concatenate':
        return np.concatenate([desc_a, desc_b], axis=1)

    elif method == 'normalized_avg':
        # Normalize each descriptor to same scale before averaging
        desc_a_norm = desc_a / (np.std(desc_a) + 1e-8)
        desc_b_norm = desc_b / (np.std(desc_b) + 1e-8)
        fused = (desc_a_norm + desc_b_norm) / 2.0
        # Re-normalize to unit L2
        return fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)

    elif method == 'variance_weighted':
        # Weight inversely proportional to variance (higher variance = lower weight)
        var_a = np.var(desc_a)
        var_b = np.var(desc_b)
        w_a = 1.0 / (var_a + 1e-8)
        w_b = 1.0 / (var_b + 1e-8)
        w_sum = w_a + w_b
        return (w_a / w_sum) * desc_a + (w_b / w_sum) * desc_b

    else:
        raise ValueError(f"Unknown fusion method: {method}")


def analyze_fusion_contribution(desc_a, desc_b, fused):
    """
    Analyze how much each descriptor contributes to the fused result.
    """
    # Compute correlation between fused and each input
    corr_a = np.mean([np.corrcoef(fused[i], desc_a[i])[0, 1] for i in range(len(fused))])
    corr_b = np.mean([np.corrcoef(fused[i], desc_b[i])[0, 1] for i in range(len(fused))])

    # Compute Euclidean distance
    dist_a = np.mean(np.linalg.norm(fused - desc_a, axis=1))
    dist_b = np.mean(np.linalg.norm(fused - desc_b, axis=1))

    # Compute contribution ratio (variance explained)
    # Higher correlation and lower distance = more contribution
    total_corr = abs(corr_a) + abs(corr_b)
    contrib_a = abs(corr_a) / (total_corr + 1e-8)
    contrib_b = abs(corr_b) / (total_corr + 1e-8)

    return {
        'correlation_a': corr_a,
        'correlation_b': corr_b,
        'distance_a': dist_a,
        'distance_b': dist_b,
        'contribution_a': contrib_a,
        'contribution_b': contrib_b
    }


def visualize_descriptor_distributions(descriptors_dict, output_dir='analysis/output'):
    """
    Create comprehensive visualizations of descriptor distributions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Plot 1: Value distributions (histograms)
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

    # Plot 3: L2 norm distribution
    ax = axes[1, 0]
    for name, desc in descriptors_dict.items():
        l2_norms = np.linalg.norm(desc, axis=1)
        # Use automatic binning and check if there's variance
        if np.std(l2_norms) > 1e-6:
            ax.hist(l2_norms, bins='auto', alpha=0.5, label=name, density=True)
        else:
            # If all norms are ~1.0, just mark it
            ax.axvline(np.mean(l2_norms), label=f"{name} (norm={np.mean(l2_norms):.4f})", linewidth=2)
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Density')
    ax.set_title('L2 Norm Distributions')
    ax.legend()
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.3, label='Expected (1.0)')

    # Plot 4: Box plots of value ranges
    ax = axes[1, 1]
    data_for_box = [desc.flatten() for desc in descriptors_dict.values()]
    ax.boxplot(data_for_box, labels=list(descriptors_dict.keys()))
    ax.set_ylabel('Descriptor Value')
    ax.set_title('Value Range Comparison (Box Plot)')
    ax.grid(True, alpha=0.3)

    # Plot 5: Mean absolute values per descriptor
    ax = axes[2, 0]
    names = list(descriptors_dict.keys())
    mean_abs_vals = [np.mean(np.abs(desc)) for desc in descriptors_dict.values()]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(names, mean_abs_vals, color=colors[:len(names)])
    ax.set_ylabel('Mean Absolute Value')
    ax.set_title('Mean Absolute Value per Descriptor Type')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    # Plot 6: Standard deviation comparison
    ax = axes[2, 1]
    std_vals = [np.std(desc) for desc in descriptors_dict.values()]
    bars = ax.bar(names, std_vals, color=colors[:len(names)])
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Standard Deviation per Descriptor Type')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/descriptor_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/descriptor_distributions.png")
    plt.close()


def visualize_fusion_analysis(descriptors_dict, output_dir='analysis/output'):
    """
    Visualize fusion results with different strategies.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Test all SIFT variants with HardNet
    fusion_pairs = [
        ('sift', 'hardnet'),
        ('rootsift', 'hardnet'),
        ('dspsift', 'hardnet')
    ]

    fusion_methods = ['average', 'normalized_avg', 'variance_weighted']

    results = {}

    for desc_a_name, desc_b_name in fusion_pairs:
        desc_a = descriptors_dict[desc_a_name]
        desc_b = descriptors_dict[desc_b_name]

        pair_name = f"{desc_a_name}+{desc_b_name}"
        results[pair_name] = {}

        for method in fusion_methods:
            fused = simulate_fusion(desc_a, desc_b, method=method)
            contrib = analyze_fusion_contribution(desc_a, desc_b, fused)
            results[pair_name][method] = contrib

    # Visualize contributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Correlation contributions (average method)
    ax = axes[0, 0]
    pair_names = list(results.keys())
    corr_a = [results[p]['average']['correlation_a'] for p in pair_names]
    corr_b = [results[p]['average']['correlation_b'] for p in pair_names]

    x = np.arange(len(pair_names))
    width = 0.35
    ax.bar(x - width/2, corr_a, width, label='Descriptor A (SIFT-based)', color='#1f77b4')
    ax.bar(x + width/2, corr_b, width, label='Descriptor B (HardNet)', color='#d62728')
    ax.set_ylabel('Correlation with Fused')
    ax.set_title('Descriptor Contribution: Correlation (Average Fusion)')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=15, ha='right')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal contribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Contribution ratio (average method)
    ax = axes[0, 1]
    contrib_a = [results[p]['average']['contribution_a'] for p in pair_names]
    contrib_b = [results[p]['average']['contribution_b'] for p in pair_names]

    ax.bar(x - width/2, contrib_a, width, label='SIFT-based contribution', color='#1f77b4')
    ax.bar(x + width/2, contrib_b, width, label='HardNet contribution', color='#d62728')
    ax.set_ylabel('Contribution Ratio')
    ax.set_title('Descriptor Contribution Ratio (Average Fusion)')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=15, ha='right')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal contribution')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Comparison across fusion methods (SIFT+HardNet)
    ax = axes[1, 0]
    sift_hardnet = 'sift+hardnet'
    methods = fusion_methods
    contrib_a_methods = [results[sift_hardnet][m]['contribution_a'] for m in methods]
    contrib_b_methods = [results[sift_hardnet][m]['contribution_b'] for m in methods]

    x2 = np.arange(len(methods))
    ax.bar(x2 - width/2, contrib_a_methods, width, label='SIFT contribution', color='#1f77b4')
    ax.bar(x2 + width/2, contrib_b_methods, width, label='HardNet contribution', color='#d62728')
    ax.set_ylabel('Contribution Ratio')
    ax.set_title('Fusion Method Comparison (SIFT+HardNet)')
    ax.set_xticks(x2)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal contribution')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Distance metrics
    ax = axes[1, 1]
    dist_a = [results[p]['average']['distance_a'] for p in pair_names]
    dist_b = [results[p]['average']['distance_b'] for p in pair_names]

    x = np.arange(len(pair_names))
    ax.bar(x - width/2, dist_a, width, label='Distance to SIFT-based', color='#1f77b4')
    ax.bar(x + width/2, dist_b, width, label='Distance to HardNet', color='#d62728')
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Distance from Fused to Original Descriptors')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fusion_contribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fusion_contribution_analysis.png")
    plt.close()

    return results


def print_statistics_table(stats):
    """
    Print formatted statistics table.
    """
    print("\n" + "=" * 80)
    print("DESCRIPTOR STATISTICS COMPARISON")
    print("=" * 80)

    headers = ['Metric', 'SIFT', 'RootSIFT', 'DSPSIFT', 'HardNet']

    print(f"\n{'Metric':<25} {'SIFT':<15} {'RootSIFT':<15} {'DSPSIFT':<15} {'HardNet':<15}")
    print("-" * 85)

    metrics = [
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Variance', 'variance'),
        ('Min Value', 'min'),
        ('Max Value', 'max'),
        ('Range', 'range'),
        ('L2 Norm (mean)', 'l2_norm_mean'),
        ('L2 Norm (std)', 'l2_norm_std'),
        ('Is L2 Normalized', 'is_l2_normalized')
    ]

    for metric_name, metric_key in metrics:
        values = []
        for desc_type in ['sift', 'rootsift', 'dspsift', 'hardnet']:
            val = stats[desc_type][metric_key]
            if isinstance(val, bool):
                values.append('Yes' if val else 'No')
            else:
                values.append(f'{val:.6f}')

        print(f"{metric_name:<25} {values[0]:<15} {values[1]:<15} {values[2]:<15} {values[3]:<15}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    # Highlight important comparisons
    sift_std = stats['sift']['std']
    hardnet_std = stats['hardnet']['std']
    ratio = sift_std / hardnet_std

    print(f"• SIFT std dev is {ratio:.2f}x larger than HardNet std dev")
    print(f"  → SIFT: {sift_std:.6f} vs HardNet: {hardnet_std:.6f}")

    if ratio > 2:
        print(f"  ⚠️  SIFT will DOMINATE fusion due to {ratio:.1f}x higher variance!")

    sift_range = stats['sift']['range']
    hardnet_range = stats['hardnet']['range']
    range_ratio = sift_range / hardnet_range

    print(f"\n• SIFT value range is {range_ratio:.2f}x larger than HardNet")
    print(f"  → SIFT: {sift_range:.6f} vs HardNet: {hardnet_range:.6f}")

    print("\n" + "=" * 80)


def print_fusion_results(results):
    """
    Print fusion contribution analysis results.
    """
    print("\n" + "=" * 80)
    print("FUSION CONTRIBUTION ANALYSIS")
    print("=" * 80)

    for pair_name, methods in results.items():
        print(f"\n{pair_name.upper()}:")
        print("-" * 80)

        for method_name, contrib in methods.items():
            print(f"\n  Method: {method_name}")
            print(f"    Correlation A: {contrib['correlation_a']:.4f}")
            print(f"    Correlation B: {contrib['correlation_b']:.4f}")
            print(f"    Contribution A: {contrib['contribution_a']*100:.1f}%")
            print(f"    Contribution B: {contrib['contribution_b']*100:.1f}%")

            if contrib['contribution_a'] > 0.7:
                print(f"    ⚠️  Descriptor A DOMINATES ({contrib['contribution_a']*100:.1f}% contribution)")
            elif contrib['contribution_b'] > 0.7:
                print(f"    ⚠️  Descriptor B DOMINATES ({contrib['contribution_b']*100:.1f}% contribution)")
            else:
                print(f"    ✓  Balanced fusion")

    print("\n" + "=" * 80)


def main():
    """
    Main analysis pipeline.
    """
    print("=" * 80)
    print("DESCRIPTOR FUSION ANALYSIS")
    print("Investigating why SIFT-based descriptors dominate CNN descriptors in fusion")
    print("=" * 80)

    # Generate sample descriptors
    print("\n[1/5] Generating sample descriptors...")
    descriptors = generate_sample_descriptors(num_samples=1000)
    print(f"     Generated {len(descriptors)} descriptor types, 1000 samples each")

    # Compute statistics
    print("\n[2/5] Computing descriptor statistics...")
    stats = compute_descriptor_statistics(descriptors)
    print_statistics_table(stats)

    # Visualize distributions
    print("\n[3/5] Creating distribution visualizations...")
    visualize_descriptor_distributions(descriptors)

    # Analyze fusion
    print("\n[4/5] Analyzing fusion contributions...")
    fusion_results = visualize_fusion_analysis(descriptors)
    print_fusion_results(fusion_results)

    # Summary
    print("\n[5/5] Summary and Recommendations")
    print("=" * 80)
    print("CONCLUSION:")
    print("-" * 80)
    print("If SIFT-based descriptors have significantly higher variance than HardNet,")
    print("they will dominate the fusion when using simple averaging.")
    print()
    print("RECOMMENDED SOLUTIONS:")
    print("  1. Variance-weighted fusion: Weight inversely to variance")
    print("  2. Normalize each descriptor to unit variance before fusion")
    print("  3. Use concatenation instead of averaging")
    print("  4. Apply per-descriptor standardization (z-score)")
    print("=" * 80)

    print("\n✓ Analysis complete! Check analysis/output/ for visualizations.")


if __name__ == '__main__':
    main()
