#!/usr/bin/env python3
"""
Analyze saved descriptors from database to understand fusion behavior.

Loads SIFT, RootSIFT, DSPSIFT, and HardNet descriptors from the database
and analyzes their variance to explain fusion failures.
"""

import sys
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_descriptors_from_db(db_path, experiment_ids):
    """
    Load descriptors from database for given experiment IDs.

    Returns:
        dict: {experiment_id: {'name': str, 'descriptors': list of numpy arrays}}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = {}

    for exp_id in experiment_ids:
        # Get experiment name
        cursor.execute("SELECT descriptor_type FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        if not row:
            print(f"Warning: Experiment {exp_id} not found")
            continue

        desc_name = row[0]

        # Load descriptors (stored as BLOB)
        cursor.execute("""
            SELECT descriptor_vector, descriptor_dimension
            FROM descriptors
            WHERE experiment_id = ?
        """, (exp_id,))

        descriptor_rows = cursor.fetchall()

        # Convert BLOBs to numpy arrays
        descriptors = []
        for (blob, dim) in descriptor_rows:
            if blob:
                # Descriptor is stored as raw bytes (float32)
                desc_array = np.frombuffer(blob, dtype=np.float32)
                if len(desc_array) == dim:
                    descriptors.append(desc_array)
                else:
                    print(f"  Warning: Unexpected descriptor size: {len(desc_array)}, expected {dim}")

        results[exp_id] = {
            'name': desc_name,
            'descriptors': np.array(descriptors) if descriptors else None
        }

        if descriptors:
            print(f"Loaded {len(descriptors)} descriptors from experiment {exp_id} ({desc_name})")
        else:
            print(f"Warning: No descriptors found for experiment {exp_id}")

    conn.close()
    return results


def compute_statistics(descriptors_dict):
    """Compute statistics for each descriptor type."""
    stats = {}

    for name, desc in descriptors_dict.items():
        if desc is None or len(desc) == 0:
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


def print_statistics_table(stats):
    """Print formatted statistics table."""
    print("\n" + "=" * 140)
    print("REAL DESCRIPTOR STATISTICS (from Database)")
    print("=" * 140)

    col_width = 30
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
    ]

    for metric_name, metric_key in metrics:
        print(f"{metric_name:<{col_width}}", end='')
        for desc_name in stats.keys():
            val = stats[desc_name][metric_key]
            if isinstance(val, (int, np.integer)):
                print(f"{val:<{col_width}}", end='')
            else:
                print(f"{val:<{col_width}.6f}", end='')
        print()

    print("\n" + "=" * 140)
    print("🔍 VARIANCE RATIO ANALYSIS - KEY TO UNDERSTANDING FUSION FAILURE:")
    print("=" * 140)

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
                print(f"  🚨 {name_a.upper()} will DOMINATE fusion (variance {ratio:.1f}x higher)")
                print(f"     → Average fusion: {name_a} gets ~{ratio/(1+ratio)*100:.0f}% weight, {name_b} gets ~{1/(1+ratio)*100:.0f}%")
                print(f"     → {name_b} contributions are NULLED OUT!")
            elif ratio < 0.5:
                print(f"  🚨 {name_b.upper()} will DOMINATE fusion (variance {1/ratio:.1f}x higher)")
                print(f"     → Average fusion: {name_b} gets ~{(1/ratio)/(1+(1/ratio))*100:.0f}% weight, {name_a} gets ~{1/(1+(1/ratio))*100:.0f}%")
                print(f"     → {name_a} contributions are NULLED OUT!")
            else:
                print(f"  ✅ Balanced variance - fusion should work well")

    print("\n" + "=" * 140)


def visualize_distributions(descriptors_dict, output_dir='analysis/output'):
    """Create comprehensive visualizations."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Value distributions
    ax = axes[0, 0]
    for name, desc in descriptors_dict.items():
        ax.hist(desc.flatten(), bins=100, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Descriptor Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Value Distributions\n(Shape reveals variance)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(-0.6, 0.6)
    ax.grid(True, alpha=0.3)

    # Plot 2: Std dev comparison (KEY PLOT)
    ax = axes[0, 1]
    names = list(descriptors_dict.keys())
    std_vals = [np.std(descriptors_dict[n]) for n in names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(names, std_vals, color=colors[:len(names)])
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title('⚠️ Standard Deviation Comparison\n(Higher = Dominates Fusion)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    # Plot 3: Per-dimension variance
    ax = axes[0, 2]
    for name, desc in descriptors_dict.items():
        per_dim_var = np.var(desc, axis=0)
        ax.plot(per_dim_var, label=name, alpha=0.7, linewidth=2)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Per-Dimension Variance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Box plots
    ax = axes[1, 0]
    data_for_box = [desc.flatten() for desc in descriptors_dict.values()]
    bp = ax.boxplot(data_for_box, tick_labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Descriptor Value', fontsize=12)
    ax.set_title('Value Range Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    # Plot 5: Variance ratio matrix
    ax = axes[1, 1]
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

    im = ax.imshow(variance_ratios, cmap='RdYlGn_r', vmin=0.2, vmax=5.0, aspect='auto')
    ax.set_xticks(np.arange(n_desc))
    ax.set_yticks(np.arange(n_desc))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(n_desc):
        for j in range(n_desc):
            val = variance_ratios[i, j]
            color = 'white' if val > 2.5 or val < 0.4 else 'black'
            text = ax.text(j, i, f'{val:.2f}',
                          ha="center", va="center", color=color, fontsize=10, fontweight='bold')

    ax.set_title('Variance Ratio Matrix\n(row std / column std)', fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Std Ratio', fontsize=11)

    # Plot 6: Mean absolute value comparison
    ax = axes[1, 2]
    mean_abs_vals = [np.mean(np.abs(desc)) for desc in descriptors_dict.values()]
    bars = ax.bar(names, mean_abs_vals, color=colors[:len(names)])
    ax.set_ylabel('Mean Absolute Value', fontsize=12)
    ax.set_title('Mean Absolute Value Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    plt.suptitle('Descriptor Fusion Analysis: Why SIFT Dominates HardNet', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/descriptor_fusion_variance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization: {output_dir}/descriptor_fusion_variance_analysis.png")


def main():
    """Main analysis pipeline."""
    print("=" * 140)
    print("DESCRIPTOR VARIANCE ANALYSIS - Understanding Fusion Failures")
    print("=" * 140)

    db_path = Path(__file__).parent.parent / 'build' / 'experiments.db'

    # Experiment IDs from the descriptor extraction run
    experiment_ids = {
        217: 'sift_at_sift_positions',
        218: 'rootsift_at_sift_positions',
        219: 'dspsift_at_sift_positions',
        220: 'hardnet_at_keynet_positions'
    }

    print(f"\nLoading descriptors from database: {db_path}")
    print("Experiments:")
    for exp_id, name in experiment_ids.items():
        print(f"  {exp_id}: {name}")

    # Load descriptors
    results = load_descriptors_from_db(db_path, list(experiment_ids.keys()))

    # Create dict by name
    descriptors = {}
    for exp_id, data in results.items():
        if data['descriptors'] is not None:
            name = data['name']
            descriptors[name] = data['descriptors']

    if len(descriptors) < 2:
        print("\nERROR: Need at least 2 descriptor types")
        return 1

    # Compute statistics
    stats = compute_statistics(descriptors)
    print_statistics_table(stats)

    # Visualize
    visualize_distributions(descriptors)

    print("\n" + "=" * 140)
    print("CONCLUSION:")
    print("=" * 140)
    print("If std ratio > 2.0: The higher-variance descriptor DOMINATES fusion")
    print("Solution: Normalize descriptors to unit variance BEFORE fusion")
    print("=" * 140)

    return 0


if __name__ == '__main__':
    sys.exit(main())
