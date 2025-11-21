#!/usr/bin/env python3
"""
Visualize descriptors from the SAME physical point across all descriptor types.

This will reveal if DSPSIFT has unusual characteristics that cause fusion failures.
"""

import sys
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_single_image_descriptors(db_path, experiment_ids, scene_name, image_name, num_descriptors=5):
    """
    Load descriptors from a single image across all experiment types.

    Returns:
        dict: {exp_name: list of descriptor arrays}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = {}

    for exp_id in experiment_ids:
        # Get experiment name
        cursor.execute("SELECT descriptor_type FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        if not row:
            continue
        desc_name = row[0]

        # Load descriptors from specific image
        cursor.execute("""
            SELECT descriptor_vector, keypoint_x, keypoint_y
            FROM descriptors
            WHERE experiment_id = ? AND scene_name = ? AND image_name = ?
            ORDER BY id
            LIMIT ?
        """, (exp_id, scene_name, image_name, num_descriptors))

        rows = cursor.fetchall()

        descriptors = []
        keypoints = []
        for blob, kp_x, kp_y in rows:
            if blob:
                desc = np.frombuffer(blob, dtype=np.float32)
                descriptors.append(desc)
                keypoints.append((kp_x, kp_y))

        results[desc_name] = {
            'descriptors': descriptors,
            'keypoints': keypoints
        }

        print(f"{desc_name}: Loaded {len(descriptors)} descriptors from {scene_name}/{image_name}")

    conn.close()
    return results


def visualize_same_point_descriptors(all_data, output_dir='analysis/output'):
    """
    Visualize multiple descriptor vectors from the same points.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Determine how many points we have
    num_points = min(len(data['descriptors']) for data in all_data.values())

    # Create figure with subplots for each point
    fig = plt.figure(figsize=(20, 4 * num_points))

    descriptor_names = list(all_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for point_idx in range(num_points):
        # Create subplot for this point
        ax = plt.subplot(num_points, 1, point_idx + 1)

        # Get keypoint location from first descriptor
        first_desc_name = descriptor_names[0]
        kp_x, kp_y = all_data[first_desc_name]['keypoints'][point_idx]

        # Plot all descriptor types for this point
        for i, desc_name in enumerate(descriptor_names):
            desc = all_data[desc_name]['descriptors'][point_idx]
            ax.plot(desc, label=desc_name, linewidth=2, alpha=0.8, color=colors[i % len(colors)])

        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Point {point_idx + 1}: Keypoint at ({kp_x:.1f}, {kp_y:.1f})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 127)

        # Add horizontal line at 0
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/same_point_descriptors_overlay.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/same_point_descriptors_overlay.png")
    plt.close()


def visualize_descriptor_grid(all_data, output_dir='analysis/output'):
    """
    Create a grid showing each descriptor type separately for comparison.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    descriptor_names = list(all_data.keys())
    num_points = min(len(data['descriptors']) for data in all_data.values())

    # Create grid: rows = descriptor types, columns = points
    fig, axes = plt.subplots(len(descriptor_names), num_points,
                             figsize=(5 * num_points, 4 * len(descriptor_names)))

    # Handle case of single point
    if num_points == 1:
        axes = axes.reshape(-1, 1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for desc_idx, desc_name in enumerate(descriptor_names):
        for point_idx in range(num_points):
            ax = axes[desc_idx, point_idx]

            desc = all_data[desc_name]['descriptors'][point_idx]
            kp_x, kp_y = all_data[desc_name]['keypoints'][point_idx]

            # Plot descriptor as bar chart
            ax.bar(range(len(desc)), desc, color=colors[desc_idx % len(colors)], alpha=0.7, width=1.0)

            # Formatting
            if point_idx == 0:
                ax.set_ylabel(f'{desc_name}\nValue', fontsize=11, fontweight='bold')

            if desc_idx == 0:
                ax.set_title(f'Point {point_idx + 1}\n({kp_x:.1f}, {kp_y:.1f})', fontsize=12)

            if desc_idx == len(descriptor_names) - 1:
                ax.set_xlabel('Dimension', fontsize=10)

            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(0, color='black', linewidth=0.5)

            # Add statistics text
            stats_text = f'μ={np.mean(desc):.3f}\nσ={np.std(desc):.3f}\n||·||₂={np.linalg.norm(desc):.3f}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=9, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/descriptor_grid_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/descriptor_grid_comparison.png")
    plt.close()


def visualize_descriptor_heatmap(all_data, output_dir='analysis/output'):
    """
    Create heatmap showing all descriptors as a 2D array.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    descriptor_names = list(all_data.keys())
    num_points = min(len(data['descriptors']) for data in all_data.values())

    # Stack all descriptors into matrix
    all_descriptors = []
    row_labels = []

    for point_idx in range(num_points):
        for desc_name in descriptor_names:
            desc = all_data[desc_name]['descriptors'][point_idx]
            all_descriptors.append(desc)
            row_labels.append(f'P{point_idx+1}_{desc_name}')

    descriptor_matrix = np.array(all_descriptors)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, len(row_labels) * 0.4)))

    im = ax.imshow(descriptor_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-0.3, vmax=0.3, interpolation='nearest')

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Descriptor Type (per point)', fontsize=12)
    ax.set_title('Descriptor Heatmap: All Types and Points', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Descriptor Value', fontsize=11)

    # Add grid lines between descriptor types
    for i in range(num_points - 1):
        ax.axhline((i + 1) * len(descriptor_names) - 0.5, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/descriptor_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/descriptor_heatmap.png")
    plt.close()


def analyze_descriptor_statistics(all_data):
    """
    Print detailed statistics for each descriptor at each point.
    """
    print("\n" + "=" * 120)
    print("DESCRIPTOR STATISTICS BY POINT")
    print("=" * 120)

    descriptor_names = list(all_data.keys())
    num_points = min(len(data['descriptors']) for data in all_data.values())

    for point_idx in range(num_points):
        first_desc_name = descriptor_names[0]
        kp_x, kp_y = all_data[first_desc_name]['keypoints'][point_idx]

        print(f"\n{'='*120}")
        print(f"POINT {point_idx + 1}: Keypoint at ({kp_x:.1f}, {kp_y:.1f})")
        print(f"{'='*120}")

        # Header
        print(f"{'Descriptor':<35} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'L2 Norm':<12} {'Range':<12}")
        print("-" * 120)

        for desc_name in descriptor_names:
            desc = all_data[desc_name]['descriptors'][point_idx]

            print(f"{desc_name:<35} {np.mean(desc):<12.6f} {np.std(desc):<12.6f} "
                  f"{np.min(desc):<12.6f} {np.max(desc):<12.6f} "
                  f"{np.linalg.norm(desc):<12.6f} {np.max(desc) - np.min(desc):<12.6f}")

        # Compare pairs
        print(f"\n{'Pairwise Comparisons:'}")
        print("-" * 120)

        for i, name_a in enumerate(descriptor_names):
            for name_b in descriptor_names[i+1:]:
                desc_a = all_data[name_a]['descriptors'][point_idx]
                desc_b = all_data[name_b]['descriptors'][point_idx]

                # Cosine similarity
                cosine_sim = np.dot(desc_a, desc_b) / (np.linalg.norm(desc_a) * np.linalg.norm(desc_b))

                # Euclidean distance
                euclidean_dist = np.linalg.norm(desc_a - desc_b)

                # Correlation
                correlation = np.corrcoef(desc_a, desc_b)[0, 1]

                print(f"{name_a:<25} vs {name_b:<25}: "
                      f"Cosine={cosine_sim:.4f}, L2_dist={euclidean_dist:.4f}, Corr={correlation:.4f}")


def test_fusion_on_same_point(all_data):
    """
    Simulate fusion on the same point to see what happens.
    """
    print("\n" + "=" * 120)
    print("FUSION SIMULATION ON SAME POINTS")
    print("=" * 120)

    descriptor_names = list(all_data.keys())
    num_points = min(len(data['descriptors']) for data in all_data.values())

    # Find SIFT, DSPSIFT, RootSIFT, HardNet
    sift_name = [n for n in descriptor_names if 'sift_at_sift' in n][0]
    dspsift_name = [n for n in descriptor_names if 'dspsift' in n][0]
    rootsift_name = [n for n in descriptor_names if 'rootsift' in n][0]
    hardnet_name = [n for n in descriptor_names if 'hardnet' in n][0]

    for point_idx in range(num_points):
        print(f"\n{'Point ' + str(point_idx + 1):=^120}")

        sift = all_data[sift_name]['descriptors'][point_idx]
        dspsift = all_data[dspsift_name]['descriptors'][point_idx]
        rootsift = all_data[rootsift_name]['descriptors'][point_idx]
        hardnet = all_data[hardnet_name]['descriptors'][point_idx]

        # Test different fusions
        fusions = {
            'SIFT + HardNet (avg)': (sift + hardnet) / 2.0,
            'RootSIFT + HardNet (avg)': (rootsift + hardnet) / 2.0,
            'DSPSIFT + HardNet (avg)': (dspsift + hardnet) / 2.0,
        }

        print(f"\n{'Fusion Type':<35} {'Mean':<12} {'Std':<12} {'L2 Norm':<12} {'Corr w/ SIFT':<15} {'Corr w/ HardNet':<15}")
        print("-" * 120)

        for fusion_name, fused in fusions.items():
            corr_sift = np.corrcoef(fused, sift)[0, 1]
            corr_hardnet = np.corrcoef(fused, hardnet)[0, 1]

            print(f"{fusion_name:<35} {np.mean(fused):<12.6f} {np.std(fused):<12.6f} "
                  f"{np.linalg.norm(fused):<12.6f} {corr_sift:<15.4f} {corr_hardnet:<15.4f}")

        # Show component dominance
        print(f"\n{'Contribution Analysis:'}")
        for fusion_name, fused in fusions.items():
            if 'SIFT' in fusion_name and 'HardNet' in fusion_name:
                # Determine which descriptor the fusion is closer to
                if 'DSPSIFT' in fusion_name:
                    base_desc = dspsift
                    base_name = 'DSPSIFT'
                elif 'RootSIFT' in fusion_name:
                    base_desc = rootsift
                    base_name = 'RootSIFT'
                else:
                    base_desc = sift
                    base_name = 'SIFT'

                dist_to_base = np.linalg.norm(fused - base_desc)
                dist_to_hardnet = np.linalg.norm(fused - hardnet)

                total_dist = dist_to_base + dist_to_hardnet
                contrib_base = dist_to_hardnet / total_dist  # Closer to base = base contributed more
                contrib_hardnet = dist_to_base / total_dist

                print(f"  {fusion_name}: {base_name} contributes {contrib_base*100:.1f}%, HardNet contributes {contrib_hardnet*100:.1f}%")


def main():
    """Main analysis."""
    print("=" * 120)
    print("VISUALIZING DESCRIPTORS FROM THE SAME PHYSICAL POINTS")
    print("=" * 120)

    db_path = Path(__file__).parent.parent / 'build' / 'experiments.db'

    # Experiment IDs (NEW extraction with correct rooting_stage)
    experiment_ids = [221, 222, 223, 224]  # SIFT, RootSIFT, DSPSIFT, HardNet

    scene_name = 'i_ajuntament'
    image_name = '1.ppm'
    num_points = 5  # Analyze first 5 keypoints

    print(f"\nLoading descriptors from:")
    print(f"  Database: {db_path}")
    print(f"  Scene: {scene_name}")
    print(f"  Image: {image_name}")
    print(f"  Points: {num_points}")

    # Load data
    all_data = load_single_image_descriptors(db_path, experiment_ids, scene_name, image_name, num_points)

    if len(all_data) < 2:
        print("\nERROR: Need at least 2 descriptor types")
        return 1

    # Analyze statistics
    analyze_descriptor_statistics(all_data)

    # Test fusion
    test_fusion_on_same_point(all_data)

    # Visualize
    print("\n" + "=" * 120)
    print("CREATING VISUALIZATIONS")
    print("=" * 120)

    visualize_same_point_descriptors(all_data)
    visualize_descriptor_grid(all_data)
    visualize_descriptor_heatmap(all_data)

    print("\n" + "=" * 120)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 120)
    print("Check analysis/output/ for visualizations:")
    print("  - same_point_descriptors_overlay.png: Overlaid line plots")
    print("  - descriptor_grid_comparison.png: Grid comparison")
    print("  - descriptor_heatmap.png: Heatmap view")
    print("=" * 120)

    return 0


if __name__ == '__main__':
    sys.exit(main())
