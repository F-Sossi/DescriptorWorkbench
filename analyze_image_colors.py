#!/usr/bin/env python3
"""
Analyze images in HPatches dataset to determine which are color vs grayscale.

Usage:
    python analyze_image_colors.py --data-path ../data
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict


def is_grayscale(image_path):
    """
    Check if an image is grayscale or color.

    Returns:
        bool: True if grayscale, False if color
    """
    img = cv2.imread(str(image_path))

    if img is None:
        return None  # Failed to load

    # If image has only 2 dimensions, it's definitely grayscale
    if len(img.shape) == 2:
        return True

    # If image has 3 dimensions, check if all channels are identical
    if len(img.shape) == 3:
        # Check if B, G, R channels are all the same
        b, g, r = cv2.split(img)

        # If all channels are identical, it's grayscale stored as 3-channel
        if np.array_equal(b, g) and np.array_equal(g, r):
            return True
        else:
            return False

    return None


def analyze_dataset(data_path):
    """
    Analyze all images in the dataset.

    Returns:
        dict: Statistics about color vs grayscale images
    """
    data_dir = Path(data_path)

    if not data_dir.exists():
        print(f"Error: Data path does not exist: {data_path}")
        return None

    # Get all scene directories
    scenes = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    # Track statistics
    stats = {
        'total_scenes': 0,
        'total_images': 0,
        'color_images': 0,
        'grayscale_images': 0,
        'failed_images': 0,
        'scenes_by_type': {
            'all_color': [],
            'all_grayscale': [],
            'mixed': []
        },
        'illumination_vs_viewpoint': {
            'illumination': {'color': 0, 'grayscale': 0, 'total': 0},
            'viewpoint': {'color': 0, 'grayscale': 0, 'total': 0}
        }
    }

    scene_details = []

    for scene_dir in scenes:
        scene_name = scene_dir.name
        stats['total_scenes'] += 1

        # Determine scene type (illumination vs viewpoint)
        scene_type = 'illumination' if scene_name.startswith('i_') else 'viewpoint'

        # Get all .ppm images in the scene
        images = sorted(scene_dir.glob('*.ppm'))

        scene_color_count = 0
        scene_gray_count = 0
        scene_failed_count = 0

        for img_path in images:
            stats['total_images'] += 1

            is_gray = is_grayscale(img_path)

            if is_gray is None:
                stats['failed_images'] += 1
                scene_failed_count += 1
            elif is_gray:
                stats['grayscale_images'] += 1
                scene_gray_count += 1
                stats['illumination_vs_viewpoint'][scene_type]['grayscale'] += 1
            else:
                stats['color_images'] += 1
                scene_color_count += 1
                stats['illumination_vs_viewpoint'][scene_type]['color'] += 1

            stats['illumination_vs_viewpoint'][scene_type]['total'] += 1

        # Categorize scene
        total_valid = scene_color_count + scene_gray_count
        if total_valid > 0:
            if scene_gray_count == 0:
                scene_category = 'all_color'
                stats['scenes_by_type']['all_color'].append(scene_name)
            elif scene_color_count == 0:
                scene_category = 'all_grayscale'
                stats['scenes_by_type']['all_grayscale'].append(scene_name)
            else:
                scene_category = 'mixed'
                stats['scenes_by_type']['mixed'].append(scene_name)
        else:
            scene_category = 'unknown'

        scene_details.append({
            'name': scene_name,
            'type': scene_type,
            'category': scene_category,
            'color': scene_color_count,
            'grayscale': scene_gray_count,
            'failed': scene_failed_count,
            'total': len(images)
        })

    return stats, scene_details


def print_report(stats, scene_details, verbose=False):
    """Print analysis report."""

    print("\n" + "="*80)
    print("IMAGE COLOR ANALYSIS REPORT")
    print("="*80)

    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total scenes:      {stats['total_scenes']}")
    print(f"  Total images:      {stats['total_images']}")
    print(f"  Color images:      {stats['color_images']} ({stats['color_images']/stats['total_images']*100:.1f}%)")
    print(f"  Grayscale images:  {stats['grayscale_images']} ({stats['grayscale_images']/stats['total_images']*100:.1f}%)")
    if stats['failed_images'] > 0:
        print(f"  Failed to load:    {stats['failed_images']}")

    print(f"\n🎨 SCENE CATEGORIES")
    print(f"  All color:         {len(stats['scenes_by_type']['all_color'])} scenes")
    print(f"  All grayscale:     {len(stats['scenes_by_type']['all_grayscale'])} scenes")
    print(f"  Mixed:             {len(stats['scenes_by_type']['mixed'])} scenes")

    print(f"\n🔍 ILLUMINATION vs VIEWPOINT")
    for scene_type in ['illumination', 'viewpoint']:
        data = stats['illumination_vs_viewpoint'][scene_type]
        total = data['total']
        if total > 0:
            print(f"  {scene_type.capitalize()}:")
            print(f"    Color:       {data['color']} ({data['color']/total*100:.1f}%)")
            print(f"    Grayscale:   {data['grayscale']} ({data['grayscale']/total*100:.1f}%)")
            print(f"    Total:       {total} images")

    # Print scene lists
    if stats['scenes_by_type']['all_color']:
        print(f"\n✅ ALL COLOR SCENES ({len(stats['scenes_by_type']['all_color'])}):")
        for scene in stats['scenes_by_type']['all_color'][:10]:
            print(f"    {scene}")
        if len(stats['scenes_by_type']['all_color']) > 10:
            print(f"    ... and {len(stats['scenes_by_type']['all_color']) - 10} more")

    if stats['scenes_by_type']['all_grayscale']:
        print(f"\n⬜ ALL GRAYSCALE SCENES ({len(stats['scenes_by_type']['all_grayscale'])}):")
        for scene in stats['scenes_by_type']['all_grayscale'][:10]:
            print(f"    {scene}")
        if len(stats['scenes_by_type']['all_grayscale']) > 10:
            print(f"    ... and {len(stats['scenes_by_type']['all_grayscale']) - 10} more")

    if stats['scenes_by_type']['mixed']:
        print(f"\n🌈 MIXED SCENES ({len(stats['scenes_by_type']['mixed'])}):")
        for scene in stats['scenes_by_type']['mixed']:
            print(f"    {scene}")

    # Verbose output: detailed per-scene breakdown
    if verbose:
        print(f"\n" + "="*80)
        print("DETAILED SCENE BREAKDOWN")
        print("="*80)
        print(f"{'Scene':<25} {'Type':<12} {'Color':<8} {'Gray':<8} {'Total':<8} {'Category':<15}")
        print("-"*80)

        for scene in scene_details:
            print(f"{scene['name']:<25} {scene['type']:<12} "
                  f"{scene['color']:<8} {scene['grayscale']:<8} "
                  f"{scene['total']:<8} {scene['category']:<15}")


def save_to_csv(scene_details, output_path):
    """Save detailed results to CSV."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'type', 'category', 'color', 'grayscale', 'failed', 'total'])
        writer.writeheader()
        writer.writerows(scene_details)

    print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HPatches dataset images to determine color vs grayscale",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-path', type=str, default='../data',
                        help='Path to HPatches dataset (default: ../data)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-scene breakdown')
    parser.add_argument('--output', '-o', type=str,
                        help='Save detailed results to CSV file')

    args = parser.parse_args()

    print(f"Analyzing images in: {args.data_path}")
    print("This may take a moment...")

    stats, scene_details = analyze_dataset(args.data_path)

    if stats is None:
        return 1

    print_report(stats, scene_details, verbose=args.verbose)

    if args.output:
        save_to_csv(scene_details, args.output)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
