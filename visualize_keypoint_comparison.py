#!/usr/bin/env python3
"""
Visualize keypoint comparison between SIFT, SURF, and their intersection sets.

Shows 4 columns:
- SIFT keypoints (original)
- SURF keypoints (original)
- Intersection A (SIFT parameters preserved)
- Intersection B (SURF parameters preserved)

Usage:
    python visualize_keypoint_comparison.py --scene i_dome
    python visualize_keypoint_comparison.py --scene v_wall --images 1,2,3
"""

import sqlite3
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def load_keypoints_from_db(db_path, keypoint_set_name, scene_name, image_name):
    """Load keypoints from database for specific scene and image."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get keypoint_set_id
    cursor.execute("""
        SELECT id FROM keypoint_sets WHERE name = ?
    """, (keypoint_set_name,))
    result = cursor.fetchone()
    if result is None:
        conn.close()
        raise ValueError(f"Keypoint set '{keypoint_set_name}' not found in database")
    keypoint_set_id = result[0]

    # Load keypoints
    cursor.execute("""
        SELECT x, y, size, angle, response, octave, class_id
        FROM locked_keypoints
        WHERE keypoint_set_id = ? AND scene_name = ? AND image_name = ?
    """, (keypoint_set_id, scene_name, image_name))

    keypoints = []
    for row in cursor.fetchall():
        x, y, size, angle, response, octave, class_id = row
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=float(size),
                          angle=float(angle), response=float(response),
                          octave=int(octave), class_id=int(class_id))
        keypoints.append(kp)

    conn.close()
    return keypoints


def draw_keypoints_on_image(image, keypoints, color=(0, 255, 0), radius=3, thickness=2):
    """Draw keypoints on image with consistent visualization."""
    img_with_kp = image.copy()

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Draw circle at keypoint location
        cv2.circle(img_with_kp, (x, y), radius, color, thickness)

        # Optionally draw orientation line (scaled by keypoint size)
        if kp.angle != -1:
            angle_rad = np.deg2rad(kp.angle)
            length = int(kp.size / 2)
            end_x = int(x + length * np.cos(angle_rad))
            end_y = int(y + length * np.sin(angle_rad))
            cv2.line(img_with_kp, (x, y), (end_x, end_y), color, 1)

    return img_with_kp


def visualize_keypoint_comparison(data_path, db_path, scene_name, images_to_show=None):
    """
    Visualize keypoint comparison between SIFT, SURF, and intersection.

    Args:
        data_path: Path to HPatches dataset
        db_path: Path to SQLite database
        scene_name: Scene name (e.g., 'i_dome', 'v_wall')
        images_to_show: List of image numbers to show (e.g., [1, 2, 3]) or None for all
    """
    # Define keypoint sets
    keypoint_sets = {
        'SIFT': 'sift_verification_keypoints',
        'SURF': 'surf_keypoints',
        'Intersection A\n(SIFT params)': 'sift_surf_intersection_a',
        'Intersection B\n(SURF params)': 'sift_surf_intersection_b'
    }

    # Define colors (BGR format for OpenCV)
    colors = {
        'SIFT': (0, 255, 0),      # Green
        'SURF': (0, 255, 0),      # Green
        'Intersection A\n(SIFT params)': (0, 255, 0),  # Green
        'Intersection B\n(SURF params)': (0, 255, 0)   # Green
    }

    # Determine which images to show
    if images_to_show is None:
        images_to_show = [1, 2, 3, 4, 5, 6]

    num_images = len(images_to_show)

    # Create figure with 4 columns (SIFT, SURF, Intersection A, Intersection B) and num_images rows
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))

    # Handle case where there's only one image
    if num_images == 1:
        axes = axes.reshape(1, -1)

    # Scene path
    scene_path = Path(data_path) / scene_name
    if not scene_path.exists():
        raise ValueError(f"Scene path does not exist: {scene_path}")

    # Process each image
    for row_idx, img_num in enumerate(images_to_show):
        image_name = f"{img_num}.ppm"
        image_path = scene_path / image_name

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Failed to load image: {image_path}")
            continue

        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process each keypoint set (SIFT, SURF, Intersection A, Intersection B)
        for col_idx, (set_label, set_name) in enumerate(keypoint_sets.items()):
            # Load keypoints
            try:
                keypoints = load_keypoints_from_db(db_path, set_name, scene_name, image_name)
            except Exception as e:
                print(f"Error loading keypoints for {set_name}/{scene_name}/{image_name}: {e}")
                keypoints = []

            # Draw keypoints
            img_with_kp = draw_keypoints_on_image(img_rgb, keypoints,
                                                   color=colors[set_label][::-1],  # RGB instead of BGR
                                                   radius=2, thickness=1)

            # Display
            ax = axes[row_idx, col_idx]
            ax.imshow(img_with_kp)

            # Set title
            if row_idx == 0:
                title = f"{set_label}\n{len(keypoints)} keypoints"
            else:
                title = f"{len(keypoints)} keypoints"
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        # Add row label on the left
        axes[row_idx, 0].text(-0.15, 0.5, f"Image {img_num}",
                               transform=axes[row_idx, 0].transAxes,
                               fontsize=12, fontweight='bold',
                               verticalalignment='center',
                               rotation=90)

    # Overall title
    fig.suptitle(f"Keypoint Comparison: {scene_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)

    return fig


def list_available_scenes(data_path):
    """List all available scenes in the dataset."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Data path does not exist: {data_path}")
        return []

    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description="Visualize keypoint comparison between SIFT, SURF, and both intersection sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all images for i_dome scene
  python visualize_keypoint_comparison.py --scene i_dome

  # Show specific images for v_wall scene
  python visualize_keypoint_comparison.py --scene v_wall --images 1,2,3

  # List available scenes
  python visualize_keypoint_comparison.py --list-scenes

  # Save figure to file
  python visualize_keypoint_comparison.py --scene i_dome --output comparison.png
        """
    )

    parser.add_argument('--scene', type=str,
                        help='Scene name (e.g., i_dome, v_wall)')
    parser.add_argument('--images', type=str,
                        help='Comma-separated list of image numbers to show (e.g., 1,2,3). Default: all (1-6)')
    parser.add_argument('--data-path', type=str, default='../data',
                        help='Path to HPatches dataset (default: ../data)')
    parser.add_argument('--db-path', type=str, default='experiments.db',
                        help='Path to SQLite database (default: experiments.db)')
    parser.add_argument('--list-scenes', action='store_true',
                        help='List available scenes and exit')
    parser.add_argument('--output', type=str,
                        help='Save figure to file instead of displaying')

    args = parser.parse_args()

    # List scenes if requested
    if args.list_scenes:
        scenes = list_available_scenes(args.data_path)
        print(f"\nAvailable scenes ({len(scenes)}):")
        for scene in scenes:
            scene_type = "Illumination" if scene.startswith("i_") else "Viewpoint"
            print(f"  - {scene:20s} ({scene_type})")
        return

    # Require scene name
    if not args.scene:
        parser.print_help()
        print("\nError: --scene is required (or use --list-scenes to see available scenes)")
        sys.exit(1)

    # Parse images to show
    images_to_show = None
    if args.images:
        try:
            images_to_show = [int(x.strip()) for x in args.images.split(',')]
        except ValueError:
            print(f"Error: Invalid image numbers: {args.images}")
            sys.exit(1)

    # Check database exists
    if not Path(args.db_path).exists():
        print(f"Error: Database not found: {args.db_path}")
        print("Make sure you're running from the build/ directory or specify --db-path")
        sys.exit(1)

    # Generate visualization
    print(f"Loading keypoints for scene: {args.scene}")
    if images_to_show:
        print(f"Showing images: {', '.join(map(str, images_to_show))}")
    else:
        print(f"Showing all images (1-6)")

    try:
        fig = visualize_keypoint_comparison(
            args.data_path,
            args.db_path,
            args.scene,
            images_to_show
        )

        if args.output:
            print(f"Saving figure to: {args.output}")
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print("Done!")
        else:
            print("Displaying figure (close window to exit)")
            plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
