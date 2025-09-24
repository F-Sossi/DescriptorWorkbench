#!/usr/bin/env python3
"""
KeyNet Keypoint Generation Script

This script generates keypoints using Kornia's KeyNet detector and stores them
in the database for comparison with SIFT-generated keypoints.

The hypothesis is that CNN descriptors (HardNet, SOSNet) perform better with
KeyNet keypoints than SIFT keypoints, since they were trained together.
"""

import sys
import os
import sqlite3
import numpy as np
import cv2
import torch
import kornia as K
import kornia.feature as KF
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import csv

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - CLI fallback when tqdm missing
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class KeyNetKeypointGenerator:
    """Generate keypoints using Kornia's KeyNet detector"""

    def __init__(self, max_keypoints: int = 2000, device: str = "auto"):
        """
        Initialize KeyNet detector

        Args:
            max_keypoints: Maximum number of keypoints to detect
            device: Device to run on ("auto", "cuda", "cpu")
        """
        if device == "auto":
            self.device = K.utils.get_cuda_device_if_available()
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize KeyNet detector via multi-resolution wrapper to control count
        self.keynet = KF.MultiResolutionDetector(
            model=KF.KeyNet(),
            num_features=max_keypoints
        ).eval().to(self.device)
        self.max_keypoints = max_keypoints

        print(f"KeyNet detector initialized with max_keypoints={max_keypoints}")

    def detect_keypoints(self, image_path: str) -> List[cv2.KeyPoint]:
        """
        Detect keypoints in an image using KeyNet

        Args:
            image_path: Path to the image file

        Returns:
            List of OpenCV KeyPoint objects
        """
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to PyTorch tensor [1, 1, H, W]
        image_tensor = K.image_to_tensor(image, keepdim=False).float() / 255.0

        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() > 4:
            # Squeeze superfluous singleton dims produced by Kornia
            while image_tensor.dim() > 4 and image_tensor.size(0) == 1:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

        if image_tensor.dim() != 4:
            raise ValueError(f"Unexpected tensor shape after conversion: {image_tensor.shape}")

        image_tensor = image_tensor.to(self.device)

        # Detect keypoints using KeyNet
        with torch.no_grad():
            lafs, responses = self.keynet(image_tensor)

        # Convert LAFs (Local Affine Frames) to OpenCV KeyPoints
        keypoints = self._lafs_to_keypoints(lafs.cpu(), responses.cpu())

        return keypoints

    def _lafs_to_keypoints(self, lafs: torch.Tensor, responses: torch.Tensor) -> List[cv2.KeyPoint]:
        """
        Convert Kornia LAFs to OpenCV KeyPoints

        Args:
            lafs: Local Affine Frames [1, N, 2, 3]
            responses: Response values [1, N, 1]

        Returns:
            List of OpenCV KeyPoint objects
        """
        keypoints = []

        # Extract from batch dimension
        lafs = lafs[0]  # [N, 2, 3]
        responses = responses[0].squeeze(-1)  # [N]

        for i in range(lafs.shape[0]):
            # Extract position (translation part of affine transform)
            x = float(lafs[i, 0, 2])
            y = float(lafs[i, 1, 2])

            # Extract scale (from affine transform matrix)
            # Scale is the norm of the first column of the 2x2 part
            scale_x = torch.norm(lafs[i, :, 0]).item()
            scale_y = torch.norm(lafs[i, :, 1]).item()
            size = (scale_x + scale_y) / 2.0 * 2.0  # Convert to OpenCV size convention

            # Extract angle (rotation)
            angle = torch.atan2(lafs[i, 1, 0], lafs[i, 0, 0]).item() * 180.0 / np.pi

            # Create KeyPoint
            response = float(responses[i])
            kp = cv2.KeyPoint(x=x, y=y, size=size, angle=angle, response=response,
                            octave=0, class_id=-1)
            keypoints.append(kp)

        return keypoints

class DatabaseManager:
    """Manage database operations for keypoint storage"""

    def __init__(self, db_path: str = "experiments.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self._create_keypoint_set()

    def _create_keypoint_set(self):
        """Create keypoint set entry for KeyNet keypoints"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if keypoint set already exists
            cursor.execute("SELECT id FROM keypoint_sets WHERE name = ?", ("keynet_detector_keypoints",))
            result = cursor.fetchone()

            if result:
                self.keypoint_set_id = result[0]
                print(f"Using existing keypoint set ID: {self.keypoint_set_id}")
            else:
                # Create new keypoint set
                cursor.execute("""
                    INSERT INTO keypoint_sets (
                        name, generator_type, generation_method, max_features,
                        dataset_path, description, boundary_filter_px, overlap_filtering, min_distance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    "keynet_detector_keypoints",
                    "keynet",
                    "learned_detection",
                    2000,
                    "../data/",
                    "KeyNet learned keypoint detector with homography transformation",
                    40,  # Same boundary filtering as SIFT
                    False,  # No overlap filtering for KeyNet
                    0.0
                ))
                self.keypoint_set_id = cursor.lastrowid
                print(f"Created new keypoint set ID: {self.keypoint_set_id}")

    def clear_keypoints(self):
        """Clear existing KeyNet keypoints"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM locked_keypoints WHERE keypoint_set_id = ?",
                         (self.keypoint_set_id,))
            print(f"Cleared existing keypoints for set {self.keypoint_set_id}")

    def store_keypoints(self, scene_name: str, image_name: str, keypoints: List[cv2.KeyPoint]):
        """Store keypoints in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Prepare batch insert data
            keypoint_data = []
            for kp in keypoints:
                keypoint_data.append((
                    self.keypoint_set_id, scene_name, image_name,
                    float(kp.pt[0]), float(kp.pt[1]), float(kp.size), float(kp.angle),
                    float(kp.response), int(kp.octave), int(kp.class_id), True
                ))

            # Batch insert
            cursor.executemany("""
                INSERT OR REPLACE INTO locked_keypoints (
                    keypoint_set_id, scene_name, image_name, x, y, size, angle,
                    response, octave, class_id, valid_bounds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, keypoint_data)

            return len(keypoint_data)

def load_homography(homography_path: str) -> np.ndarray:
    """Load homography matrix from file"""
    try:
        H = np.loadtxt(homography_path)
        if H.shape != (3, 3):
            raise ValueError(f"Invalid homography shape: {H.shape}")
        return H
    except Exception as e:
        print(f"Warning: Could not load homography {homography_path}: {e}")
        return np.eye(3)  # Return identity matrix as fallback

def apply_homography_to_keypoints(keypoints: List[cv2.KeyPoint], H: np.ndarray) -> List[cv2.KeyPoint]:
    """Apply homography transformation to keypoints"""
    if len(keypoints) == 0:
        return []

    # Extract points
    points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    points = points.reshape(-1, 1, 2)

    # Apply homography
    transformed_points = cv2.perspectiveTransform(points, H)
    transformed_points = transformed_points.reshape(-1, 2)

    # Create new keypoints with transformed positions
    transformed_keypoints = []
    for i, kp in enumerate(keypoints):
        new_kp = cv2.KeyPoint(
            x=float(transformed_points[i][0]),
            y=float(transformed_points[i][1]),
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id
        )
        transformed_keypoints.append(new_kp)

    return transformed_keypoints

def filter_boundary_keypoints(keypoints: List[cv2.KeyPoint], image_shape: Tuple[int, int],
                            border: int = 40) -> List[cv2.KeyPoint]:
    """Filter out keypoints too close to image boundaries"""
    height, width = image_shape[:2]
    filtered = []

    for kp in keypoints:
        if (border <= kp.pt[0] <= width - border and
            border <= kp.pt[1] <= height - border):
            filtered.append(kp)

    return filtered

def write_keypoints_to_csv(output_path: str, keypoints: List[cv2.KeyPoint]) -> None:
    """Write keypoints to CSV in the expected format"""
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "size", "angle", "response"])
        for kp in keypoints:
            writer.writerow([kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response])


def run_single_image_mode(args: argparse.Namespace) -> int:
    if not args.input or not args.output:
        print("Error: --input and --output are required for single-image mode")
        return 2

    generator = KeyNetKeypointGenerator(max_keypoints=args.max_keypoints, device=args.device)

    try:
        keypoints = generator.detect_keypoints(args.input)
        write_keypoints_to_csv(args.output, keypoints)
        print(f"Detected {len(keypoints)} keypoints and wrote them to {args.output}")
        return 0
    except Exception as exc:
        print(f"KeyNet detection error: {exc}", file=sys.stderr)
        return 1


def run_dataset_mode(args: argparse.Namespace) -> int:
    generator = KeyNetKeypointGenerator(max_keypoints=args.max_keypoints, device=args.device)
    db = DatabaseManager(args.db_path)

    if args.clear:
        db.clear_keypoints()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return 1

    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    total_keypoints = 0
    total_scenes = 0

    print(f"Processing {len(scene_dirs)} scenes...")

    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_name = scene_dir.name

        ref_image_path = scene_dir / "1.ppm"
        if not ref_image_path.exists():
            print(f"Warning: Reference image not found: {ref_image_path}")
            continue

        try:
            ref_keypoints = generator.detect_keypoints(str(ref_image_path))
            ref_image = cv2.imread(str(ref_image_path), cv2.IMREAD_GRAYSCALE)
            ref_keypoints = filter_boundary_keypoints(ref_keypoints, ref_image.shape)

            stored = db.store_keypoints(scene_name, "1.ppm", ref_keypoints)
            total_keypoints += stored

            for i in range(2, 7):
                target_image_path = scene_dir / f"{i}.ppm"
                homography_path = scene_dir / f"H_1_{i}"

                if not target_image_path.exists():
                    continue

                H = load_homography(str(homography_path))
                transformed_keypoints = apply_homography_to_keypoints(ref_keypoints, H)

                target_image = cv2.imread(str(target_image_path), cv2.IMREAD_GRAYSCALE)
                transformed_keypoints = filter_boundary_keypoints(transformed_keypoints, target_image.shape)

                stored = db.store_keypoints(scene_name, f"{i}.ppm", transformed_keypoints)
                total_keypoints += stored

            total_scenes += 1

        except Exception as exc:
            print(f"Error processing scene {scene_name}: {exc}")
            continue

    print("\n✅ KeyNet keypoint generation complete!")
    print(f"📊 Processed: {total_scenes} scenes")
    print(f"🎯 Generated: {total_keypoints:,} keypoints")
    print("💾 Stored in keypoint set: keynet_detector_keypoints")
    print("\n🔬 Ready to test CNN descriptors with KeyNet keypoints!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="KeyNet keypoint generation")
    parser.add_argument("--data_dir", default="../data", help="HPatches dataset directory (dataset mode)")
    parser.add_argument("--db_path", default="experiments.db", help="Database path (dataset mode)")
    parser.add_argument("--max_keypoints", type=int, default=2000, help="Maximum keypoints per image")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--clear", action="store_true", help="Clear existing keypoints (dataset mode)")
    parser.add_argument("--input", help="Single image input path (single-image mode)")
    parser.add_argument("--output", help="Single image output CSV path (single-image mode)")

    args = parser.parse_args()

    if args.input or args.output:
        sys.exit(run_single_image_mode(args))

    sys.exit(run_dataset_mode(args))

if __name__ == "__main__":
    main()
