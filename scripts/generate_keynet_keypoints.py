#!/usr/bin/env python3
"""
Generate KeyNet keypoints with Kornia and store them in experiments.db.

Features:
- Independent per-image detection (default) or homography-projected set to mirror locked SIFT flow.
- Configurable keypoint set naming/description for storage in keypoint_sets table.
- Optional overwrite flag to clear existing entries before regeneration.
- Re-usable single-image mode for the C++ KeynetDetector bridge.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


# Add project root so we can import repository modules if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


@dataclass
class DBSetConfig:
    name: str
    generator_type: str = "keynet"
    generation_method: str = "independent_detection"
    max_features: int = 2000
    dataset_path: str = "../data/"
    description: str = "KeyNet detector (Kornia)"
    boundary_filter_px: int = 40
    overlap_filtering: bool = False
    min_distance: float = 0.0


class DatabaseManager:
    """Helper for inserting keypoints into experiments.db."""

    def __init__(self, db_path: Path, config: DBSetConfig, overwrite: bool) -> None:
        self.db_path = db_path
        self.config = config
        self.set_id = self._ensure_keypoint_set(overwrite)
        if overwrite:
            self.clear_keypoints()

    def _ensure_keypoint_set(self, overwrite: bool) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM keypoint_sets WHERE name = ?",
                (self.config.name,),
            )
            row = cursor.fetchone()
            if row and not overwrite:
                print(f"Using existing keypoint set '{self.config.name}' (id={row[0]})")
                return int(row[0])

            if row and overwrite:
                set_id = int(row[0])
                cursor.execute(
                    "UPDATE keypoint_sets SET generator_type=?, generation_method=?, max_features=?, "
                    "dataset_path=?, description=?, boundary_filter_px=?, overlap_filtering=?, min_distance=? "
                    "WHERE id=?",
                    (
                        self.config.generator_type,
                        self.config.generation_method,
                        self.config.max_features,
                        self.config.dataset_path,
                        self.config.description,
                        self.config.boundary_filter_px,
                        1 if self.config.overlap_filtering else 0,
                        self.config.min_distance,
                        set_id,
                    ),
                )
                return set_id

            cursor.execute(
                """
                INSERT INTO keypoint_sets (
                    name, generator_type, generation_method, max_features,
                    dataset_path, description, boundary_filter_px, overlap_filtering, min_distance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.config.name,
                    self.config.generator_type,
                    self.config.generation_method,
                    self.config.max_features,
                    self.config.dataset_path,
                    self.config.description,
                    self.config.boundary_filter_px,
                    1 if self.config.overlap_filtering else 0,
                    self.config.min_distance,
                ),
            )
            set_id = cursor.lastrowid
            print(f"Created keypoint set '{self.config.name}' (id={set_id})")
            return int(set_id)

    def clear_keypoints(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM locked_keypoints WHERE keypoint_set_id = ?",
                (self.set_id,),
            )
        print(f"Cleared keypoints for set id {self.set_id}")

    def store_keypoints(
        self, scene_name: str, image_name: str, keypoints: List[cv2.KeyPoint]
    ) -> int:
        if not keypoints:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            payload = [
                (
                    self.set_id,
                    scene_name,
                    image_name,
                    float(kp.pt[0]),
                    float(kp.pt[1]),
                    float(kp.size),
                    float(kp.angle),
                    float(kp.response),
                    int(kp.octave),
                    int(kp.class_id),
                    True,
                )
                for kp in keypoints
            ]
            cursor.executemany(
                """
                INSERT OR REPLACE INTO locked_keypoints (
                    keypoint_set_id, scene_name, image_name, x, y, size, angle,
                    response, octave, class_id, valid_bounds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(keypoints)


class KeyNetKeypointGenerator:
    def __init__(self, max_keypoints: int = 2000, device: str = "auto") -> None:
        if device == "auto":
            torch_device = K.utils.get_cuda_device_if_available()
        else:
            torch_device = torch.device(device)
        self.device = torch_device
        print(f"Using device: {self.device}")

        self.detector = KF.KeyNetDetector(
            num_features=max_keypoints,
            pretrained=True,
        ).to(self.device).eval()
        self.max_keypoints = max_keypoints
        print(f"KeyNet detector initialized with max_keypoints={max_keypoints}")

    def detect(self, image_path: Path) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        tensor = K.image_to_tensor(image, keepdim=False).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            lafs, responses = self.detector(tensor)

        keypoints = self._lafs_to_keypoints(lafs.cpu(), responses.cpu())
        return keypoints, image

    def _lafs_to_keypoints(
        self, lafs: torch.Tensor, responses: torch.Tensor
    ) -> List[cv2.KeyPoint]:
        keypoints: List[cv2.KeyPoint] = []
        lafs = lafs[0]
        responses = responses[0].squeeze(-1)

        for laf, response in zip(lafs, responses):
            x = float(laf[0, 2])
            y = float(laf[1, 2])
            scale_x = torch.norm(laf[:, 0]).item()
            scale_y = torch.norm(laf[:, 1]).item()
            size = (scale_x + scale_y) / 2.0 * 2.0
            angle = torch.atan2(laf[1, 0], laf[0, 0]).item() * 180.0 / np.pi
            kp = cv2.KeyPoint(x, y, size, angle)
            kp.response = float(response)
            kp.octave = 0
            kp.class_id = -1
            keypoints.append(kp)

        return keypoints


# -----------------------------------------------------------------------------
# Processing helpers
# -----------------------------------------------------------------------------

def filter_boundary_keypoints(
    keypoints: List[cv2.KeyPoint],
    image_shape: Tuple[int, int],
    border: int,
) -> List[cv2.KeyPoint]:
    h, w = image_shape
    filtered = [
        kp
        for kp in keypoints
        if border <= kp.pt[0] <= (w - border) and border <= kp.pt[1] <= (h - border)
    ]
    return filtered


def top_n_by_response(keypoints: List[cv2.KeyPoint], limit: int) -> List[cv2.KeyPoint]:
    if limit <= 0 or len(keypoints) <= limit:
        return keypoints
    keypoints.sort(key=lambda kp: kp.response, reverse=True)
    return keypoints[:limit]


def process_scene_independent(
    generator: KeyNetKeypointGenerator,
    scene_dir: Path,
    border: int,
    max_keypoints: int,
    db: DatabaseManager,
) -> Tuple[int, int]:
    stored = 0
    total = 0
    for image_path in sorted(scene_dir.glob("*.ppm")):
        keypoints, image = generator.detect(image_path)
        keypoints = filter_boundary_keypoints(keypoints, image.shape, border)
        keypoints = top_n_by_response(keypoints, max_keypoints)
        count = db.store_keypoints(scene_dir.name, image_path.name, keypoints)
        stored += count
        total += 1
    return stored, total


def load_homography(path: Path) -> np.ndarray:
    try:
        H = np.loadtxt(str(path))
        if H.shape != (3, 3):
            raise ValueError("Invalid homography shape")
        return H
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not load homography {path}: {exc}")


def apply_homography_to_keypoints(
    keypoints: List[cv2.KeyPoint], H: np.ndarray
) -> List[cv2.KeyPoint]:
    if not keypoints:
        return []

    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)

    projected: List[cv2.KeyPoint] = []
    for kp, (x, y) in zip(keypoints, transformed):
        new_kp = cv2.KeyPoint(float(x), float(y), kp.size, kp.angle)
        new_kp.response = kp.response
        new_kp.octave = kp.octave
        new_kp.class_id = kp.class_id
        projected.append(new_kp)
    return projected


def process_scene_projected(
    generator: KeyNetKeypointGenerator,
    scene_dir: Path,
    border: int,
    max_keypoints: int,
    db: DatabaseManager,
) -> Tuple[int, int]:
    reference_path = scene_dir / "1.ppm"
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image missing: {reference_path}")

    ref_keypoints, ref_image = generator.detect(reference_path)
    ref_keypoints = filter_boundary_keypoints(ref_keypoints, ref_image.shape, border)
    ref_keypoints = top_n_by_response(ref_keypoints, max_keypoints)

    stored = db.store_keypoints(scene_dir.name, "1.ppm", ref_keypoints)
    total_images = 1

    for idx in range(2, 7):
        image_path = scene_dir / f"{idx}.ppm"
        if not image_path.exists():
            continue

        H_path = scene_dir / f"H_1_{idx}"
        if not H_path.exists():
            continue

        H = load_homography(H_path)
        transformed = apply_homography_to_keypoints(ref_keypoints, H)

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        transformed = filter_boundary_keypoints(transformed, image.shape, border)
        count = db.store_keypoints(scene_dir.name, image_path.name, transformed)
        stored += count
        total_images += 1

    return stored, total_images


# -----------------------------------------------------------------------------
# CLI operations
# -----------------------------------------------------------------------------

def run_single_image_mode(args: argparse.Namespace) -> int:
    if not args.input or not args.output:
        print("Error: --input and --output required for single-image mode", file=sys.stderr)
        return 2

    generator = KeyNetKeypointGenerator(max_keypoints=args.max_keypoints, device=args.device)
    keypoints, _ = generator.detect(Path(args.input))
    keypoints = top_n_by_response(keypoints, args.max_keypoints)

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "size", "angle", "response"])
        for kp in keypoints:
            writer.writerow([kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response])

    print(f"Detected {len(keypoints)} keypoints -> {args.output}")
    return 0


def run_dataset_mode(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).resolve()
    if not data_root.exists():
        print(f"Error: data directory does not exist: {data_root}", file=sys.stderr)
        return 1

    config = DBSetConfig(
        name=args.set_name,
        generator_type="keynet",
        generation_method="independent_detection" if args.mode == "independent" else "homography_projection",
        max_features=args.max_keypoints,
        dataset_path=str(data_root),
        description=args.description,
        boundary_filter_px=args.border,
        overlap_filtering=False,
        min_distance=0.0,
    )

    db = DatabaseManager(Path(args.db_path), config, overwrite=args.overwrite)
    generator = KeyNetKeypointGenerator(max_keypoints=args.max_keypoints, device=args.device)

    total_keypoints = 0
    total_images = 0
    scenes = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if args.scenes:
        filter_set = set(args.scenes)
        scenes = [d for d in scenes if d.name in filter_set]
    print(f"Processing {len(scenes)} scenes...")

    for scene_dir in tqdm(scenes, desc="Scenes"):
        if args.mode == "independent":
            stored, count = process_scene_independent(generator, scene_dir, args.border, args.max_keypoints, db)
        else:
            stored, count = process_scene_projected(generator, scene_dir, args.border, args.max_keypoints, db)
        total_keypoints += stored
        total_images += count

    print("\n✅ KeyNet keypoint generation complete")
    print(f"Scenes processed: {len(scenes)}")
    print(f"Images processed: {total_images}")
    print(f"Keypoints stored: {total_keypoints}")
    print(f"Database set name: {args.set_name}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KeyNet keypoints with Kornia")
    parser.add_argument("--data_dir", default="../data", help="HPatches dataset directory (dataset mode)")
    parser.add_argument("--db_path", default="experiments.db", help="Path to experiments.db")
    parser.add_argument("--max_keypoints", type=int, default=2000, help="Maximum keypoints per image")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--mode", choices=["independent", "projected"], default="independent", help="Detection strategy")
    parser.add_argument("--border", type=int, default=40, help="Boundary filter in pixels")
    parser.add_argument("--set-name", default="keynet_detector_keypoints", help="Destination keypoint set name")
    parser.add_argument("--description", default="KeyNet detector (Kornia, independent detection)", help="Keypoint set description")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing entries for this set before writing")
    parser.add_argument("--scenes", nargs="*", help="Optional subset of scenes to process")
    parser.add_argument("--input", help="Single image input path (single-image mode)")
    parser.add_argument("--output", help="Single image CSV output path (single-image mode)")
    parser.add_argument("--clear", action="store_true", help="Legacy flag - no longer used (kept for compatibility)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input or args.output:
        return run_single_image_mode(args)
    return run_dataset_mode(args)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
