#!/usr/bin/env python3
"""
Evaluate descriptors on stored keypoints from experiments.db using a YAML config.

This script mirrors the C++ experiment pipeline:
- Reads dataset/keypoint/descriptors/matching settings from a YAML file
- Loads keypoints from the specified keypoint set in experiments.db
- Computes descriptors (currently supports Kornia HardNet and OpenCV SIFT)
- Applies matching (ratio test or brute force cross-check)
- Computes IR-style metrics (micro/macro mAP, P@K) with the same single-GT policy

Outputs per-scene statistics and an optional CSV summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
import kornia as K
import kornia.feature as KF

# -----------------------------------------------------------------------------
# Dataclasses for reporting
# -----------------------------------------------------------------------------

@dataclass
class PairMetrics:
    matches: int = 0
    inliers: int = 0
    precision: float = 0.0
    mean_error: float = math.inf


@dataclass
class SceneMetrics:
    name: str
    matches: int = 0
    inliers: int = 0
    precision: float = 0.0
    map_score: float = 0.0
    queries_with_gt: int = 0
    excluded_queries: int = 0
    total_queries: int = 0


@dataclass
class AggregateMetrics:
    scenes: List[SceneMetrics] = field(default_factory=list)
    total_matches: int = 0
    total_inliers: int = 0
    overall_precision: float = 0.0
    micro_map: float = 0.0
    macro_map: float = 0.0
    micro_map_including_zeros: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    queries_with_gt: int = 0
    excluded_queries: int = 0
    total_queries: int = 0


# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------

def get_keypoint_set_id(db_path: Path, set_name: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM keypoint_sets WHERE name = ?", (set_name,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Keypoint set '{set_name}' not found in {db_path}")
        return int(row[0])


def load_keypoints_from_db(
    db_path: Path,
    set_id: int,
    scene: str,
    image: str,
) -> List[cv2.KeyPoint]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT x, y, size, angle, response, octave, class_id
            FROM locked_keypoints
            WHERE keypoint_set_id = ? AND scene_name = ? AND image_name = ?
            ORDER BY id
            """,
            (set_id, scene, image),
        )
        rows = cursor.fetchall()

    keypoints: List[cv2.KeyPoint] = []
    for x, y, size, angle, response, octave, class_id in rows:
        kp = cv2.KeyPoint(float(x), float(y), float(size), float(angle))
        kp.response = float(response)
        kp.octave = int(octave)
        kp.class_id = int(class_id)
        keypoints.append(kp)
    return keypoints


# -----------------------------------------------------------------------------
# Descriptor computation helpers
# -----------------------------------------------------------------------------

def keypoints_to_lafs(
    keypoints: List[cv2.KeyPoint],
    device: torch.device,
) -> torch.Tensor:
    if not keypoints:
        return torch.empty(1, 0, 2, 3, device=device)

    centers = torch.tensor(
        [[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=torch.float32, device=device
    ).unsqueeze(0)

    # scale is derived from stored keypoint size (~diameter). Convert to scale radius.
    scales = torch.tensor(
        [max(kp.size, 1e-3) / 2.0 for kp in keypoints], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)

    orientations = torch.tensor(
        [kp.angle for kp in keypoints], dtype=torch.float32, device=device
    ).view(1, -1, 1)

    lafs = KF.laf_from_center_scale_ori(centers, scales, orientations)
    return lafs


def hardnet_descriptors(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    device: torch.device,
    descriptor_model: KF.LAFDescriptor,
) -> np.ndarray:
    if not keypoints:
        return np.empty((0, 128), dtype=np.float32)

    tensor = K.image_to_tensor(image, keepdim=False).float().to(device) / 255.0
    lafs = keypoints_to_lafs(keypoints, device)

    with torch.no_grad():
        descriptors = descriptor_model(tensor, lafs)
    return descriptors.cpu().numpy()[0]


def sift_descriptors(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
) -> np.ndarray:
    if not keypoints:
        return np.empty((0, 128), dtype=np.float32)

    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(image, keypoints)
    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)
    return descriptors.astype(np.float32)


# -----------------------------------------------------------------------------
# Matching helpers
# -----------------------------------------------------------------------------

def ratio_test_matches(desc1: np.ndarray, desc2: np.ndarray, threshold: float) -> np.ndarray:
    if desc1.size == 0 or desc2.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    matcher = KF.DescriptorMatcher("snn", threshold)
    with torch.no_grad():
        _, match_idx = matcher(
            torch.from_numpy(desc1).float(),
            torch.from_numpy(desc2).float(),
        )
    return match_idx.numpy()


def brute_force_cross_check(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    if desc1.size == 0 or desc2.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    t_desc1 = torch.from_numpy(desc1).float()
    t_desc2 = torch.from_numpy(desc2).float()
    dists = torch.cdist(t_desc1, t_desc2, p=2)
    nn12 = torch.argmin(dists, dim=1)
    nn21 = torch.argmin(dists, dim=0)

    matches: List[Tuple[int, int]] = []
    for i, j in enumerate(nn12.tolist()):
        if nn21[j].item() == i:
            matches.append((i, j))
    return np.array(matches, dtype=np.int32)


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------

def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), H)
    return projected.reshape(-1, 2)


def find_relevant_index(
    query_point: np.ndarray,
    centers_dst: np.ndarray,
    homography: np.ndarray,
    pixel_threshold: float,
) -> Optional[int]:
    if centers_dst.size == 0:
        return None

    projected = apply_homography(query_point.reshape(-1, 2), homography)[0]
    dists = np.linalg.norm(centers_dst - projected, axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] <= pixel_threshold:
        return idx
    return None


def evaluate_matches(
    centers_ref: np.ndarray,
    centers_dst: np.ndarray,
    matches: np.ndarray,
    homography: np.ndarray,
    pixel_threshold: float,
) -> PairMetrics:
    if matches.size == 0:
        return PairMetrics(0, 0, 0.0, math.inf)

    matched_ref = centers_ref[matches[:, 0]]
    matched_dst = centers_dst[matches[:, 1]]
    projected = apply_homography(matched_ref, homography)
    errors = np.linalg.norm(projected - matched_dst, axis=1)

    inlier_mask = errors <= pixel_threshold
    inliers = int(inlier_mask.sum())
    precision = inliers / len(errors)
    mean_error = float(errors[inlier_mask].mean()) if inliers else float(errors.mean())

    return PairMetrics(len(errors), inliers, precision, mean_error)


def accumulate_map_metrics(
    centers_ref: np.ndarray,
    desc_ref: np.ndarray,
    centers_dst: np.ndarray,
    desc_dst: np.ndarray,
    homography: np.ndarray,
    pixel_threshold: float,
    hits_at_k: Dict[int, int],
    ap_values_global: List[float],
    ap_values_scene: List[float],
    ks: Iterable[int],
) -> Tuple[int, int]:
    queries_with_gt = 0
    excluded = 0

    if centers_dst.size == 0:
        return queries_with_gt, len(centers_ref)

    torch_desc2 = torch.from_numpy(desc_dst).float()

    for qi, query_point in enumerate(centers_ref):
        relevant_idx = find_relevant_index(query_point, centers_dst, homography, pixel_threshold)
        if relevant_idx is None:
            excluded += 1
            continue

        queries_with_gt += 1

        query_desc = torch.from_numpy(desc_ref[qi]).float().unsqueeze(0)
        dists = torch.cdist(query_desc, torch_desc2, p=2)[0]
        order = torch.argsort(dists)
        # Stable rank: find first occurrence of relevant index
        rank_pos = int((order == relevant_idx).nonzero(as_tuple=False)[0].item()) + 1

        ap = 1.0 / rank_pos
        ap_values_global.append(ap)
        ap_values_scene.append(ap)

        for k in ks:
            if rank_pos <= k:
                hits_at_k[k] += 1

    return queries_with_gt, excluded


# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    raw_dataset_path = Path(config["dataset"]["path"])
    if raw_dataset_path.is_absolute():
        dataset_path = raw_dataset_path.resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent
        candidate_paths = [
            (Path.cwd() / raw_dataset_path).resolve(),
            (repo_root / raw_dataset_path).resolve(),
            (repo_root / "build" / raw_dataset_path).resolve(),
            (config_path.parent / raw_dataset_path).resolve(),
        ]
        dataset_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                dataset_path = candidate
                break
        if dataset_path is None:
            raise FileNotFoundError(
                f"Unable to resolve dataset path '{raw_dataset_path}' relative to current working directory, repo root, or build directory"
            )
    scenes_cfg = config["dataset"].get("scenes") or []

    keypoint_set = config["keypoints"].get("keypoint_set_name")
    if not keypoint_set:
        raise ValueError("Keypoint set name must be provided in config")

    descriptor_cfg = config["descriptors"][0]
    descriptor_type = descriptor_cfg.get("type")

    matching_cfg = config["evaluation"]["matching"]
    matching_method = matching_cfg.get("method", "ratio_test")
    ratio_threshold = matching_cfg.get("ratio_threshold", 0.8)

    pixel_threshold = config["evaluation"]["validation"].get("threshold", 0.05)
    # Convert HPatches normalized threshold (0.05 ~ 3px) to pixels by multiplying width? For simplicity use 3.0 px
    # The original pipeline uses 3px tolerance; override here if explicit threshold_px provided.
    pixel_threshold_px = args.pixel_threshold or 3.0

    db_path = Path(args.db_path)
    set_id = get_keypoint_set_id(db_path, keypoint_set)

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    descriptor_name = descriptor_type.lower() if descriptor_type else ""
    if descriptor_name not in {"libtorch_hardnet", "sift"}:
        raise ValueError(f"Descriptor '{descriptor_type}' not supported for python evaluation")

    hardnet_module = None
    if descriptor_name == "libtorch_hardnet":
        hardnet_module = KF.LAFDescriptor(KF.HardNet(pretrained=True)).to(device).eval()

    ks = (1, 5, 10)
    global_hits_at_k: Dict[int, int] = {k: 0 for k in ks}
    global_ap_values: List[float] = []
    global_matches = 0
    global_inliers = 0
    global_queries_with_gt = 0
    global_excluded = 0
    global_total_queries = 0

    scenes = sorted(d for d in dataset_path.iterdir() if d.is_dir())
    if scenes_cfg:
        filter_set = set(scenes_cfg)
        scenes = [d for d in scenes if d.name in filter_set]

    print(f"Using keypoint set: {keypoint_set} (id={set_id})")
    print(f"Descriptor: {descriptor_name}")
    print(f"Matching: {matching_method} (ratio={ratio_threshold if matching_method == 'ratio_test' else 'N/A'})")
    print(f"Pixel tolerance: {pixel_threshold_px}px")
    print(f"Scenes: {len(scenes)}")

    scene_results: List[SceneMetrics] = []

    for scene_dir in scenes:
        scene_name = scene_dir.name
        print(f"\nScene: {scene_name}")

        scene_matches = 0
        scene_inliers = 0
        scene_ap_values: List[float] = []
        scene_queries_with_gt = 0
        scene_excluded = 0
        scene_total_queries = 0

        ref_keypoints: List[cv2.KeyPoint] = []
        ref_descriptors: Optional[np.ndarray] = None
        ref_centers: Optional[np.ndarray] = None

        for idx in range(1, 7):
            image_name = f"{idx}.ppm"
            image_path = scene_dir / image_name
            if not image_path.exists():
                continue

            keypoints = load_keypoints_from_db(db_path, set_id, scene_name, image_name)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")

            if descriptor_name == "libtorch_hardnet":
                descriptors = hardnet_descriptors(image, keypoints, device, hardnet_module)
            else:  # SIFT
                descriptors = sift_descriptors(image, keypoints)

            # Cache per-image data for matches against reference image (1.ppm)
            if idx == 1:
                ref_keypoints = keypoints
                ref_descriptors = descriptors
                ref_centers = np.array([[kp.pt[0], kp.pt[1]] for kp in ref_keypoints], dtype=np.float32)
                continue

            # Reference must exist
            if not ref_keypoints or ref_descriptors is None or ref_centers is None:
                continue

            centers_dst = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

            if matching_method == "ratio_test":
                matches = ratio_test_matches(ref_descriptors, descriptors, ratio_threshold)
            else:
                matches = brute_force_cross_check(ref_descriptors, descriptors)

            H_path = scene_dir / f"H_1_{idx}"
            if not H_path.exists():
                continue
            H = np.loadtxt(str(H_path))

            pair_stats = evaluate_matches(ref_centers, centers_dst, matches, H, pixel_threshold_px)
            scene_matches += pair_stats.matches
            scene_inliers += pair_stats.inliers

            queries_with_gt, excluded = accumulate_map_metrics(
                ref_centers,
                ref_descriptors,
                centers_dst,
                descriptors,
                H,
                pixel_threshold_px,
                global_hits_at_k,
                global_ap_values,
                scene_ap_values,
                ks,
            )

            scene_queries_with_gt += queries_with_gt
            scene_excluded += excluded
            scene_total_queries += len(ref_keypoints)
            global_queries_with_gt += queries_with_gt
            global_excluded += excluded
            global_total_queries += len(ref_keypoints)

        precision = scene_inliers / scene_matches if scene_matches else 0.0
        scene_map = sum(scene_ap_values) / len(scene_ap_values) if scene_ap_values else 0.0
        print(
            "  Scene totals: matches={}, inliers={}, precision={:.3f}, micro mAP={:.4f}".format(
                scene_matches,
                scene_inliers,
                precision,
                scene_map,
            )
        )

        scene_results.append(
            SceneMetrics(
                name=scene_name,
                matches=scene_matches,
                inliers=scene_inliers,
                precision=precision,
                map_score=scene_map,
                queries_with_gt=scene_queries_with_gt,
                excluded_queries=scene_excluded,
                total_queries=scene_total_queries,
            )
        )

        global_matches += scene_matches
        global_inliers += scene_inliers

    global_precision = global_inliers / global_matches if global_matches else 0.0
    micro_map = sum(global_ap_values) / len(global_ap_values) if global_ap_values else 0.0
    scene_maps = [s.map_score for s in scene_results if s.map_score > 0]
    macro_map = sum(scene_maps) / len(scene_maps) if scene_maps else 0.0

    total_queries_all = global_queries_with_gt + global_excluded
    micro_map_incl_zeros = (
        sum(global_ap_values) / total_queries_all if total_queries_all else 0.0
    )
    precision_at_k = {
        k: (global_hits_at_k[k] / len(global_ap_values) if global_ap_values else 0.0)
        for k in ks
    }

    print("\n================ Summary ================")
    print(f"Scenes evaluated: {len(scene_results)}")
    print(f"Total matches: {global_matches}")
    print(f"Total inliers: {global_inliers}")
    print(f"Overall precision (homography inliers / matches): {global_precision:.3f}")
    print(f"Queries with GT matches: {global_queries_with_gt}")
    print(f"Queries without GT (excluded): {global_excluded}")
    print(f"Micro mAP (queries with GT): {micro_map:.4f}")
    print(f"Macro mAP (per-scene): {macro_map:.4f}")
    print(f"Micro mAP incl. zeros: {micro_map_incl_zeros:.4f}")
    print(
        "Precision@K: " + ", ".join(f"P@{k}={precision_at_k[k]:.4f}" for k in ks)
    )

    metrics = AggregateMetrics(
        scenes=scene_results,
        total_matches=global_matches,
        total_inliers=global_inliers,
        overall_precision=global_precision,
        micro_map=micro_map,
        macro_map=macro_map,
        micro_map_including_zeros=micro_map_incl_zeros,
        precision_at_k=precision_at_k,
        queries_with_gt=global_queries_with_gt,
        excluded_queries=global_excluded,
        total_queries=total_queries_all,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(
                "scene,matches,inliers,precision,map,queries_with_gt,excluded,total_queries\n"
            )
            for scene in scene_results:
                fh.write(
                    f"{scene.name},{scene.matches},{scene.inliers},{scene.precision:.4f},"
                    f"{scene.map_score:.4f},{scene.queries_with_gt},{scene.excluded_queries},{scene.total_queries}\n"
                )
            fh.write(
                "TOTAL,{},{},{:.4f},{:.4f},{},{},{}\n".format(
                    global_matches,
                    global_inliers,
                    global_precision,
                    micro_map,
                    global_queries_with_gt,
                    global_excluded,
                    global_total_queries,
                )
            )
        print(f"Saved summary to {output_path}")

    experiment_id = None
    if args.record:
        processing_ms = (time.perf_counter() - start_time) * 1000.0
        experiment_cfg = config.get("experiment", {})
        experiment_id = record_results(
            db_path,
            experiment_cfg,
            dataset_path,
            descriptor_cfg,
            descriptor_name,
            matching_method,
            ratio_threshold,
            keypoint_set,
            metrics,
            pixel_threshold_px,
            device_str,
            processing_ms,
        )

    if args.store_descriptors:
        if experiment_id is None:
            raise RuntimeError("--store-descriptors requires --record so experiment_id is available")
        store_descriptors_for_run(
            db_path,
            experiment_id,
            dataset_path,
            scenes,
            set_id,
            descriptor_name,
            descriptor_cfg,
            device,
            hardnet_module,
        )


# -----------------------------------------------------------------------------
# Database recording
# -----------------------------------------------------------------------------

def record_results(
    db_path: Path,
    experiment_cfg: Dict,
    dataset_path: Path,
    descriptor_cfg: Dict,
    descriptor_name: str,
    matching_method: str,
    ratio_threshold: float,
    keypoint_set: str,
    metrics: AggregateMetrics,
    pixel_threshold_px: float,
    device_str: str,
    processing_ms: float,
) -> None:
    timestamp = datetime.utcnow().isoformat()
    pooling_strategy = descriptor_cfg.get("pooling", "none")
    similarity_threshold = ratio_threshold if matching_method == "ratio_test" else 0.0
    max_features = experiment_cfg.get("max_features") or descriptor_cfg.get("max_features") or 0

    experiment_name = experiment_cfg.get("name", f"python_{descriptor_name}_{keypoint_set}")
    description = experiment_cfg.get("description", "")

    params_dict = {
        "experiment_name": experiment_name,
        "description": description,
        "descriptor_type": descriptor_name,
        "matching_method": matching_method,
        "ratio_threshold": ratio_threshold if matching_method == "ratio_test" else None,
        "pixel_tolerance_px": pixel_threshold_px,
        "keypoint_set": keypoint_set,
        "device": device_str,
    }
    param_str = ";".join(
        f"{key}={value}"
        for key, value in params_dict.items()
        if value is not None and value != ""
    )

    metadata = {
        "precision": metrics.overall_precision,
        "precision_at_k": metrics.precision_at_k,
        "queries_with_gt": metrics.queries_with_gt,
        "excluded_queries": metrics.excluded_queries,
        "scenes": [
            {
                "name": scene.name,
                "matches": scene.matches,
                "inliers": scene.inliers,
                "precision": scene.precision,
                "map": scene.map_score,
                "queries_with_gt": scene.queries_with_gt,
                "excluded": scene.excluded_queries,
            }
            for scene in metrics.scenes
        ],
    }

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            (
                "INSERT INTO experiments (descriptor_type, dataset_name, pooling_strategy, "
                "similarity_threshold, max_features, timestamp, parameters) VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                descriptor_name,
                str(dataset_path),
                pooling_strategy,
                similarity_threshold,
                int(max_features),
                timestamp,
                param_str,
            ),
        )
        experiment_id = cursor.lastrowid

        cursor.execute(
            (
                "INSERT INTO results (experiment_id, true_map_macro, true_map_micro, "
                "true_map_macro_with_zeros, true_map_micro_with_zeros, mean_average_precision, "
                "legacy_mean_precision, precision_at_1, precision_at_5, recall_at_1, recall_at_5, "
                "total_matches, total_keypoints, processing_time_ms, timestamp, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                experiment_id,
                metrics.macro_map,
                metrics.micro_map,
                metrics.micro_map_including_zeros,
                metrics.micro_map_including_zeros,
                metrics.macro_map,
                metrics.overall_precision,
                metrics.precision_at_k.get(1, 0.0),
                metrics.precision_at_k.get(5, 0.0),
                metrics.precision_at_k.get(1, 0.0),
                metrics.precision_at_k.get(5, 0.0),
                metrics.total_matches,
                metrics.total_queries,
                processing_ms,
                timestamp,
                json.dumps(metadata),
            ),
        )

        conn.commit()

    print(f"Recorded experiment results in database (experiment_id={experiment_id})")
    return experiment_id


def store_descriptors_for_run(
    db_path: Path,
    experiment_id: int,
    dataset_path: Path,
    scenes: List[Path],
    set_id: int,
    descriptor_name: str,
    descriptor_cfg: Dict,
    device: torch.device,
    descriptor_module,
) -> None:
    processing_method = "python_" + descriptor_name
    normalization = descriptor_cfg.get("normalize_after_pooling", False)
    normalization_label = "l2" if normalization or descriptor_name == "libtorch_hardnet" else "none"
    rooting_label = "none"
    pooling_label = descriptor_cfg.get("pooling", "none")

    sql = (
        "INSERT INTO descriptors (experiment_id, scene_name, image_name, keypoint_x, keypoint_y, "
        "descriptor_vector, descriptor_dimension, processing_method, normalization_applied, rooting_applied, pooling_applied) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for scene_dir in scenes:
            scene_name = scene_dir.name
            for idx in range(1, 7):
                image_name = f"{idx}.ppm"
                image_path = scene_dir / image_name
                if not image_path.exists():
                    continue

                keypoints = load_keypoints_from_db(db_path, set_id, scene_name, image_name)
                if not keypoints:
                    continue

                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                if descriptor_name == "libtorch_hardnet":
                    descriptors = hardnet_descriptors(image, keypoints, device, descriptor_module)
                else:
                    descriptors = sift_descriptors(image, keypoints)

                if descriptors.size == 0:
                    continue

                records = []
                seen = set()
                for kp, desc in zip(keypoints, descriptors):
                    key = (round(float(kp.pt[0]), 4), round(float(kp.pt[1]), 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        (
                            experiment_id,
                            scene_name,
                            image_name,
                            key[0],
                            key[1],
                            sqlite3.Binary(desc.astype(np.float32).tobytes()),
                            int(desc.shape[0]),
                            processing_method,
                            normalization_label,
                            rooting_label,
                            pooling_label,
                        )
                    )

                cursor.executemany(sql, records)

        conn.commit()

    print(f"Stored {descriptor_name} descriptors for experiment_id={experiment_id}")

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate descriptors on stored keypoints via YAML config")
    parser.add_argument("--config", required=True, help="YAML config path (shared with C++ pipeline)")
    parser.add_argument("--db-path", default="experiments.db", help="Path to experiments.db")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device for Kornia descriptors")
    parser.add_argument("--output", help="Optional CSV summary output")
    parser.add_argument("--pixel-threshold", type=float, help="Override pixel tolerance (default 3px)")
    parser.add_argument("--record", action="store_true", help="Store results back into experiments.db")
    parser.add_argument("--store-descriptors", action="store_true", help="Persist per-keypoint descriptors to the database (requires --record)")
    return parser.parse_args()


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
