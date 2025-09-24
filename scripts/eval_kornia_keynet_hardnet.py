#!/usr/bin/env python3
"""
Evaluate Kornia's KeyNet + HardNet pipeline on HPatches and report metrics
comparable to the C++ experiment_runner (mAP, P@K, etc.).

This script mirrors the tutorial pipeline from Kornia documentation:
- KeyNet detector (multi-resolution) + HardNet descriptor
- Ratio-test matching (SNN, threshold configurable)
- Homography-based geometric verification using dataset ground truth

In addition to match precision it computes Information Retrieval style
mean Average Precision (micro/macro) using the single-ground-truth policy
employed in the DescriptorWorkbench project.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF


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
# Utility helpers
# -----------------------------------------------------------------------------

def load_image(path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    tensor = K.image_to_tensor(img, keepdim=False).float() / 255.0
    return tensor, img


def detect_and_describe(
    model: KF.KeyNetHardNet,
    image_tensor: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        lafs, responses, descriptors = model(image_tensor)
    centers = KF.laf.get_laf_center(lafs).cpu().numpy()[0]  # (N, 2)
    responses = responses.cpu().numpy()[0]
    descriptors = descriptors.cpu().numpy()[0]
    return centers, responses, descriptors


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    matcher = KF.DescriptorMatcher("snn", threshold)
    with torch.no_grad():
        _, indices = matcher(
            torch.from_numpy(desc1).float(),
            torch.from_numpy(desc2).float(),
        )
    return indices.numpy()


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    points_h = cv2.perspectiveTransform(points.reshape(-1, 1, 2), H)
    return points_h.reshape(-1, 2)


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


def evaluate_ratio_matches(
    centers_ref: np.ndarray,
    centers_dst: np.ndarray,
    matches: np.ndarray,
    homography: np.ndarray,
    pixel_threshold: float,
) -> PairMetrics:
    if matches.size == 0:
        return PairMetrics(matches=0, inliers=0, precision=0.0, mean_error=math.inf)

    matched_ref = centers_ref[matches[:, 0]]
    matched_dst = centers_dst[matches[:, 1]]

    projected = apply_homography(matched_ref, homography)
    errors = np.linalg.norm(projected - matched_dst, axis=1)

    inlier_mask = errors <= pixel_threshold
    inliers = int(inlier_mask.sum())
    precision = inliers / len(errors) if len(errors) else 0.0
    mean_error = float(errors[inlier_mask].mean()) if inliers else float(errors.mean())

    return PairMetrics(
        matches=int(len(errors)),
        inliers=inliers,
        precision=precision,
        mean_error=mean_error,
    )


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

    for qi, query_point in enumerate(centers_ref):
        relevant_idx = find_relevant_index(query_point, centers_dst, homography, pixel_threshold)
        if relevant_idx is None:
            excluded += 1
            continue

        queries_with_gt += 1
        query_desc = desc_ref[qi]
        distances = np.linalg.norm(desc_dst - query_desc, axis=1)
        order = np.argsort(distances, kind="stable")
        rank_position = int(np.nonzero(order == relevant_idx)[0][0]) + 1

        ap = 1.0 / rank_position
        ap_values_global.append(ap)
        ap_values_scene.append(ap)

        for k in ks:
            if rank_position <= k:
                hits_at_k[k] += 1

    return queries_with_gt, excluded


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    dataset_root = Path(args.data_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model = KF.KeyNetHardNet(num_features=args.max_keypoints, device=device, upright=args.upright)
    model.eval()

    pixel_threshold = args.pixel_threshold
    ks = (1, 5, 10)

    scenes = sorted(d for d in dataset_root.iterdir() if d.is_dir())
    if args.scenes:
        filters = set(args.scenes)
        scenes = [d for d in scenes if d.name in filters]

    global_hits_at_k: Dict[int, int] = {k: 0 for k in ks}
    global_ap_values: List[float] = []
    global_matches = 0
    global_inliers = 0
    global_queries_with_gt = 0
    global_excluded = 0
    global_total_queries = 0

    scene_results: List[SceneMetrics] = []

    for scene_dir in scenes:
        print(f"\nScene: {scene_dir.name}")
        ref_tensor, _ = load_image(scene_dir / "1.ppm")
        centers_ref, responses_ref, desc_ref = detect_and_describe(model, ref_tensor)
        print(f"  Reference keypoints: {len(centers_ref)} (mean response {responses_ref.mean():.4f})")

        scene_matches = 0
        scene_inliers = 0
        scene_ap_values: List[float] = []
        scene_queries_with_gt = 0
        scene_excluded = 0
        scene_total_queries = 0

        for idx in range(2, 7):
            img_path = scene_dir / f"{idx}.ppm"
            if not img_path.exists():
                continue

            homography_path = scene_dir / f"H_1_{idx}"
            if not homography_path.exists():
                print(f"    Missing homography for {img_path.name}, skipping")
                continue

            dst_tensor, _ = load_image(img_path)
            centers_dst, responses_dst, desc_dst = detect_and_describe(model, dst_tensor)

            matches = match_descriptors(desc_ref, desc_dst, threshold=args.ratio_threshold)
            H = np.loadtxt(str(homography_path))

            pair_stats = evaluate_ratio_matches(centers_ref, centers_dst, matches, H, pixel_threshold)
            scene_matches += pair_stats.matches
            scene_inliers += pair_stats.inliers

            print(
                f"    {img_path.name}: matches={pair_stats.matches}, inliers={pair_stats.inliers}, "
                f"precision={pair_stats.precision:.3f}, mean_error={pair_stats.mean_error:.2f}"
            )

            queries_with_gt, excluded = accumulate_map_metrics(
                centers_ref,
                desc_ref,
                centers_dst,
                desc_dst,
                H,
                pixel_threshold,
                global_hits_at_k,
                global_ap_values,
                scene_ap_values,
                ks,
            )

            scene_queries_with_gt += queries_with_gt
            scene_excluded += excluded
            scene_total_queries += len(centers_ref)
            global_queries_with_gt += queries_with_gt
            global_excluded += excluded
            global_total_queries += len(centers_ref)

        scene_precision = scene_inliers / scene_matches if scene_matches else 0.0
        scene_map = sum(scene_ap_values) / len(scene_ap_values) if scene_ap_values else 0.0
        print(
            "  Scene totals: matches={}, inliers={}, precision={:.3f}, micro mAP={:.4f}".format(
                scene_matches,
                scene_inliers,
                scene_precision,
                scene_map,
            )
        )

        scene_results.append(
            SceneMetrics(
                name=scene_dir.name,
                matches=scene_matches,
                inliers=scene_inliers,
                precision=scene_precision,
                map_score=scene_map,
                queries_with_gt=scene_queries_with_gt,
                excluded_queries=scene_excluded,
                total_queries=scene_total_queries,
            )
        )

        global_matches += scene_matches
        global_inliers += scene_inliers

    overall_precision = global_inliers / global_matches if global_matches else 0.0
    micro_map = sum(global_ap_values) / len(global_ap_values) if global_ap_values else 0.0

    scene_maps = [s.map_score for s in scene_results if s.map_score > 0]
    macro_map = sum(scene_maps) / len(scene_maps) if scene_maps else 0.0

    total_queries_all = global_queries_with_gt + global_excluded
    micro_map_including_zeros = (
        sum(global_ap_values) / total_queries_all
        if total_queries_all > 0
        else 0.0
    )

    precision_at_k = {
        k: (global_hits_at_k[k] / len(global_ap_values) if global_ap_values else 0.0)
        for k in ks
    }

    print("\n================ Summary ================")
    print(f"Scenes evaluated: {len(scene_results)}")
    print(f"Total matches: {global_matches}")
    print(f"Total inliers: {global_inliers}")
    print(f"Overall precision (homography inliers / matches): {overall_precision:.3f}")
    print(f"Queries with GT matches: {global_queries_with_gt}")
    print(f"Queries without GT (excluded): {global_excluded}")
    print(f"Micro mAP (queries with GT): {micro_map:.4f}")
    print(f"Macro mAP (per-scene): {macro_map:.4f}")
    print(f"Micro mAP incl. zeros: {micro_map_including_zeros:.4f}")
    print(
        "Precision@K: "
        + ", ".join(f"P@{k}={precision_at_k[k]:.4f}" for k in ks)
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
                    overall_precision,
                    micro_map,
                    global_queries_with_gt,
                    global_excluded,
                    global_total_queries,
                )
            )
        print(f"Saved summary to {output_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kornia KeyNet+HardNet on HPatches")
    parser.add_argument("--data-root", default="data", help="HPatches dataset root")
    parser.add_argument("--max-keypoints", type=int, default=2000, help="Keypoints per image")
    parser.add_argument("--ratio-threshold", type=float, default=0.8, help="SNN ratio threshold")
    parser.add_argument("--pixel-threshold", type=float, default=3.0, help="Inlier threshold in pixels")
    parser.add_argument("--upright", action="store_true", help="Force upright descriptors (no orientation)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--scenes", nargs="*", help="Subset of scene names to evaluate")
    parser.add_argument("--output", help="Optional CSV summary path")
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_evaluation(parse_args())
