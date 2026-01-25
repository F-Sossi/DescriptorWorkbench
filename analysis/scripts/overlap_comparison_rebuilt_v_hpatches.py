#!/usr/bin/env python3
"""
Compare overlap distributions between original HPatches and rebuilt patches.

This script reads *.overlaps files for e/h/t difficulty across all scenes and
prints summary stats (min/p25/median/p75/max). It is used to sanity-check whether
the rebuilt dataset matches the official overlap distribution.

Usage:
  python3 analysis/scripts/overlap_comparison_rebuilt_v_hpatches.py

Notes:
  - Only scene folders with names like i_* or v_* are considered.
  - The rebuilt dataset path is hardcoded below; update if needed.
  - A "Remainder of file ignored" warning from conda is harmless and can be
    removed by upgrading setuptools in the active env.

Interpreting the output:
  - Each overlap value is the IoU between the jittered patch region and the
    reference region for a specific scene/difficulty/image pair.
  - The medians for HPatches are typically around:
      easy ~0.85, hard ~0.72, tough ~0.50 (paper guidance).
  - If rebuilt medians are lower, the dataset is harder (more geometric noise).
  - If rebuilt medians are higher, the dataset is easier (less noise).
  - Use p25/p75 to assess spread; a wider gap indicates more variability.
"""

from pathlib import Path
import numpy as np

def collect_overlaps(root):
    data = { 'e': [], 'h': [], 't': [] }
    for scene_dir in Path(root).iterdir():
        if not scene_dir.is_dir():
            continue
        name = scene_dir.name
        if not (len(name) > 2 and name[1] == '_' and name[0] in ('i','v')):
            continue
        for prefix in ['e','h','t']:
            for i in range(1,6):
                path = scene_dir / f"{prefix}{i}.overlaps"
                if not path.exists():
                    continue
                vals = np.loadtxt(path, dtype=float)
                if vals.size == 0:
                    continue
                data[prefix].append(np.ravel(vals))
    out = {}
    for prefix, chunks in data.items():
        if not chunks:
            out[prefix] = np.array([])
        else:
            out[prefix] = np.concatenate(chunks)
    return out

def summarize(label, data):
    print(label)
    for prefix in ['e','h','t']:
        vals = data[prefix]
        if vals.size == 0:
            print(f"  {prefix}: no data")
            continue
        print(
            f"  {prefix}: n={vals.size} min={vals.min():.4f} "
            f"p25={np.percentile(vals,25):.4f} median={np.median(vals):.4f} "
            f"p75={np.percentile(vals,75):.4f} max={vals.max():.4f}"
        )

orig = collect_overlaps('/home/frank/repos/DescriptorWorkbench/hpatches-release')
rebuilt = collect_overlaps('/home/frank/repos/DescriptorWorkbench/hpatches-release-rebuilt-bw')

summarize('original', orig)
summarize('rebuilt', rebuilt)
