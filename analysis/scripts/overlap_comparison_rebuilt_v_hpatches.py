#!/usr/bin/env python3

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
