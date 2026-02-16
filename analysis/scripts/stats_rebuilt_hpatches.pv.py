#!/usr/bin/env python3

import pathlib
root = pathlib.Path('/home/frank/repos/DescriptorWorkbench/hpatches-release-rebuilt-bw')
counts = []
for path in root.rglob('ref.rotjitter'):
    with path.open('r') as f:
        n = sum(1 for _ in f)
    counts.append((path.parent.name, n))

values = [n for _, n in counts]
min_scene, min_val = min(counts, key=lambda x: x[1])
max_scene, max_val = max(counts, key=lambda x: x[1])
avg_val = sum(values) / len(values)
print(f'scenes={len(values)} min={min_val} ({min_scene}) max={max_val} ({max_scene}) avg={avg_val:.2f}')