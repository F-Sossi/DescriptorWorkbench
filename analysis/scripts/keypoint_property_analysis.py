#!/usr/bin/env python3
"""
Keypoint Property Analysis for Intersection Investigation

Analyzes differences in keypoint properties between full SIFT set
and SIFT-SURF intersection set.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup
output_dir = Path(__file__).parent.parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

db_path = 'build/experiments.db'
conn = sqlite3.connect(db_path)

print("=" * 80)
print("KEYPOINT PROPERTY ANALYSIS")
print("=" * 80)
print()

# Load all keypoints with properties (sample from each set)
query1 = """
SELECT
    ks.name as keypoint_set,
    lk.response,
    lk.size,
    lk.octave,
    lk.scene_name
FROM locked_keypoints lk
JOIN keypoint_sets ks ON lk.keypoint_set_id = ks.id
WHERE ks.name = 'sift_8000'
ORDER BY RANDOM()
LIMIT 50000
"""

query2 = """
SELECT
    ks.name as keypoint_set,
    lk.response,
    lk.size,
    lk.octave,
    lk.scene_name
FROM locked_keypoints lk
JOIN keypoint_sets ks ON lk.keypoint_set_id = ks.id
WHERE ks.name = 'sift_surf_intersection_a'
ORDER BY RANDOM()
LIMIT 50000
"""

print("Loading keypoint properties (sampling 50k from each set)...")
df1 = pd.read_sql(query1, conn)
df2 = pd.read_sql(query2, conn)
df = pd.concat([df1, df2], ignore_index=True)
print(f"Loaded {len(df)} keypoints ({len(df1)} from sift_8000, {len(df2)} from intersection)")
print()

# Summary statistics
print("Summary Statistics:")
print("-" * 80)
summary = df.groupby('keypoint_set').agg({
    'response': ['count', 'mean', 'std', 'min', 'median', 'max'],
    'size': ['mean', 'std', 'min', 'median', 'max'],
    'octave': ['mean', 'std', 'min', 'max']
})
print(summary)
print()

# Percentile analysis
print("Response Percentiles:")
print("-" * 80)
for name in ['sift_8000', 'sift_surf_intersection_a']:
    subset = df[df['keypoint_set'] == name]['response']
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(subset, percentiles)
    print(f"{name}:")
    for p, v in zip(percentiles, values):
        print(f"  P{p:02d}: {v:.6f}")
    print()

print("Size Percentiles:")
print("-" * 80)
for name in ['sift_8000', 'sift_surf_intersection_a']:
    subset = df[df['keypoint_set'] == name]['size']
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(subset, percentiles)
    print(f"{name}:")
    for p, v in zip(percentiles, values):
        print(f"  P{p:02d}: {v:.2f}")
    print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Keypoint Property Comparison: Full Set vs Intersection', fontsize=14, fontweight='bold')

# Plot 1: Response distribution
ax1 = axes[0, 0]
for name in ['sift_8000', 'sift_surf_intersection_a']:
    subset = df[df['keypoint_set'] == name]['response']
    label = 'Full (sift_8000)' if 'sift_8000' in name else 'Intersection (surf-sift)'
    ax1.hist(subset, bins=50, alpha=0.6, label=label, density=True)
ax1.set_xlabel('Response Strength', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Response Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Size distribution
ax2 = axes[0, 1]
for name in ['sift_8000', 'sift_surf_intersection_a']:
    subset = df[df['keypoint_set'] == name]['size']
    label = 'Full (sift_8000)' if 'sift_8000' in name else 'Intersection (surf-sift)'
    ax2.hist(subset, bins=50, alpha=0.6, label=label, density=True)
ax2.set_xlabel('Keypoint Size (scale)', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Size Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 50)  # Focus on main distribution

# Plot 3: Octave distribution
ax3 = axes[1, 0]
octave_counts = df.groupby(['keypoint_set', 'octave']).size().unstack(fill_value=0)
octave_counts_pct = octave_counts.div(octave_counts.sum(axis=1), axis=0) * 100
octave_counts_pct.T.plot(kind='bar', ax=ax3, alpha=0.7)
ax3.set_xlabel('Octave Level', fontsize=11)
ax3.set_ylabel('Percentage of Keypoints', fontsize=11)
ax3.set_title('Octave Distribution', fontsize=12, fontweight='bold')
ax3.legend(['Full (sift_8000)', 'Intersection (surf-sift)'], loc='best')
ax3.grid(alpha=0.3, axis='y')

# Plot 4: Response vs Size scatter (sample)
ax4 = axes[1, 1]
sample_size = 5000
for name in ['sift_8000', 'sift_surf_intersection_a']:
    subset = df[df['keypoint_set'] == name].sample(n=min(sample_size, len(df[df['keypoint_set'] == name])))
    label = 'Full (sift_8000)' if 'sift_8000' in name else 'Intersection (surf-sift)'
    ax4.scatter(subset['size'], subset['response'], alpha=0.3, s=10, label=label)
ax4.set_xlabel('Keypoint Size (scale)', fontsize=11)
ax4.set_ylabel('Response Strength', fontsize=11)
ax4.set_title('Response vs Size (5k sample per set)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 50)
ax4.set_ylim(0, 0.15)

plt.tight_layout()
output_path = output_dir / 'keypoint_properties_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved plot: {output_path}")

# Statistical tests
from scipy import stats

print()
print("=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)
print()

response_full = df[df['keypoint_set'] == 'sift_8000']['response']
response_intersection = df[df['keypoint_set'] == 'sift_surf_intersection_a']['response']

size_full = df[df['keypoint_set'] == 'sift_8000']['size']
size_intersection = df[df['keypoint_set'] == 'sift_surf_intersection_a']['size']

# T-test for response
t_stat_response, p_value_response = stats.ttest_ind(response_full, response_intersection)
print("Response Strength T-Test:")
print(f"  Full set mean:         {response_full.mean():.6f}")
print(f"  Intersection mean:     {response_intersection.mean():.6f}")
print(f"  Difference:            {response_intersection.mean() - response_full.mean():.6f} ({((response_intersection.mean() - response_full.mean()) / response_full.mean() * 100):.2f}%)")
print(f"  t-statistic:           {t_stat_response:.4f}")
print(f"  p-value:               {p_value_response:.2e}")
if p_value_response < 0.001:
    print(f"  Result:                ✅ Highly significant difference (p < 0.001)")
elif p_value_response < 0.05:
    print(f"  Result:                ✅ Significant difference (p < 0.05)")
else:
    print(f"  Result:                ❌ No significant difference")
print()

# T-test for size
t_stat_size, p_value_size = stats.ttest_ind(size_full, size_intersection)
print("Keypoint Size T-Test:")
print(f"  Full set mean:         {size_full.mean():.2f}")
print(f"  Intersection mean:     {size_intersection.mean():.2f}")
print(f"  Difference:            {size_intersection.mean() - size_full.mean():.2f} ({((size_intersection.mean() - size_full.mean()) / size_full.mean() * 100):.2f}%)")
print(f"  t-statistic:           {t_stat_size:.4f}")
print(f"  p-value:               {p_value_size:.2e}")
if p_value_size < 0.001:
    print(f"  Result:                ✅ Highly significant difference (p < 0.001)")
elif p_value_size < 0.05:
    print(f"  Result:                ✅ Significant difference (p < 0.05)")
else:
    print(f"  Result:                ❌ No significant difference")
print()

# Effect size (Cohen's d)
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (x.mean() - y.mean()) / np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / dof)

d_response = cohens_d(response_intersection, response_full)
d_size = cohens_d(size_intersection, size_full)

print("Effect Sizes (Cohen's d):")
print(f"  Response:  {d_response:.4f} ({'small' if abs(d_response) < 0.5 else 'medium' if abs(d_response) < 0.8 else 'large'})")
print(f"  Size:      {d_size:.4f} ({'small' if abs(d_size) < 0.5 else 'medium' if abs(d_size) < 0.8 else 'large'})")
print()

# Conclusions
print("=" * 80)
print("PRELIMINARY CONCLUSIONS")
print("=" * 80)
print()

if abs(d_response) > 0.2:
    print("📊 Response Strength:")
    print(f"   → Intersection keypoints have {((response_intersection.mean() - response_full.mean()) / response_full.mean() * 100):.2f}% higher response")
    print(f"   → Effect size: {d_response:.4f} ({'small' if abs(d_response) < 0.5 else 'medium' if abs(d_response) < 0.8 else 'large'})")
    print(f"   → This suggests H2 (quality selection) may be valid")
else:
    print("📊 Response Strength:")
    print(f"   → No meaningful difference in response values")
    print(f"   → H2 (quality selection) less likely")
print()

if abs(d_size) > 0.5:
    print("📏 Keypoint Size (Scale):")
    print(f"   → Intersection keypoints are {((size_intersection.mean() - size_full.mean()) / size_full.mean() * 100):.2f}% larger")
    print(f"   → Effect size: {d_size:.4f} ({'small' if abs(d_size) < 0.5 else 'medium' if abs(d_size) < 0.8 else 'large'})")
    print(f"   → This suggests H3 (scale selection) may be valid")
    print(f"   → Intersection favors larger-scale features")
else:
    print("📏 Keypoint Size (Scale):")
    print(f"   → No meaningful difference in keypoint scale")
    print(f"   → H3 (scale selection) less likely")
print()

print("🔬 Next Steps:")
print("   1. Generate control keypoint sets (random, top-response, top-size)")
print("   2. Run experiments to measure MAP differences")
print("   3. Determine if property differences explain MAP gains")
print()

conn.close()
print("=" * 80)
print("Analysis complete!")
print("=" * 80)
