# Investigation 3: Per-Transformation Breakdown

## Overview

**Goal**: Analyze how intersection benefit varies across different transformation types (blur, zoom, rotation, lighting changes).

**Hypothesis**: Intersection keypoints may provide greater benefits for specific transformation types, revealing which image conditions benefit most from detector consensus.

---

## 1. Background

HPatches dataset includes two categories:
- **HP-V (Viewpoint)**: Geometric transformations (rotation, scale, perspective)
- **HP-I (Illumination)**: Photometric changes (lighting, exposure, blur)

Within each scene, images 2-6 represent **increasing transformation severity**.

From our initial analysis:
- Viewpoint scenes: Intersection gains **+3.8%** mAP
- Illumination scenes: Intersection gains **+2.0%** mAP

**Question**: Can we identify specific transformations where intersection helps most?

---

## 2. HPatches Transformation Structure

### 2.1 Scene Categories

| Category | Prefix | Transformations |
|----------|--------|-----------------|
| Viewpoint | v_ | Rotation, scale, perspective warp |
| Illumination | i_ | Lighting change, blur, exposure |

### 2.2 Transformation Severity

| Image | Severity | Description |
|-------|----------|-------------|
| 1 | Reference | Base image |
| 2 | Mild | Small change |
| 3 | Moderate | Medium change |
| 4 | Medium-High | Notable change |
| 5 | High | Significant change |
| 6 | Extreme | Maximum change |

---

## 3. Experimental Design

### 3.1 Data Available

We already have per-scene mAP in the results metadata. We can:
1. Group scenes by transformation type
2. Analyze by image pair (1→2, 1→3, etc.)
3. Correlate with transformation severity

### 3.2 Grouping Strategy

**Option A: By Scene Category (Already Done)**
- HP-V vs HP-I

**Option B: By Transformation Severity**
- Mild (image 2) vs Extreme (image 6)

**Option C: By Specific Scene Characteristics**
- Manual labeling of scene types (blur-dominant, rotation-dominant, etc.)

### 3.3 Analysis Approach

**Phase 1: Severity Analysis**
Compare intersection benefit by image pair:

```python
def analyze_by_severity(metadata):
    """
    Parse per-pair metrics from metadata.

    Metadata contains: scene_pair_2_map, scene_pair_3_map, etc.
    """
    results = {}
    for pair in range(2, 7):
        int_maps = extract_pair_maps(metadata['intersection'], pair)
        pure_maps = extract_pair_maps(metadata['pure_scale'], pair)

        results[pair] = {
            'intersection_avg': np.mean(int_maps),
            'pure_scale_avg': np.mean(pure_maps),
            'gain': np.mean(int_maps) - np.mean(pure_maps)
        }

    return results
```

**Phase 2: Scene Clustering**
Group scenes by transformation characteristics and compare intersection gains.

---

## 4. Metrics to Compute

### 4.1 By Transformation Severity

| Metric | Description |
|--------|-------------|
| mAP by pair | Average mAP for image pairs 1→2, 1→3, ..., 1→6 |
| Intersection gain by pair | How much intersection helps at each severity |
| Degradation rate | How fast mAP drops with severity |

### 4.2 By Scene Category

| Category | Metrics |
|----------|---------|
| HP-V | mAP, gain, degradation rate |
| HP-I | mAP, gain, degradation rate |
| HP-V subscenes | Rotation-heavy, scale-heavy, perspective-heavy |
| HP-I subscenes | Blur-heavy, lighting-heavy, exposure-heavy |

---

## 5. Expected Outcomes

### 5.1 If Intersection Helps More with Geometric Transformations

- Larger gains in HP-V than HP-I (already observed: +3.8% vs +2.0%)
- Gains increase with transformation severity
- Intersection degradation rate is lower

### 5.2 If Intersection Helps Uniformly

- Similar gains across all transformation types
- No clear pattern with severity

### 5.3 If Intersection Helps More with Specific Transformations

- Identify which transformation types benefit most
- Could guide when to use multi-detector intersection

---

## 6. Implementation Plan

### 6.1 Phase 1: Check Metadata Structure

First, verify what per-pair data is available in metadata:

```sql
SELECT r.metadata
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE e.descriptor_type LIKE '%intersection%'
LIMIT 1;
```

Parse to see if per-pair metrics exist or only per-scene.

### 6.2 Phase 2: Per-Pair Analysis (if available)

If per-pair data exists:

```python
def extract_pair_metrics(metadata_str):
    """Extract per-pair metrics from metadata string."""
    import re

    pairs = {}
    for pair in range(2, 7):
        pattern = rf'(\w+)_pair_{pair}_map=([0-9.]+)'
        for match in re.finditer(pattern, metadata_str):
            scene = match.group(1)
            map_val = float(match.group(2))
            if pair not in pairs:
                pairs[pair] = {}
            pairs[pair][scene] = map_val

    return pairs
```

### 6.3 Phase 3: Scene Categorization

Create scene categories based on dominant transformation:

```python
# Manual categorization (or automated via homography analysis)
SCENE_CATEGORIES = {
    'rotation_heavy': ['v_adam', 'v_bark', ...],
    'scale_heavy': ['v_boat', 'v_bricks', ...],
    'perspective_heavy': ['v_graffiti', 'v_wall', ...],
    'blur_heavy': ['i_dc', 'i_fog', ...],
    'lighting_heavy': ['i_crownnight', 'i_lionnight', ...],
}
```

### 6.4 Phase 4: Homography Analysis (Automated)

Extract transformation type from homography matrices:

```python
def classify_homography(H):
    """
    Classify homography by dominant transformation type.

    Returns: 'rotation', 'scale', 'perspective', 'translation'
    """
    # Decompose homography
    # H = [R | t] for affine, plus perspective terms

    # Rotation: off-diagonal elements of upper-left 2x2
    rotation_magnitude = abs(H[0,1]) + abs(H[1,0])

    # Scale: diagonal elements differ from 1
    scale_change = abs(H[0,0] - 1) + abs(H[1,1] - 1)

    # Perspective: bottom row differs from [0, 0, 1]
    perspective = abs(H[2,0]) + abs(H[2,1])

    return classify(rotation_magnitude, scale_change, perspective)
```

---

## 7. Visualization Plan

### 7.1 Degradation Curves

```
mAP |
80% |*
    | \*
70% |  \*           ← Intersection
    |   \*
60% |    \*----*
    |     \*
50% |*     \*       ← Pure Scale
    | \*    \*
40% |  \*----\*
    +----+----+----+----+----+
         2    3    4    5    6
            Image Pair (Severity)
```

### 7.2 Gain by Severity

```
Intersection Gain
+8% |              *
    |           *
+6% |        *
    |     *
+4% |  *
    +----+----+----+----+----+
         2    3    4    5    6
            Image Pair (Severity)
```

### 7.3 Heatmap by Scene × Severity

| Scene | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Pair 6 |
|-------|--------|--------|--------|--------|--------|
| v_colors | +5% | +10% | +20% | +25% | +32% |
| i_brooklyn | -2% | -5% | -8% | -10% | -12% |
| ... | ... | ... | ... | ... | ... |

---

## 8. Experiment Configuration

**Option A: Use existing per-scene data**
- No new experiments needed
- Analyze metadata from completed experiments

**Option B: Run per-pair experiments**
- Modify experiment runner to report per-pair metrics
- Store in metadata with format: `scene_pair_N_map=X.XX`

---

## 9. Questions to Answer

1. Does intersection gain increase with transformation severity?
2. Which transformation type benefits most from intersection?
3. Are there scenes where intersection hurts at high severity?
4. Does the degradation rate differ between keypoint sets?
5. Can we predict intersection benefit from homography characteristics?

---

## 10. Dependencies

- [x] Per-scene mAP in metadata (already available)
- [ ] Per-pair mAP in metadata (check if available)
- [ ] Scene categorization by transformation type
- [ ] Homography analysis script
- [ ] Visualization scripts

---

## 11. Status

| Phase | Status | Notes |
|-------|--------|-------|
| Design | ✅ Complete | This document |
| Data check | ⬜ Not started | Verify metadata structure |
| Implementation | ⬜ Not started | |
| Analysis | ⬜ Not started | |
| Documentation | ⬜ Not started | |

---

## 12. Quick Start

Check what per-pair data is available:

```bash
sqlite3 build/experiments.db "
SELECT r.metadata
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE e.descriptor_type LIKE '%intersection%'
LIMIT 1;
" | tr ';' '\n' | grep -E 'pair|_2_|_3_|_4_|_5_|_6_' | head -20
```

---

## 13. Related Documents

- `intersection_mechanism_analysis.md` - Parent investigation
- `investigation_2_repeatability_tracking.md` - Related: repeatability by severity
- HPatches dataset documentation

---

*Document created: December 2025*
