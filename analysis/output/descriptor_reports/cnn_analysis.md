# CNN Descriptor Analysis (HardNet / SOSNet)

This document summarizes all experiments involving learned CNN descriptors: HardNet and SOSNet.

## 1. Descriptor Overview

**HardNet (128D)**: Learned descriptor trained with hardest-in-batch triplet loss
- Zero-centered, dense embeddings
- Uses KeyNet detector by default
- LibTorch C++ implementation

**SOSNet (128D)**: Second Order Similarity network
- Similar architecture to HardNet
- Trained with second-order similarity loss
- Uses KeyNet detector by default

---

## 2. Baseline Performance by Keypoint Set

### HardNet (128D)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| keynet_8000 (native) | 65.80% | 63.81% | 65.30% | Baseline |
| keynet_scale_matched_6px | 78.95% | 76.89% | 79.29% | **+13.15%** |
| keynet_sift_8k_a (intersection) | 82.44% | 81.51% | 82.70% | SIFT-KeyNet intersection |
| keynet_sift_8k_b (intersection) | 20.80% | 20.57% | 28.71% | **WRONG SIDE** |
| sift_keynet_scale_matched_intersection_a | 50.95% | 50.01% | 52.34% | SIFT keypoints |

### SOSNet (128D)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| keynet_8000 (native) | 65.48% | 63.37% | 65.19% | -0.32% vs HardNet |
| keynet_scale_matched_6px | 78.79% | 76.58% | 79.34% | -0.16% vs HardNet |
| keynet_sift_8k_a (intersection) | 82.27% | 81.22% | 82.72% | -0.17% vs HardNet |
| keynet_sift_8k_b (intersection) | 20.69% | 20.41% | 28.36% | **WRONG SIDE** |
| sift_keynet_scale_matched_intersection_b | **93.69%** | 92.42% | 93.47% | **Best Single CNN** |

### Tolerance Study (HardNet)

| Tolerance | mAP | HP-V | HP-I | Notes |
|-----------|-----|------|------|-------|
| 1.0px (strict) | **90.68%** | 89.14% | 91.81% | Highest quality |
| 2.0px (moderate) | 85.24% | 84.02% | 85.30% | |
| 5.0px (relaxed) | 80.62% | 79.41% | 81.33% | |
| 10.0px (very relaxed) | 79.96% | 78.52% | 80.67% | Diminishing returns |

---

## 3. Comparison to SIFT Family

### On Comparable Keypoint Sets

| Descriptor | Unfiltered | Scale-Filtered | Best Intersection |
|------------|------------|----------------|-------------------|
| HardNet | 65.80% | 78.95% | 82.44% |
| SOSNet | 65.48% | 78.79% | **93.69%** |
| SIFT | 42.64% | 63.86% | N/A |
| DSPSIFT_V2 | 46.57% | 65.31% | 74.93% |

**Finding:** CNN descriptors outperform SIFT by ~23% on unfiltered, ~13% on filtered, and up to ~19% on optimal intersection configurations.

### Performance Gap Analysis

| Comparison | SIFT | HardNet | Gap |
|------------|------|---------|-----|
| Unfiltered | 42.64% | 65.80% | **+23.16%** |
| Scale-filtered | 63.86% | 78.95% | **+15.09%** |
| Intersection (comparable) | 74.93% | 82.44% | **+7.51%** |

**Finding:** The gap between SIFT and CNN descriptors **narrows** as keypoint quality increases.

---

## 4. Effect of Keypoint Quality

### Scale Filtering Impact

| Descriptor | Unfiltered | Scale-Filtered | Improvement |
|------------|------------|----------------|-------------|
| HardNet | 65.80% | 78.95% | **+13.15%** |
| SOSNet | 65.48% | 78.79% | **+13.31%** |

**Finding:** Scale filtering provides ~13% improvement for CNN descriptors (vs ~20% for SIFT).

### Intersection Set Impact

| Descriptor | Scale-Filtered | Intersection | Improvement |
|------------|----------------|--------------|-------------|
| HardNet | 78.95% | 82.44% | **+3.49%** |
| SOSNet | 78.79% | 93.69% | **+14.90%** |

**Finding:** SOSNet shows dramatic improvement on optimal intersection configuration.

### Important: Intersection Side Matters

| Descriptor | Intersection A (KeyNet side) | Intersection B (SIFT side) |
|------------|------------------------------|---------------------------|
| HardNet | 82.44% | 20.80% |
| SOSNet | 82.27% | 93.69% (scale-matched) |

**Critical Finding:** Using the wrong intersection side causes **catastrophic failure** (20% mAP). CNN descriptors must be computed on KeyNet keypoints, not SIFT keypoints.

---

## 5. Fusion Experiments

### HardNet + SOSNet Fusion (CNN + CNN)

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| keynet_sift_8k_a | concatenate | 83.07% | +0.63% vs HardNet |
| keynet_sift_8k_a | weighted_avg | 81.19% | -1.25% vs HardNet |
| sift_keynet_scale_matched_intersection_b | concatenate | **94.07%** | +0.38% vs SOSNet |
| sift_keynet_scale_matched_intersection_b | weighted_avg | 93.05% | -0.64% vs SOSNet |
| tol 2px | concatenate | 85.84% | - |
| tol 5px | concatenate | 81.28% | - |

**Finding:** CNN + CNN fusion provides **marginal benefit** (+0.4-0.6%). Concatenation outperforms averaging.

### SIFT + HardNet Fusion (Cross-Family)

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| keynet_sift_8k (unfiltered) | concatenate | 55.32% | **-10.48% vs HardNet** |
| keynet_sift_8k (unfiltered) | weighted_avg | 55.32% | **-10.48% vs HardNet** |
| sift_keynet_scale_matched_intersection | concatenate | 72.33% | **-10.11% vs HardNet** |
| sift_keynet_scale_matched_intersection | weighted_avg | 72.32% | **-10.12% vs HardNet** |

**Finding:** SIFT + HardNet fusion **degrades performance by ~10%**.

### RootSIFT + HardNet Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| keynet_sift_8k | concatenate | 55.32% | Same as SIFT+HardNet |
| keynet_sift_8k | weighted_avg | 55.32% | Same |
| sift_keynet_scale_matched_intersection | concatenate | 72.33% | Same |
| sift_keynet_scale_matched_intersection | weighted_avg | 72.32% | Same |

**Finding:** RootSIFT provides **no improvement** over SIFT for CNN fusion.

### DSPSIFT + HardNet Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| keynet_sift_8k | concatenate | 16.76% | **CATASTROPHIC FAIL** |
| keynet_sift_8k | weighted_avg | 13.21% | **CATASTROPHIC FAIL** |
| sift_keynet_scale_matched_intersection | concatenate | 32.41% | **SEVERE FAIL** |
| sift_keynet_scale_matched_intersection | weighted_avg | 27.44% | **SEVERE FAIL** |

**Finding:** DSPSIFT + HardNet fusion **fails catastrophically** (13-32% mAP vs 82% baseline).

---

## 6. Why Cross-Family Fusion Fails

### Value Distribution Analysis

| Descriptor | Value Range | Mean | Distribution |
|------------|-------------|------|--------------|
| SIFT | [0, 1] | ~0.01 | Non-negative, sparse |
| HardNet | [-1, 1] | ~0 | Zero-centered, dense |

**Root Cause:** L2 distance computation on concatenated/averaged descriptors is dominated by the mismatched distributions:
1. SIFT values are all positive, HardNet values are zero-centered
2. Averaging destroys learned representations
3. Concatenation creates heterogeneous feature space

### DSP Makes It Worse

DSP pooling averages across scales, which:
1. Smooths out distinctive features
2. Changes the value distribution further
3. Creates even greater incompatibility with CNN embeddings

---

## 7. Summary Rankings

### Best CNN Configurations

1. **HardNet+SOSNet (concat) on sift_keynet_scale_matched_intersection_b**: 94.07%
2. **SOSNet on sift_keynet_scale_matched_intersection_b**: 93.69%
3. **HardNet+SOSNet (avg) on sift_keynet_scale_matched_intersection_b**: 93.05%
4. **HardNet (tol 1px strict)**: 90.68%
5. **HardNet+SOSNet (concat) on tol 2px**: 85.84%
6. **HardNet on keynet_sift_8k_a**: 82.44%

### Key Conclusions

1. **CNN descriptors significantly outperform SIFT** (+15-23% depending on keypoint quality)
2. **Scale filtering provides ~13% improvement** for CNN (vs ~20% for SIFT)
3. **CNN + CNN fusion provides marginal benefit** (+0.4-0.6%)
4. **Cross-family fusion with SIFT fails** (-10% degradation)
5. **DSPSIFT + CNN fusion fails catastrophically** (13-32% mAP)
6. **Intersection side matters critically** - wrong side = 20% mAP
7. **Best overall result: 94.07% mAP** (HardNet+SOSNet concatenation)

---

## 8. Missing Experiments / Gaps

| Gap | Description | Priority |
|-----|-------------|----------|
| L2-Net evaluation | Listed in framework but not tested | Medium |
| SOSNet on strict tolerance | Only HardNet tested with 1px tolerance | Low |
| CNN on SURF intersection | CNN on SIFT-SURF intersection | Low |
| HardNet on SIFT detector | HardNet computed on SIFT keypoints (not intersection) | Low |

---

## 9. Recommendations

1. **Always use KeyNet keypoints** for CNN descriptors
2. **Use scale-matched intersection** for best results (93-94% mAP)
3. **Prefer concatenation** over averaging for fusion
4. **Never fuse SIFT with CNN** - degrades both
5. **Avoid DSP pooling** when planning CNN fusion
6. **Use strict tolerance (1px)** for highest quality intersection

