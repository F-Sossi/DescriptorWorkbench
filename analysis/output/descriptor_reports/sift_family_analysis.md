# SIFT Family Descriptor Analysis

This document summarizes all experiments involving SIFT-family descriptors: SIFT, RootSIFT, DSPSIFT_V2, RGBSIFT, DSPRGBSIFT_V2, and RGBSIFT_CHANNEL_AVG.

## 1. Baseline Performance by Keypoint Set

### SIFT (128D, Grayscale)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 (unfiltered) | 42.64% | 45.88% | 43.14% | Baseline |
| sift_scale_matched_6px | 63.86% | 65.66% | 59.79% | **+21.22%** |
| keynet_sift_8k_b (intersection) | 55.32% | 57.36% | 53.38% | Cross-detector |

### RootSIFT (128D, Grayscale, Hellinger kernel)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 (native) | 44.79% | 48.02% | 45.43% | +2.15% vs SIFT |
| sift_scale_matched_6px | 65.31% | 67.04% | 61.76% | +1.45% vs SIFT |
| keynet_sift_8k_b (intersection) | 57.82% | 59.88% | 55.85% | +2.50% vs SIFT |

### DSPSIFT_V2 (128D, Domain-Size Pooling)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 (baseline) | 46.57% | 49.68% | 46.25% | +3.93% vs SIFT |
| sift_scale_matched_6px | 65.31% | 66.75% | 60.59% | +1.45% vs SIFT |
| sift_surf_8k_a (intersection) | 62.24% | 64.35% | 58.63% | SIFT-SURF intersection |
| sift_surf_scale_matched_intersection_a | **74.93%** | 75.91% | 70.97% | **Best DSPSIFT** |
| keynet_sift_8k_b (intersection) | 56.97% | 58.91% | 54.95% | Cross-detector |

### RGBSIFT (384D, Per-Channel Color)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 (full) | 43.61% | 46.90% | 44.36% | +0.97% vs SIFT |
| sift_scale_matched_6px | 64.69% | 66.50% | 61.14% | +0.83% vs SIFT |
| sift_surf_8k_a | 61.96% | 64.05% | 58.76% | Intersection |
| sift_surf_scale_matched_intersection_a | **75.03%** | 75.99% | 71.81% | **Best RGBSIFT** |

### RGBSIFT_CHANNEL_AVG (128D, Averaged Channels)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 | 42.81% | 46.05% | 43.43% | +0.17% vs SIFT |
| sift_scale_matched_6px | 64.03% | 65.84% | 60.24% | +0.17% vs SIFT |
| sift_surf_8k_a | 61.22% | 63.30% | 57.87% | Intersection |
| sift_surf_scale_matched_intersection_a | 74.52% | 75.42% | 71.14% | Scale-matched |

### DSPRGBSIFT_V2 (384D, DSP + Color)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 (full) | 47.50% | 50.65% | 47.46% | +4.86% vs SIFT |
| sift_scale_matched_6px | **66.03%** | 67.55% | 61.90% | **Best on scale-controlled** |
| sift_surf_scale_matched_intersection_a | 74.69% | 75.44% | 70.71% | -0.34% vs RGBSIFT |

**Note:** DSP pooling slightly **hurts** on intersection sets (74.69% vs RGBSIFT's 75.03%).

---

## 2. Effect of Keypoint Quality

### Scale Filtering Impact

| Descriptor | Unfiltered | Scale-Filtered | Improvement |
|------------|------------|----------------|-------------|
| SIFT | 42.64% | 63.86% | **+21.22%** |
| RootSIFT | 44.79% | 65.31% | **+20.52%** |
| DSPSIFT_V2 | 46.57% | 65.31% | **+18.74%** |
| RGBSIFT | 43.61% | 64.69% | **+21.08%** |
| DSPRGBSIFT_V2 | 47.50% | 66.03% | **+18.53%** |

**Key Finding:** Scale filtering provides ~20% absolute mAP improvement across all SIFT variants.

### Intersection Set Impact

| Descriptor | Scale-Filtered | Intersection | Improvement |
|------------|----------------|--------------|-------------|
| DSPSIFT_V2 | 65.31% | 74.93% | **+9.62%** |
| RGBSIFT | 64.69% | 75.03% | **+10.34%** |
| DSPRGBSIFT_V2 | 66.03% | 74.69% | **+8.66%** |
| RGBSIFT_CHANNEL_AVG | 64.03% | 74.52% | **+10.49%** |

**Key Finding:** SIFT-SURF intersection provides additional ~9-10% improvement beyond scale filtering.

**Important:** DSP pooling benefit diminishes on intersection sets:
- DSPRGBSIFT_V2 is best on scale-filtered (66.03%) but NOT best on intersection (74.69%)
- Plain RGBSIFT wins on intersection (75.03%)

---

## 3. Fusion Experiments

### SIFT + RGBSIFT Fusion

| Configuration | Method | mAP | vs Best Single |
|---------------|--------|-----|----------------|
| sift_8000 | channel_wise 128D | 42.64% | = SIFT |
| sift_8000 | channel_wise 384D | 42.64% | = SIFT |
| sift_8000 | concatenate 512D | 42.64% | = SIFT |
| sift_8000 | 4-way equal | 42.75% | +0.11% |
| sift_scale_matched_6px | channel_wise 128D | 63.86% | = SIFT |
| sift_scale_matched_6px | 4-way equal | 63.98% | +0.12% |

**Finding:** SIFT + RGBSIFT fusion provides **negligible benefit** (<0.2%).

### DSPSIFT + DSPRGBSIFT Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_8000 | 4-way equal 128D | 46.62% | -0.88% vs DSPRGBSIFT |
| sift_8000 | concatenate 512D | 47.17% | -0.33% vs DSPRGBSIFT |
| sift_scale_matched_6px | 4-way equal 128D | 65.37% | -0.66% vs DSPRGBSIFT |
| sift_scale_matched_6px | concatenate 512D | 65.77% | -0.26% vs DSPRGBSIFT |

**Finding:** DSP fusion **underperforms** the best single descriptor (DSPRGBSIFT_V2 at 66.03%).

### SIFT + SURF Fusion (Cross-Detector)

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_surf_8k (unfiltered) | concatenate | 61.06% | -0.90% vs RGBSIFT |
| sift_surf_scale_matched_intersection | concatenate | 74.37% | -0.66% vs SURF |

**Finding:** Cross-detector SIFT+SURF fusion also **underperforms** single descriptors on intersection.

---

## 4. SIFT + CNN Fusion (Cross-Family)

| Configuration | Method | mAP | Notes |
|---------------|--------|-----|-------|
| SIFT + HardNet (keynet_sift_8k) | concatenate | 55.32% | **FAIL** - same as SIFT alone |
| SIFT + HardNet (keynet_sift_8k) | weighted_avg | 55.32% | **FAIL** |
| SIFT + HardNet (scale_matched_intersection) | concatenate | 72.33% | -10% vs HardNet alone |
| SIFT + HardNet (scale_matched_intersection) | weighted_avg | 72.32% | -10% vs HardNet alone |
| DSPSIFT + HardNet (scale_matched_intersection) | concatenate | 32.41% | **CATASTROPHIC FAIL** |
| RootSIFT + HardNet (scale_matched_intersection) | concatenate | 72.33% | Same as SIFT+HardNet |

**Finding:** Cross-family fusion with CNN descriptors **fails significantly** due to incompatible value distributions.

---

## 5. Tolerance Study (DSPSIFT)

| Tolerance | mAP | HP-V | HP-I | Notes |
|-----------|-----|------|------|-------|
| 1.0px (strict) | 67.41% | 68.57% | 66.30% | Fewer but higher-quality |
| 2.0px (moderate) | 62.10% | 63.50% | 58.88% | |
| 5.0px (relaxed) | 57.31% | 58.89% | 54.84% | |
| 10.0px (very relaxed) | 57.06% | 58.69% | 54.20% | Diminishing returns |

**Finding:** Stricter tolerance (1px) yields better results, but 6px is optimal for scale-matched sets.

---

## 6. Summary Rankings

### Best Configurations (mAP)

1. **RGBSIFT on sift_surf_scale_matched_intersection_a**: 75.03%
2. **DSPSIFT_V2 on sift_surf_scale_matched_intersection_a**: 74.93%
3. **DSPRGBSIFT_V2 on sift_surf_scale_matched_intersection_a**: 74.69%
4. **RGBSIFT_CHANNEL_AVG on sift_surf_scale_matched_intersection_a**: 74.52%
5. **DSPRGBSIFT_V2 on sift_scale_matched_6px**: 66.03%
6. **DSPSIFT_V2 on sift_scale_matched_6px**: 65.31%
7. **RootSIFT on sift_scale_matched_6px**: 65.31%

### Key Conclusions

1. **Scale filtering is the most important optimization** (+20% mAP)
2. **Intersection sets provide additional quality filtering** (+9-10% mAP)
3. **Detector consensus provides independent benefit** (see Scale vs Intersection Study)
4. **DSP pooling helps on unfiltered/filtered keypoints** (+4% / +1-2%) but **hurts on intersection** (-0.3%)
5. **Color (RGBSIFT) provides minimal benefit** on same keypoints (<1%)
6. **Same-family fusion provides no benefit** - best single descriptor wins
7. **Cross-family fusion with CNN fails** due to distribution mismatch

---

## 7. Missing Experiments / Gaps

| Gap | Description | Priority |
|-----|-------------|----------|
| DSPRGBSIFT on intersection | Not tested on SIFT-SURF intersection | Medium |
| RootSIFT on intersection | Only tested on KeyNet intersection, not SIFT-SURF | Low |
| SIFT on sift_surf_intersection | Direct SIFT (not DSP) on intersection | Low |

