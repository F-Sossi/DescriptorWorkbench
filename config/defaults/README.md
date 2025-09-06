Defaults Overview (Schema v1)

This folder contains minimal, copy‑pasteable templates for common experiment setups using the strict Schema v1 configuration. These are intended as starting points for new experiments.

Templates
- minimal.yaml: Smallest runnable config with SIFT and no pooling.
- sift_baseline.yaml: Baseline SIFT with homography‑based evaluation.
- dsp_gaussian.yaml: SIFT + Domain Size Pooling with gaussian scale weighting.
- stacking_sift_rgbsift.yaml: Stacking SIFT with RGBSIFT as secondary.

Notes
- The config schema is strict: no legacy keys are supported.
- Prefer normalize_after_pooling for most use cases.
- For realistic evaluation, set keypoints.source to independent_detection.

Schema v1 (high level)
- experiment: { name, description, version, author }
- dataset: { type, path, scenes[] }
- keypoints: { generator, max_features, contrast_threshold, edge_threshold, sigma, num_octaves, source }
- descriptors[]: { name, type, pooling, scales[], scale_weights[], scale_weighting, scale_weight_sigma, normalize_before_pooling, normalize_after_pooling, norm_type, use_color, secondary_descriptor, stacking_weight }
- evaluation: matching { method, norm, cross_check, threshold }, validation { method, threshold, min_matches }
- output: { results_path, save_keypoints, save_descriptors, save_matches, save_visualizations }
- database: { enabled, connection }

