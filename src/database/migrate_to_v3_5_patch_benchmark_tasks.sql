-- Migration: add verification/retrieval metrics to patch_benchmark_results (schema v3.5)
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_overall REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_hard REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_tough REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_illumination REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_viewpoint REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_illumination_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_illumination_hard REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_viewpoint_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_same_viewpoint_hard REAL;

ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_overall REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_hard REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_tough REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_illumination REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_viewpoint REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_illumination_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_illumination_hard REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_viewpoint_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_diff_viewpoint_hard REAL;

ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_overall REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_hard REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_tough REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_illumination REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_viewpoint REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_illumination_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_illumination_hard REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_viewpoint_easy REAL;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_viewpoint_hard REAL;

ALTER TABLE patch_benchmark_results ADD COLUMN verification_negatives_per_query INTEGER DEFAULT 0;
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_negatives_per_query INTEGER DEFAULT 0;
ALTER TABLE patch_benchmark_results ADD COLUMN verification_negative_source TEXT DEFAULT 'both';
ALTER TABLE patch_benchmark_results ADD COLUMN retrieval_negative_source TEXT DEFAULT 'diff_seq';
