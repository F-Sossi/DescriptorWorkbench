-- Migration script: Add keypoint verification metrics to results table
-- Schema version: v3.2 (October 2025)
-- Purpose: Add verification task metrics with distractors
-- Based on: "On the Comparison of Classic and Deep Keypoint Detector and Descriptor Methods"
--           (Bojanic et al., 2020) - Section III-B (Keypoint Verification)

ALTER TABLE results ADD COLUMN keypoint_verification_ap REAL DEFAULT -1.0;
ALTER TABLE results ADD COLUMN verification_viewpoint_ap REAL DEFAULT -1.0;
ALTER TABLE results ADD COLUMN verification_illumination_ap REAL DEFAULT -1.0;

-- Create index for verification queries
CREATE INDEX IF NOT EXISTS idx_results_verification ON results(experiment_id, keypoint_verification_ap);
