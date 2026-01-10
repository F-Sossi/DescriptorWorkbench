-- Database Migration: Promote True MAP Metrics to First-Class Columns
-- This migration adds proper IR-style mAP columns and populates them from metadata

-- Add new columns for true MAP metrics
ALTER TABLE results ADD COLUMN true_map_macro REAL;
ALTER TABLE results ADD COLUMN true_map_micro REAL;  
ALTER TABLE results ADD COLUMN true_map_macro_with_zeros REAL;
ALTER TABLE results ADD COLUMN true_map_micro_with_zeros REAL;
ALTER TABLE results ADD COLUMN legacy_mean_precision REAL;

-- Copy existing mean_average_precision to legacy column for backward compatibility
UPDATE results SET legacy_mean_precision = mean_average_precision;

-- Extract true MAP values from metadata and populate new columns
-- This uses SQLite's regex-like functionality via REPLACE and SUBSTR

-- Note: This would need to be done programmatically since SQLite doesn't have regex
-- The actual migration will be handled by a Python script

-- For now, create a view that demonstrates the desired structure
CREATE VIEW results_enhanced AS
SELECT 
    id,
    experiment_id,
    -- New primary metrics (to be populated by migration script)
    true_map_macro,
    true_map_micro,
    true_map_macro_with_zeros,
    true_map_micro_with_zeros,
    -- Existing precision metrics
    precision_at_1,
    precision_at_5,
    recall_at_1,
    recall_at_5,
    -- Processing info
    total_matches,
    total_keypoints,
    processing_time_ms,
    -- Legacy compatibility
    legacy_mean_precision,
    mean_average_precision, -- Keep existing for transition period
    -- Metadata
    timestamp,
    metadata
FROM results;