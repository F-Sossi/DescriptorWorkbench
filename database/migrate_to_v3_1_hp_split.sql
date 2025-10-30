-- Migration script: Add HP-V vs HP-I metrics to results table
-- Schema version: v3.1 (October 2025)
-- Purpose: Add viewpoint vs illumination category-specific metrics
--
-- Based on: "On the Comparison of Classic and Deep Keypoint Detector and Descriptor Methods"
--           (Bojanic et al., 2020)
--
-- Usage: sqlite3 experiments.db < migrate_to_v3_1_hp_split.sql

-- Add category-specific mAP columns to results table
ALTER TABLE results ADD COLUMN viewpoint_map REAL DEFAULT 0.0;
ALTER TABLE results ADD COLUMN illumination_map REAL DEFAULT 0.0;
ALTER TABLE results ADD COLUMN viewpoint_map_with_zeros REAL DEFAULT 0.0;
ALTER TABLE results ADD COLUMN illumination_map_with_zeros REAL DEFAULT 0.0;

-- Add index for category-specific queries
CREATE INDEX IF NOT EXISTS idx_results_hp_v_vs_i ON results(experiment_id, viewpoint_map, illumination_map);

-- Verification query (run after migration)
-- SELECT COUNT(*) FROM pragma_table_info('results') WHERE name IN ('viewpoint_map', 'illumination_map');
-- Expected result: 2 (if migration successful)

SELECT 'Migration v3.1 complete: HP-V vs HP-I metrics added to results table' AS status;
