-- Migration script: v3.2 → v3.3
-- Adds keypoint retrieval metrics (Bojanic et al. 2020, Eq. 5-6)
--
-- Usage:
--   sqlite3 experiments.db < database/migrate_to_v3_3_retrieval.sql

-- Add retrieval metrics columns to results table
ALTER TABLE results ADD COLUMN keypoint_retrieval_ap REAL DEFAULT -1.0;
ALTER TABLE results ADD COLUMN retrieval_viewpoint_ap REAL DEFAULT -1.0;
ALTER TABLE results ADD COLUMN retrieval_illumination_ap REAL DEFAULT -1.0;
ALTER TABLE results ADD COLUMN retrieval_num_true_positives INTEGER DEFAULT 0;
ALTER TABLE results ADD COLUMN retrieval_num_hard_negatives INTEGER DEFAULT 0;
ALTER TABLE results ADD COLUMN retrieval_num_distractors INTEGER DEFAULT 0;

-- Verify migration
SELECT
    'Migration complete: v3.2 → v3.3' as status,
    COUNT(*) as total_experiments
FROM experiments;

SELECT
    'New columns added to results table' as info,
    COUNT(*) as total_results
FROM results;
