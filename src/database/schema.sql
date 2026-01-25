-- Database schema for descriptor research experiments
-- This matches the schema defined in DatabaseManager.cpp
--
-- SCHEMA VERSION: v3.7 (January 2026)
-- MAJOR UPGRADE: Keypoint retrieval metrics (Bojanic et al. 2020, Eq. 5-6)
--
-- Migration notes (v3.7):
-- - Added patch_benchmark_descriptor_sets + patch_benchmark_descriptors tables
-- - Stores descriptor matrices for patch benchmark caching/reuse
--
-- Migration notes (v3.4):
-- - Added patch_benchmark_results table for HPatches patch benchmark runs
-- - Stores mAP breakdowns and metadata for patch-based evaluation
--
-- Migration notes (v3.3):
-- - Added keypoint_retrieval_ap and category variants to results table
-- - Added three-tier label counts (true_positives, hard_negatives, distractors)
-- - Adds retrieval with three-tier labeling evaluation (optional expensive metric)
-- - Based on Bojanic et al. (2020) retrieval task methodology
--
-- Previous migration notes (v3.2):
-- - Added keypoint_verification_ap and category variants to results table
-- - Adds verification with distractors evaluation (optional expensive metric)
-- - Based on Bojanic et al. (2020) verification task methodology
--
-- Previous migration notes (v3.1):
-- - Added viewpoint_map, illumination_map columns to results table
-- - Added _with_zeros variants for conservative evaluation
-- - Enables separate analysis of viewpoint (v_*) vs illumination (i_*) sequences
-- - Based on Bojanic et al. (2020) evaluation methodology
-- - Use database/migrate_to_v3_1_hp_split.sql to upgrade existing databases
--
-- Previous migration notes (v3.0):
-- - Added keypoint_set_id and keypoint_source columns to experiments table
-- - Foreign key relationship: experiments.keypoint_set_id → keypoint_sets.id
-- - Enables tracking which keypoint set was used for each experiment
-- - Use database/migrate_to_v3_keypoint_tracking.sql to upgrade existing databases
--
-- Previous migration notes (v2.0):
-- - true_map_macro/micro are now primary evaluation metrics
-- - legacy_mean_precision preserves backward compatibility
-- - mean_average_precision serves as primary display metric

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor_type TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    pooling_strategy TEXT,
    similarity_threshold REAL,
    max_features INTEGER,
    timestamp TEXT NOT NULL,
    parameters TEXT,
    keypoint_set_id INTEGER DEFAULT NULL,
    keypoint_source TEXT DEFAULT NULL,
    descriptor_dimension INTEGER DEFAULT 0,
    execution_device TEXT DEFAULT 'cpu',
    FOREIGN KEY(keypoint_set_id) REFERENCES keypoint_sets(id)
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    -- PRIMARY IR-style mAP metrics (NEW: v2.0 schema upgrade)
    true_map_macro REAL,                    -- Scene-balanced mAP (primary metric)
    true_map_micro REAL,                    -- Overall mAP weighted by query count
    true_map_macro_with_zeros REAL,         -- Conservative: includes R=0 queries as AP=0
    true_map_micro_with_zeros REAL,         -- Conservative: includes R=0 queries as AP=0
    image_retrieval_map REAL DEFAULT -1,    -- Image-level retrieval MAP (optional; -1 when disabled)
    -- Category-specific metrics (v3.1): Viewpoint vs Illumination
    viewpoint_map REAL DEFAULT 0.0,         -- mAP for v_* sequences only (geometric changes)
    illumination_map REAL DEFAULT 0.0,      -- mAP for i_* sequences only (photometric changes)
    viewpoint_map_with_zeros REAL DEFAULT 0.0,     -- Conservative: includes R=0 queries
    illumination_map_with_zeros REAL DEFAULT 0.0,  -- Conservative: includes R=0 queries
    -- Keypoint verification metrics (v3.2): Bojanic et al. (2020) verification task
    keypoint_verification_ap REAL DEFAULT -1.0,    -- Verification AP with distractors (-1 when disabled)
    verification_viewpoint_ap REAL DEFAULT -1.0,   -- Verification AP for viewpoint scenes only
    verification_illumination_ap REAL DEFAULT -1.0, -- Verification AP for illumination scenes only
    -- Keypoint retrieval metrics (v3.3): Bojanic et al. (2020) retrieval task (Eq. 5-6)
    keypoint_retrieval_ap REAL DEFAULT -1.0,       -- Retrieval AP with three-tier labels (-1 when disabled)
    retrieval_viewpoint_ap REAL DEFAULT -1.0,      -- Retrieval AP for viewpoint scenes only
    retrieval_illumination_ap REAL DEFAULT -1.0,   -- Retrieval AP for illumination scenes only
    retrieval_num_true_positives INTEGER DEFAULT 0, -- Count of y=+1 labels (in-sequence AND closest)
    retrieval_num_hard_negatives INTEGER DEFAULT 0, -- Count of y=0 labels (in-sequence but NOT closest)
    retrieval_num_distractors INTEGER DEFAULT 0,   -- Count of y=-1 labels (out-of-sequence)
    -- Legacy/compatibility metrics
    mean_average_precision REAL,            -- Primary display metric (uses true_map_macro when available)
    legacy_mean_precision REAL,             -- Original arithmetic mean for backward compatibility
    -- Standard retrieval metrics
    precision_at_1 REAL,
    precision_at_5 REAL,
    recall_at_1 REAL,
    recall_at_5 REAL,
    -- Experiment metadata
    total_matches INTEGER,
    total_keypoints INTEGER,
    processing_time_ms REAL,
    timestamp TEXT NOT NULL,
    descriptor_time_cpu_ms REAL,
    descriptor_time_gpu_ms REAL,
    match_time_cpu_ms REAL,
    match_time_gpu_ms REAL,
    total_pipeline_cpu_ms REAL,
    total_pipeline_gpu_ms REAL,
    metadata TEXT,                          -- Additional metrics and profiling data
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

-- Patch benchmark results (HPatches pre-extracted patches)
CREATE TABLE IF NOT EXISTS patch_benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    descriptor_name TEXT NOT NULL,
    descriptor_dimension INTEGER DEFAULT 0,
    map_overall REAL,
    accuracy_overall REAL,
    map_easy REAL,
    map_hard REAL,
    map_tough REAL,
    map_illumination REAL,
    map_viewpoint REAL,
    map_illumination_easy REAL,
    map_illumination_hard REAL,
    map_viewpoint_easy REAL,
    map_viewpoint_hard REAL,
    verification_same_overall REAL,
    verification_same_easy REAL,
    verification_same_hard REAL,
    verification_same_tough REAL,
    verification_same_illumination REAL,
    verification_same_viewpoint REAL,
    verification_same_illumination_easy REAL,
    verification_same_illumination_hard REAL,
    verification_same_viewpoint_easy REAL,
    verification_same_viewpoint_hard REAL,
    verification_diff_overall REAL,
    verification_diff_easy REAL,
    verification_diff_hard REAL,
    verification_diff_tough REAL,
    verification_diff_illumination REAL,
    verification_diff_viewpoint REAL,
    verification_diff_illumination_easy REAL,
    verification_diff_illumination_hard REAL,
    verification_diff_viewpoint_easy REAL,
    verification_diff_viewpoint_hard REAL,
    retrieval_overall REAL,
    retrieval_easy REAL,
    retrieval_hard REAL,
    retrieval_tough REAL,
    retrieval_illumination REAL,
    retrieval_viewpoint REAL,
    retrieval_illumination_easy REAL,
    retrieval_illumination_hard REAL,
    retrieval_viewpoint_easy REAL,
    retrieval_viewpoint_hard REAL,
    verification_negatives_per_query INTEGER DEFAULT 0,
    retrieval_negatives_per_query INTEGER DEFAULT 0,
    verification_negative_source TEXT DEFAULT 'both',
    retrieval_negative_source TEXT DEFAULT 'diff_seq',
    num_scenes INTEGER,
    num_patches INTEGER,
    processing_time_ms REAL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

-- Patch benchmark task sets (verification/retrieval task definitions)
CREATE TABLE IF NOT EXISTS patch_benchmark_task_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patch_benchmark_descriptor_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    experiment_id INTEGER,
    descriptor_name TEXT NOT NULL,
    descriptor_dimension INTEGER DEFAULT 0,
    patches_dir TEXT NOT NULL,
    color INTEGER DEFAULT 0,
    patch_keypoint_size REAL DEFAULT 0.0,
    params_hash TEXT,
    params_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS patch_benchmark_descriptors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor_set_id INTEGER NOT NULL,
    scene_name TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    target_key TEXT NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    cv_type INTEGER NOT NULL,
    data BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(descriptor_set_id) REFERENCES patch_benchmark_descriptor_sets(id),
    UNIQUE(descriptor_set_id, scene_name, difficulty, target_key)
);

CREATE TABLE IF NOT EXISTS patch_benchmark_verification_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_set_id INTEGER NOT NULL,
    split TEXT NOT NULL,      -- full, illum, view, a, b, c
    neg_type TEXT NOT NULL,   -- pos, inter, intra
    s1 TEXT NOT NULL,
    t1 INTEGER NOT NULL,
    idx1 INTEGER NOT NULL,
    s2 TEXT NOT NULL,
    t2 INTEGER NOT NULL,
    idx2 INTEGER NOT NULL,
    FOREIGN KEY(task_set_id) REFERENCES patch_benchmark_task_sets(id)
);

CREATE TABLE IF NOT EXISTS patch_benchmark_retrieval_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_set_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    s TEXT NOT NULL,
    idx INTEGER NOT NULL,
    FOREIGN KEY(task_set_id) REFERENCES patch_benchmark_task_sets(id)
);

CREATE TABLE IF NOT EXISTS patch_benchmark_retrieval_distractors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_set_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    s TEXT NOT NULL,
    idx INTEGER NOT NULL,
    FOREIGN KEY(task_set_id) REFERENCES patch_benchmark_task_sets(id)
);

-- Keypoint sets to manage different keypoint generation strategies
CREATE TABLE IF NOT EXISTS keypoint_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,  -- e.g., "homography_projection_default", "independent_detection_v1"
    generator_type TEXT NOT NULL,  -- e.g., "SIFT", "ORB", "AKAZE", "Harris"
    generation_method TEXT NOT NULL,  -- "homography_projection", "independent_detection", "non_overlapping_detection"
    max_features INTEGER,
    dataset_path TEXT,
    description TEXT,
    boundary_filter_px INTEGER DEFAULT 40,
    -- Non-overlapping constraint support (NEW: for CNN optimization)
    overlap_filtering BOOLEAN DEFAULT FALSE,  -- Whether non-overlapping constraint was applied
    min_distance REAL DEFAULT 0.0,          -- Minimum distance in pixels (0 = no constraint)
    -- Intersection set support (NEW: for paired detector keypoint sets)
    source_set_a_id INTEGER DEFAULT NULL,   -- Reference to first source set for intersection
    source_set_b_id INTEGER DEFAULT NULL,   -- Reference to second source set for intersection
    tolerance_px REAL DEFAULT NULL,         -- Spatial tolerance in pixels for intersection matching
    intersection_method TEXT DEFAULT NULL,  -- Method used for intersection (e.g., "spatial_nearest")
    detection_time_cpu_ms REAL,
    detection_time_gpu_ms REAL,
    total_generation_cpu_ms REAL,
    total_generation_gpu_ms REAL,
    intersection_time_ms REAL,
    avg_keypoints_per_image REAL,
    total_keypoints INTEGER,
    source_a_keypoints INTEGER,
    source_b_keypoints INTEGER,
    keypoint_reduction_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_set_a_id) REFERENCES keypoint_sets(id),
    FOREIGN KEY(source_set_b_id) REFERENCES keypoint_sets(id)
);

-- Locked-in keypoints storage with keypoint set reference
CREATE TABLE IF NOT EXISTS locked_keypoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keypoint_set_id INTEGER NOT NULL DEFAULT 1,  -- Foreign key to keypoint_sets
    scene_name TEXT NOT NULL,
    image_name TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    size REAL NOT NULL,
    angle REAL NOT NULL,
    response REAL NOT NULL,
    octave INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    valid_bounds BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(keypoint_set_id) REFERENCES keypoint_sets(id),
    UNIQUE(keypoint_set_id, scene_name, image_name, x, y, size, angle, response, octave)
);

-- Descriptor storage for research analysis
CREATE TABLE IF NOT EXISTS descriptors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    scene_name TEXT NOT NULL,
    image_name TEXT NOT NULL,
    keypoint_x REAL NOT NULL,
    keypoint_y REAL NOT NULL,
    descriptor_vector BLOB NOT NULL,  -- Binary storage of cv::Mat descriptor
    descriptor_dimension INTEGER NOT NULL,  -- e.g., 128 for SIFT
    processing_method TEXT,  -- e.g., "SIFT-BW-None-NoNorm-NoRoot-L2"
    normalization_applied TEXT,  -- e.g., "NoNorm", "L2", "L1"
    rooting_applied TEXT,  -- e.g., "NoRoot", "RBef", "RAft"
    pooling_applied TEXT,  -- e.g., "None", "Dom", "Stack"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id),
    -- Link to specific keypoint for traceability
    UNIQUE(experiment_id, scene_name, image_name, keypoint_x, keypoint_y)
);

-- Indexes for efficient keypoint set queries  
CREATE INDEX IF NOT EXISTS idx_keypoint_sets_method ON keypoint_sets(generation_method);
CREATE INDEX IF NOT EXISTS idx_keypoint_sets_generator ON keypoint_sets(generator_type);
CREATE INDEX IF NOT EXISTS idx_keypoint_sets_overlap ON keypoint_sets(overlap_filtering);
CREATE INDEX IF NOT EXISTS idx_locked_keypoints_set ON locked_keypoints(keypoint_set_id);
CREATE INDEX IF NOT EXISTS idx_locked_keypoints_scene ON locked_keypoints(keypoint_set_id, scene_name, image_name);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_experiment ON patch_benchmark_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_task_sets_name ON patch_benchmark_task_sets(name);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_descriptor_sets_name ON patch_benchmark_descriptor_sets(name);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_descriptors_lookup ON patch_benchmark_descriptors(descriptor_set_id, scene_name, difficulty, target_key);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_verif_pairs ON patch_benchmark_verification_pairs(task_set_id, split, neg_type);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_retr_queries ON patch_benchmark_retrieval_queries(task_set_id, split);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_retr_distractors ON patch_benchmark_retrieval_distractors(task_set_id, split);

-- Matches storage for research analysis
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    scene_name TEXT NOT NULL,
    query_image TEXT NOT NULL,  -- e.g., "1.ppm"
    train_image TEXT NOT NULL,  -- e.g., "2.ppm"
    query_keypoint_x REAL NOT NULL,
    query_keypoint_y REAL NOT NULL,
    train_keypoint_x REAL NOT NULL,
    train_keypoint_y REAL NOT NULL,
    distance REAL NOT NULL,
    match_confidence REAL,
    is_correct_match BOOLEAN,  -- Based on homography validation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

-- Visualizations storage for debugging and analysis
CREATE TABLE IF NOT EXISTS visualizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    scene_name TEXT NOT NULL,
    visualization_type TEXT NOT NULL,  -- "keypoints", "matches", "homography"
    image_pair TEXT,  -- e.g., "1_2" for 1.ppm -> 2.ppm
    image_data BLOB NOT NULL,  -- PNG/JPEG encoded visualization
    image_format TEXT DEFAULT 'PNG',
    metadata TEXT,  -- JSON metadata about visualization
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

-- Index for efficient descriptor queries by experiment and processing method
CREATE INDEX IF NOT EXISTS idx_descriptors_experiment ON descriptors(experiment_id, processing_method);
CREATE INDEX IF NOT EXISTS idx_descriptors_keypoint ON descriptors(scene_name, image_name, keypoint_x, keypoint_y);
CREATE INDEX IF NOT EXISTS idx_descriptors_method ON descriptors(processing_method, normalization_applied, rooting_applied);

-- Indexes for efficient match queries
CREATE INDEX IF NOT EXISTS idx_matches_experiment ON matches(experiment_id, scene_name);
CREATE INDEX IF NOT EXISTS idx_matches_correctness ON matches(experiment_id, is_correct_match);
CREATE INDEX IF NOT EXISTS idx_matches_image_pair ON matches(experiment_id, scene_name, query_image, train_image);

-- Indexes for efficient visualization queries
CREATE INDEX IF NOT EXISTS idx_visualizations_experiment ON visualizations(experiment_id, scene_name);
CREATE INDEX IF NOT EXISTS idx_visualizations_type ON visualizations(visualization_type);
CREATE INDEX IF NOT EXISTS idx_visualizations_pair ON visualizations(experiment_id, scene_name, image_pair);
