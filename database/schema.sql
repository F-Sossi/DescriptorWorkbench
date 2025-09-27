-- Database schema for descriptor research experiments
-- This matches the schema defined in DatabaseManager.cpp
--
-- SCHEMA VERSION: v2.0 (September 2025)
-- MAJOR UPGRADE: True IR-style mAP metrics promoted to first-class columns
-- 
-- Migration notes:
-- - true_map_macro/micro are now primary evaluation metrics  
-- - legacy_mean_precision preserves backward compatibility
-- - mean_average_precision serves as primary display metric
-- - Use migrate_database.py to upgrade existing databases

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor_type TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    pooling_strategy TEXT,
    similarity_threshold REAL,
    max_features INTEGER,
    timestamp TEXT NOT NULL,
    parameters TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    -- PRIMARY IR-style mAP metrics (NEW: v2.0 schema upgrade)
    true_map_macro REAL,                    -- Scene-balanced mAP (primary metric)
    true_map_micro REAL,                    -- Overall mAP weighted by query count
    true_map_macro_with_zeros REAL,         -- Conservative: includes R=0 queries as AP=0
    true_map_micro_with_zeros REAL,         -- Conservative: includes R=0 queries as AP=0  
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
    metadata TEXT,                          -- Additional metrics and profiling data
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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