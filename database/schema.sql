-- Database schema for descriptor research experiments
-- This matches the schema defined in DatabaseManager.cpp

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
    mean_average_precision REAL,
    precision_at_1 REAL,
    precision_at_5 REAL,
    recall_at_1 REAL,
    recall_at_5 REAL,
    total_matches INTEGER,
    total_keypoints INTEGER,
    processing_time_ms REAL,
    timestamp TEXT NOT NULL,
    metadata TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

-- Keypoint sets to manage different keypoint generation strategies
CREATE TABLE IF NOT EXISTS keypoint_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,  -- e.g., "homography_projection_default", "independent_detection_v1"
    generator_type TEXT NOT NULL,  -- e.g., "SIFT", "ORB", "AKAZE"
    generation_method TEXT NOT NULL,  -- "homography_projection" or "independent_detection"  
    max_features INTEGER,
    dataset_path TEXT,
    description TEXT,
    boundary_filter_px INTEGER DEFAULT 40,
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
CREATE INDEX IF NOT EXISTS idx_locked_keypoints_set ON locked_keypoints(keypoint_set_id);
CREATE INDEX IF NOT EXISTS idx_locked_keypoints_scene ON locked_keypoints(keypoint_set_id, scene_name, image_name);

-- Index for efficient descriptor queries by experiment and processing method
CREATE INDEX IF NOT EXISTS idx_descriptors_experiment ON descriptors(experiment_id, processing_method);
CREATE INDEX IF NOT EXISTS idx_descriptors_keypoint ON descriptors(scene_name, image_name, keypoint_x, keypoint_y);
CREATE INDEX IF NOT EXISTS idx_descriptors_method ON descriptors(processing_method, normalization_applied, rooting_applied);