-- Migration: add patch_benchmark_results table (schema v3.4)
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
    num_scenes INTEGER,
    num_patches INTEGER,
    processing_time_ms REAL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_patch_benchmark_experiment ON patch_benchmark_results(experiment_id);
