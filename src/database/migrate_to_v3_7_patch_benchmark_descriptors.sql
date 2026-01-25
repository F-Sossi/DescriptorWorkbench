-- Migration to schema v3.7: patch benchmark descriptor cache tables

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

CREATE INDEX IF NOT EXISTS idx_patch_benchmark_descriptor_sets_name
    ON patch_benchmark_descriptor_sets(name);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_descriptors_lookup
    ON patch_benchmark_descriptors(descriptor_set_id, scene_name, difficulty, target_key);
