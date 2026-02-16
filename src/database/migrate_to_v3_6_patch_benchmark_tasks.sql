-- Migration to schema v3.6: Add patch benchmark task tables

CREATE TABLE IF NOT EXISTS patch_benchmark_task_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patch_benchmark_verification_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_set_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    neg_type TEXT NOT NULL,
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

CREATE INDEX IF NOT EXISTS idx_patch_benchmark_task_sets_name
    ON patch_benchmark_task_sets(name);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_verif_pairs
    ON patch_benchmark_verification_pairs(task_set_id, split, neg_type);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_retr_queries
    ON patch_benchmark_retrieval_queries(task_set_id, split);
CREATE INDEX IF NOT EXISTS idx_patch_benchmark_retr_distractors
    ON patch_benchmark_retrieval_distractors(task_set_id, split);

PRAGMA user_version = 6;
