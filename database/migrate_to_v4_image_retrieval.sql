-- Migration: add image_retrieval_map column to results table (schema v3.1)
ALTER TABLE results ADD COLUMN image_retrieval_map REAL DEFAULT -1;
