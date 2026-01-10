#!/usr/bin/env python3
"""
Database Migration Script: Add Overlap Tracking to Keypoint Sets

This script migrates the existing database to support non-overlapping keypoint
generation by adding overlap_filtering and min_distance columns to keypoint_sets.
"""

import sqlite3
import sys
import os
import shutil
from pathlib import Path

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns

def migrate_database(db_path):
    """Perform the keypoint schema migration"""
    print(f"Migrating database: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Step 1: Check if keypoint_sets table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='keypoint_sets'
        """)
        if not cursor.fetchone():
            print("⚠ keypoint_sets table not found. Creating with new schema...")
            cursor.execute("""
                CREATE TABLE keypoint_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    generator_type TEXT NOT NULL,
                    generation_method TEXT NOT NULL,
                    max_features INTEGER,
                    dataset_path TEXT,
                    description TEXT,
                    boundary_filter_px INTEGER DEFAULT 40,
                    overlap_filtering BOOLEAN DEFAULT FALSE,
                    min_distance REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✓ Created keypoint_sets table with new schema")
        else:
            # Step 2: Add new columns if they don't exist
            new_columns = [
                ('overlap_filtering', 'BOOLEAN DEFAULT FALSE'),
                ('min_distance', 'REAL DEFAULT 0.0')
            ]
            
            for column_name, column_def in new_columns:
                if not check_column_exists(cursor, 'keypoint_sets', column_name):
                    try:
                        cursor.execute(f"ALTER TABLE keypoint_sets ADD COLUMN {column_name} {column_def}")
                        print(f"✓ Added column: {column_name}")
                    except sqlite3.OperationalError as e:
                        print(f"⚠ Failed to add column {column_name}: {e}")
                else:
                    print(f"⚠ Column {column_name} already exists, skipping")
        
        # Step 3: Create new indexes
        new_indexes = [
            ('idx_keypoint_sets_generator', 'keypoint_sets(generator_type)'),
            ('idx_keypoint_sets_overlap', 'keypoint_sets(overlap_filtering)')
        ]
        
        for index_name, index_def in new_indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {index_def}")
                print(f"✓ Created index: {index_name}")
            except sqlite3.OperationalError as e:
                print(f"⚠ Index creation warning for {index_name}: {e}")
        
        # Step 4: Update existing records to set overlap_filtering based on generation_method
        cursor.execute("""
            UPDATE keypoint_sets 
            SET overlap_filtering = TRUE,
                min_distance = 32.0
            WHERE generation_method = 'non_overlapping_detection'
              AND overlap_filtering = FALSE
        """)
        non_overlapping_updated = cursor.rowcount
        print(f"✓ Updated {non_overlapping_updated} records to mark non-overlapping sets")
        
        # Step 5: Show migration summary
        cursor.execute("""
            SELECT 
                COUNT(*) as total_sets,
                COUNT(CASE WHEN overlap_filtering = TRUE THEN 1 END) as non_overlapping_sets,
                COUNT(CASE WHEN generation_method = 'homography_projection' THEN 1 END) as homography_sets,
                COUNT(CASE WHEN generation_method = 'independent_detection' THEN 1 END) as independent_sets,
                COUNT(CASE WHEN generation_method = 'non_overlapping_detection' THEN 1 END) as overlap_filtered_sets,
                COUNT(DISTINCT generator_type) as detector_types
            FROM keypoint_sets
        """)
        summary = cursor.fetchone()
        
        print(f"\n📊 Migration Summary:")
        print(f"   Total keypoint sets:        {summary[0]}")
        print(f"   Non-overlapping sets:       {summary[1]}")
        print(f"   Homography projection sets: {summary[2]}")
        print(f"   Independent detection sets: {summary[3]}")
        print(f"   Overlap filtered sets:      {summary[4]}")
        print(f"   Unique detector types:      {summary[5]}")
        
        # Step 6: Show detector type breakdown
        cursor.execute("""
            SELECT 
                generator_type,
                generation_method,
                COUNT(*) as count,
                COUNT(CASE WHEN overlap_filtering = TRUE THEN 1 END) as non_overlapping_count
            FROM keypoint_sets
            GROUP BY generator_type, generation_method
            ORDER BY generator_type, generation_method
        """)
        detector_breakdown = cursor.fetchall()
        
        if detector_breakdown:
            print(f"\n📋 Detector Type Breakdown:")
            print("   Detector | Method                | Total | Non-Overlap")
            print("   ---------|----------------------|-------|------------")
            for row in detector_breakdown:
                detector, method, total, non_overlap = row
                print(f"   {detector:<8} | {method:<20} | {total:<5} | {non_overlap}")
        
        # Commit changes
        conn.commit()
        print(f"\n✅ Migration completed successfully!")
        
        # Show sample of migrated data
        print(f"\n📋 Sample of keypoint sets:")
        cursor.execute("""
            SELECT 
                id,
                name,
                generator_type,
                generation_method,
                overlap_filtering,
                ROUND(min_distance, 1) as min_dist
            FROM keypoint_sets 
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        sample_rows = cursor.fetchall()
        if sample_rows:
            print("   ID | Name                     | Detector | Method               | Overlap | Min Dist")
            print("   ---|--------------------------|----------|----------------------|---------|----------")
            for row in sample_rows:
                overlap_str = "Yes" if row[4] else "No"
                print(f"   {row[0]:<2} | {row[1][:24]:<24} | {row[2]:<8} | {row[3]:<20} | {overlap_str:<7} | {row[5]}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise e
    finally:
        conn.close()

def main():
    """Main migration function"""
    # Determine database path
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Try default locations
        possible_paths = [
            "build/experiments.db",
            "../build/experiments.db", 
            "experiments.db"
        ]
        db_path = None
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if not db_path:
            print("❌ Could not find database file. Please specify path:")
            print(f"   Usage: {sys.argv[0]} <path_to_experiments.db>")
            sys.exit(1)
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        sys.exit(1)
    
    # Create backup
    backup_path = f"{db_path}.keypoint_schema_backup"
    print(f"🔄 Creating backup: {backup_path}")
    
    try:
        shutil.copy2(db_path, backup_path)
        print(f"✓ Backup created successfully")
    except Exception as e:
        print(f"⚠ Could not create backup: {e}")
        response = input("Continue without backup? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Perform migration
    try:
        migrate_database(db_path)
        print(f"\n🎉 Keypoint schema migration completed!")
        print(f"   • Added overlap_filtering and min_distance columns")
        print(f"   • Created new indexes for efficient queries")
        print(f"   • Updated existing non-overlapping sets")
        print(f"   • Backup saved to: {backup_path}")
        print(f"\n💡 New CLI commands available:")
        print(f"   • ./keypoint_manager generate-detector <folder> <detector> [name]")
        print(f"   • ./keypoint_manager generate-non-overlapping <folder> <detector> <min_distance> [name]")
        print(f"   • ./keypoint_manager list-detectors")
    except Exception as e:
        print(f"\n💥 Migration failed: {e}")
        if os.path.exists(backup_path):
            print(f"   • Restore from backup: cp {backup_path} {db_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()