#!/usr/bin/env python3
"""
Database Migration Script: Promote True MAP Metrics to Primary Columns

This script migrates the existing database to use true IR-style mAP metrics
as primary columns instead of buried in metadata strings.
"""

import sqlite3
import re
import sys
import os
from pathlib import Path

def extract_metric_from_metadata(metadata_str, metric_name):
    """Extract a numeric metric value from metadata string"""
    if not metadata_str:
        return None
    
    pattern = rf'{metric_name}=([0-9.]+)'
    match = re.search(pattern, metadata_str)
    return float(match.group(1)) if match else None

def migrate_database(db_path):
    """Perform the database migration"""
    print(f"Migrating database: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Step 1: Add new columns (skip if already exist)
        new_columns = [
            'true_map_macro REAL',
            'true_map_micro REAL',
            'true_map_macro_with_zeros REAL', 
            'true_map_micro_with_zeros REAL',
            'legacy_mean_precision REAL'
        ]
        
        for column_def in new_columns:
            column_name = column_def.split()[0]
            try:
                cursor.execute(f"ALTER TABLE results ADD COLUMN {column_def}")
                print(f"✓ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"⚠ Column {column_name} already exists, skipping")
                else:
                    raise e
        
        # Step 2: Copy existing mean_average_precision to legacy column
        cursor.execute("""
            UPDATE results 
            SET legacy_mean_precision = mean_average_precision 
            WHERE legacy_mean_precision IS NULL
        """)
        legacy_updated = cursor.rowcount
        print(f"✓ Copied {legacy_updated} legacy mean precision values")
        
        # Step 3: Extract and populate true MAP metrics from metadata
        cursor.execute("SELECT id, metadata FROM results WHERE metadata IS NOT NULL")
        rows = cursor.fetchall()
        
        updates_made = 0
        for row_id, metadata in rows:
            # Extract metrics from metadata
            true_map_macro = extract_metric_from_metadata(metadata, 'true_map_macro_by_scene')
            true_map_micro = extract_metric_from_metadata(metadata, 'true_map_micro')
            true_map_macro_zeros = extract_metric_from_metadata(metadata, 'true_map_macro_with_zeros')
            true_map_micro_zeros = extract_metric_from_metadata(metadata, 'true_map_micro_with_zeros')
            
            # Update row with extracted values
            cursor.execute("""
                UPDATE results SET 
                    true_map_macro = ?,
                    true_map_micro = ?,
                    true_map_macro_with_zeros = ?,
                    true_map_micro_with_zeros = ?
                WHERE id = ?
            """, (true_map_macro, true_map_micro, true_map_macro_zeros, true_map_micro_zeros, row_id))
            
            if any(val is not None for val in [true_map_macro, true_map_micro, true_map_macro_zeros, true_map_micro_zeros]):
                updates_made += 1
        
        print(f"✓ Extracted and populated true MAP metrics for {updates_made} records")
        
        # Step 4: Update mean_average_precision to use true_map_macro as primary
        cursor.execute("""
            UPDATE results 
            SET mean_average_precision = COALESCE(true_map_macro, legacy_mean_precision)
        """)
        primary_updated = cursor.rowcount
        print(f"✓ Updated {primary_updated} records to use true MAP as primary metric")
        
        # Step 5: Show migration summary
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(true_map_macro) as macro_map_populated,
                COUNT(true_map_micro) as micro_map_populated,
                COUNT(legacy_mean_precision) as legacy_preserved,
                AVG(true_map_macro) as avg_macro_map,
                AVG(true_map_micro) as avg_micro_map,
                AVG(legacy_mean_precision) as avg_legacy_precision
            FROM results
        """)
        summary = cursor.fetchone()
        
        print(f"\n📊 Migration Summary:")
        print(f"   Total records:           {summary[0]}")
        print(f"   Macro MAP populated:     {summary[1]}")
        print(f"   Micro MAP populated:     {summary[2]}")
        print(f"   Legacy values preserved: {summary[3]}")
        print(f"   Avg Macro MAP:           {summary[4]:.4f}" if summary[4] else "   Avg Macro MAP:           None")
        print(f"   Avg Micro MAP:           {summary[5]:.4f}" if summary[5] else "   Avg Micro MAP:           None")
        print(f"   Avg Legacy Precision:    {summary[6]:.4f}" if summary[6] else "   Avg Legacy Precision:    None")
        
        # Commit changes
        conn.commit()
        print(f"\n✅ Migration completed successfully!")
        
        # Show sample of migrated data
        print(f"\n📋 Sample of migrated data:")
        cursor.execute("""
            SELECT 
                id,
                ROUND(true_map_macro, 4) as macro_map,
                ROUND(true_map_micro, 4) as micro_map, 
                ROUND(legacy_mean_precision, 4) as legacy,
                ROUND(mean_average_precision, 4) as primary_map
            FROM results 
            WHERE true_map_macro IS NOT NULL 
            ORDER BY true_map_macro DESC 
            LIMIT 5
        """)
        
        sample_rows = cursor.fetchall()
        print("   ID | Macro MAP | Micro MAP | Legacy   | Primary MAP")
        print("   ---|-----------|-----------|----------|------------")
        for row in sample_rows:
            print(f"   {row[0]:<2} | {row[1]:<9} | {row[2]:<9} | {row[3]:<8} | {row[4]}")
        
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
    backup_path = f"{db_path}.backup"
    print(f"🔄 Creating backup: {backup_path}")
    
    try:
        import shutil
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
        print(f"\n🎉 Database migration completed!")
        print(f"   • True MAP metrics are now primary columns")
        print(f"   • Legacy metrics preserved for backward compatibility")
        print(f"   • Backup saved to: {backup_path}")
    except Exception as e:
        print(f"\n💥 Migration failed: {e}")
        if os.path.exists(backup_path):
            print(f"   • Restore from backup: cp {backup_path} {db_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()