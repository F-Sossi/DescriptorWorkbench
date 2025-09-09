#!/usr/bin/env python3
"""
Quick demonstration of the descriptor analysis capabilities
Shows what insights can be extracted from the database
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🔬 DescriptorWorkbench Quick Analysis Demo")
    print("=" * 50)
    
    # Connect to database
    db_path = "build/experiments.db"
    conn = sqlite3.connect(db_path)
    
    # Load experiment data
    query = """
    SELECT 
        e.id,
        e.descriptor_type,
        e.pooling_strategy,
        r.mean_average_precision,
        r.precision_at_1,
        r.precision_at_5,
        r.processing_time_ms
    FROM experiments e 
    JOIN results r ON e.id = r.experiment_id
    ORDER BY r.mean_average_precision DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\n📊 Analysis Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Descriptor types: {df['descriptor_type'].nunique()}")
    print(f"Pooling strategies: {df['pooling_strategy'].nunique()}")
    
    print(f"\n🏆 Top 3 Performing Configurations:")
    for i, row in df.head(3).iterrows():
        print(f"{i+1}. {row['descriptor_type']} ({row['pooling_strategy']})")
        print(f"   MAP: {row['mean_average_precision']:.3f} | P@1: {row['precision_at_1']:.3f}")
    
    # Pooling strategy analysis
    print(f"\n🔄 Pooling Strategy Effects:")
    pooling_stats = df.groupby('pooling_strategy')['mean_average_precision'].agg(['mean', 'count'])
    for pooling, stats in pooling_stats.iterrows():
        print(f"• {pooling}: {stats['mean']:.3f} MAP (n={stats['count']})")
    
    # Find best improvement
    baseline_map = df[df['pooling_strategy'] == 'none']['mean_average_precision'].iloc[0]
    best_map = df['mean_average_precision'].max()
    improvement = (best_map - baseline_map) / baseline_map * 100
    
    print(f"\n📈 Performance Insights:")
    print(f"Baseline MAP: {baseline_map:.3f}")
    print(f"Best MAP: {best_map:.3f}")
    print(f"Maximum improvement: +{improvement:.1f}%")
    
    # Processing time analysis  
    avg_time = df['processing_time_ms'].mean() / 1000
    print(f"Average processing time: {avg_time:.1f} seconds")
    
    print(f"\n✨ Ready for detailed analysis in Jupyter notebooks!")
    print(f"Run: ./start_analysis.sh")
    

if __name__ == "__main__":
    main()