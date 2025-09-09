#!/usr/bin/env python3
"""
Test script to verify the notebook functionality works correctly
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# Database connection
DB_PATH = "build/experiments.db"

def load_experiment_data():
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        e.id as experiment_id,
        e.descriptor_type,
        e.pooling_strategy,
        e.dataset_name,
        e.similarity_threshold,
        e.max_features,
        r.mean_average_precision,
        r.precision_at_1,
        r.precision_at_5,
        r.recall_at_1,
        r.recall_at_5,
        r.total_matches,
        r.total_keypoints,
        r.processing_time_ms
    FROM experiments e 
    JOIN results r ON e.id = r.experiment_id
    ORDER BY e.id DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def extract_descriptor_features(df):
    """Extract features from descriptor names for analysis"""
    df = df.copy()
    
    # Extract color information
    df['uses_color'] = df['descriptor_type'].str.contains('rgb', case=False)
    
    # Extract base descriptor type
    df['base_descriptor'] = df['descriptor_type'].str.extract(r'(sift|rgbsift|honc|vgg|dnn)', expand=False).str.upper()
    
    # Extract normalization information from descriptor name
    df['normalization'] = 'L2'  # Default
    df.loc[df['descriptor_type'].str.contains('l1', case=False), 'normalization'] = 'L1'
    
    # Clean pooling strategy names
    pooling_map = {
        'none': 'None',
        'domain_size_pooling': 'DSP',
        'stacking': 'Stacking'
    }
    df['pooling_clean'] = df['pooling_strategy'].map(pooling_map).fillna(df['pooling_strategy'])
    
    return df

def main():
    print("Testing notebook functionality...")
    
    # Test 1: Data loading
    print("\n1. Testing data loading...")
    df = load_experiment_data()
    print(f"✓ Loaded {len(df)} experiment results")
    
    # Test 2: Feature extraction
    print("\n2. Testing feature extraction...")
    df_processed = extract_descriptor_features(df)
    print(f"✓ Base descriptors: {df_processed['base_descriptor'].unique()}")
    print(f"✓ Pooling strategies: {df_processed['pooling_clean'].unique()}")
    
    # Test 3: Basic statistics
    print("\n3. Testing statistics...")
    stats = df_processed[['mean_average_precision', 'precision_at_1', 'precision_at_5']].describe()
    print(f"✓ Statistics computed successfully")
    print(f"  MAP range: {stats.loc['min', 'mean_average_precision']:.4f} - {stats.loc['max', 'mean_average_precision']:.4f}")
    
    # Test 4: Visualization
    print("\n4. Testing visualization...")
    try:
        # Create a simple test plot
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Test Plot'])
        
        desc_performance = df_processed.groupby(['base_descriptor', 'pooling_clean'])['mean_average_precision'].mean().reset_index()
        
        for pooling in desc_performance['pooling_clean'].unique():
            data = desc_performance[desc_performance['pooling_clean'] == pooling]
            fig.add_trace(
                go.Bar(name=f'{pooling}', x=data['base_descriptor'], y=data['mean_average_precision']),
                row=1, col=1
            )
        
        fig.update_layout(title="Test Visualization")
        print("✓ Plotly visualization created successfully")
        
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        return False
    
    # Test 5: Output directory creation
    print("\n5. Testing output directory...")
    output_dir = Path("analysis/outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")
    
    print("\n🎉 All tests passed! The notebook should work correctly.")
    print(f"\nData Summary:")
    print(f"- Total experiments: {len(df_processed)}")
    print(f"- Descriptor types: {len(df_processed['descriptor_type'].unique())}")
    print(f"- Best MAP: {df_processed['mean_average_precision'].max():.4f}")
    print(f"- Processing time range: {df_processed['processing_time_ms'].min()/1000:.1f}s - {df_processed['processing_time_ms'].max()/1000:.1f}s")
    
    return True

if __name__ == "__main__":
    main()