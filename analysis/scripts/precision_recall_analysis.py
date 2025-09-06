#!/usr/bin/env python3
"""
Precision-Recall Analysis Script for Descriptor Comparison Experiments
Reads CSV outputs from descriptor_compare and generates analysis reports
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import json

class ExperimentAnalyzer:
    def __init__(self, results_folder):
        self.results_folder = Path(results_folder)
        self.data = {}
        self.summary_stats = {}

    def load_experiment_data(self):
        """Load all CSV files from results folder"""
        print(f"Loading experiment data from: {self.results_folder}")

        csv_files = list(self.results_folder.glob("*.csv"))
        if not csv_files:
            print("❌ No CSV files found in results folder")
            return False

        print(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
                # Extract experiment info from filename
                experiment_name = csv_file.stem
                df = pd.read_csv(csv_file)

                # Clean up column names (strip whitespace)
                df.columns = df.columns.str.strip()

                self.data[experiment_name] = df
                print(f"✅ Loaded: {experiment_name} ({len(df)} rows)")

            except Exception as e:
                print(f"⚠️ Failed to load {csv_file}: {e}")

        return len(self.data) > 0

    def calculate_summary_statistics(self):
        """Calculate summary statistics for all experiments"""
        print("\nCalculating summary statistics...")

        for exp_name, df in self.data.items():
            try:
                # Try to identify key columns (flexible column names)
                precision_cols = [col for col in df.columns if 'precision' in col.lower()]
                recall_cols = [col for col in df.columns if 'recall' in col.lower()]
                map_cols = [col for col in df.columns if 'map' in col.lower() or 'average' in col.lower()]

                stats = {
                    'total_comparisons': len(df),
                    'columns': list(df.columns),
                }

                # Calculate statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats['numeric_summary'] = df[numeric_cols].describe().to_dict()

                # Specific metrics if available
                if precision_cols:
                    stats['precision_stats'] = df[precision_cols].describe().to_dict()
                if recall_cols:
                    stats['recall_stats'] = df[recall_cols].describe().to_dict()
                if map_cols:
                    stats['map_stats'] = df[map_cols].describe().to_dict()

                self.summary_stats[exp_name] = stats
                print(f"✅ Statistics calculated for: {exp_name}")

            except Exception as e:
                print(f"⚠️ Failed to calculate statistics for {exp_name}: {e}")

    def generate_precision_recall_plots(self, output_dir):
        """Generate precision-recall plots"""
        print(f"\nGenerating precision-recall plots...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        plt.style.use('default')

        for exp_name, df in self.data.items():
            try:
                # Find precision and recall columns
                precision_cols = [col for col in df.columns if 'precision' in col.lower()]
                recall_cols = [col for col in df.columns if 'recall' in col.lower()]

                if not precision_cols or not recall_cols:
                    print(f"⚠️ No precision/recall columns found in {exp_name}")
                    continue

                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Analysis: {exp_name}', fontsize=16)

                # Plot 1: Precision distribution
                if precision_cols:
                    axes[0, 0].hist(df[precision_cols[0]].dropna(), bins=30, alpha=0.7, color='blue')
                    axes[0, 0].set_title('Precision Distribution')
                    axes[0, 0].set_xlabel('Precision')
                    axes[0, 0].set_ylabel('Frequency')

                # Plot 2: Recall distribution
                if recall_cols:
                    axes[0, 1].hist(df[recall_cols[0]].dropna(), bins=30, alpha=0.7, color='green')
                    axes[0, 1].set_title('Recall Distribution')
                    axes[0, 1].set_xlabel('Recall')
                    axes[0, 1].set_ylabel('Frequency')

                # Plot 3: Precision vs Recall scatter
                if len(precision_cols) > 0 and len(recall_cols) > 0:
                    valid_data = df[[precision_cols[0], recall_cols[0]]].dropna()
                    if len(valid_data) > 0:
                        axes[1, 0].scatter(valid_data[recall_cols[0]], valid_data[precision_cols[0]],
                                         alpha=0.6, color='red')
                        axes[1, 0].set_title('Precision vs Recall')
                        axes[1, 0].set_xlabel('Recall')
                        axes[1, 0].set_ylabel('Precision')

                # Plot 4: Summary statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    summary_data = df[numeric_cols].mean()
                    axes[1, 1].bar(range(len(summary_data)), summary_data.values)
                    axes[1, 1].set_title('Mean Values by Metric')
                    axes[1, 1].set_xticks(range(len(summary_data)))
                    axes[1, 1].set_xticklabels(summary_data.index, rotation=45, ha='right')

                plt.tight_layout()

                plot_file = output_dir / f"{exp_name}_analysis.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"✅ Plot saved: {plot_file}")

            except Exception as e:
                print(f"⚠️ Failed to generate plot for {exp_name}: {e}")

    def generate_comparison_report(self, output_dir):
        """Generate comprehensive comparison report"""
        print(f"\nGenerating comparison report...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create comparison plots if multiple experiments
        if len(self.data) > 1:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('Experiment Comparison', fontsize=16)

                # Collect metrics across experiments
                all_metrics = {}
                for exp_name, df in self.data.items():
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        all_metrics[exp_name] = df[numeric_cols].mean()

                if all_metrics:
                    comparison_df = pd.DataFrame(all_metrics).T

                    # Plot comparison heatmap
                    if len(comparison_df.columns) > 1:
                        sns.heatmap(comparison_df, annot=True, fmt='.3f',
                                  cmap='viridis', ax=axes[0, 0])
                        axes[0, 0].set_title('Metrics Heatmap')

                    # Plot bar comparisons for key metrics
                    if 'precision' in str(comparison_df.columns).lower():
                        precision_cols = [col for col in comparison_df.columns if 'precision' in col.lower()]
                        if precision_cols:
                            comparison_df[precision_cols[0]].plot(kind='bar', ax=axes[0, 1])
                            axes[0, 1].set_title('Precision Comparison')
                            axes[0, 1].tick_params(axis='x', rotation=45)

                plt.tight_layout()
                comparison_file = output_dir / "experiment_comparison.png"
                plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"✅ Comparison plot saved: {comparison_file}")

            except Exception as e:
                print(f"⚠️ Failed to generate comparison plot: {e}")

    def generate_summary_report(self, output_dir):
        """Generate text summary report"""
        output_dir = Path(output_dir)
        report_file = output_dir / "analysis_summary.txt"

        with open(report_file, 'w') as f:
            f.write("DESCRIPTOR COMPARISON ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
            f.write(f"Results Folder: {self.results_folder}\n")
            f.write(f"Experiments Analyzed: {len(self.data)}\n\n")

            for exp_name, stats in self.summary_stats.items():
                f.write(f"\nEXPERIMENT: {exp_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total comparisons: {stats.get('total_comparisons', 'N/A')}\n")
                f.write(f"Columns: {', '.join(stats.get('columns', []))}\n")

                if 'numeric_summary' in stats:
                    f.write("\nNumeric Summary:\n")
                    for col, summary in stats['numeric_summary'].items():
                        if isinstance(summary, dict) and 'mean' in summary:
                            f.write(f"  {col}: mean={summary['mean']:.3f}, std={summary['std']:.3f}\n")

                f.write("\n")

        print(f"✅ Summary report saved: {report_file}")

    def generate_json_report(self, output_dir):
        """Generate JSON report for programmatic access"""
        output_dir = Path(output_dir)
        json_file = output_dir / "analysis_results.json"

        report_data = {
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'results_folder': str(self.results_folder),
                'experiments_count': len(self.data)
            },
            'experiments': self.summary_stats
        }

        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"✅ JSON report saved: {json_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze descriptor comparison results')
    parser.add_argument('results_folder', help='Path to folder containing CSV results')
    parser.add_argument('--output', '-o', default='analysis/outputs',
                       help='Output directory for analysis results')
    parser.add_argument('--plots-only', action='store_true',
                       help='Generate only plots, skip detailed analysis')

    args = parser.parse_args()

    if not os.path.exists(args.results_folder):
        print(f"❌ Results folder not found: {args.results_folder}")
        sys.exit(1)

    print("=== Descriptor Comparison Analysis ===")
    print(f"Results folder: {args.results_folder}")
    print(f"Output directory: {args.output}")

    # Initialize analyzer
    analyzer = ExperimentAnalyzer(args.results_folder)

    # Load data
    if not analyzer.load_experiment_data():
        print("❌ Failed to load experiment data")
        sys.exit(1)

    # Calculate statistics
    if not args.plots_only:
        analyzer.calculate_summary_statistics()

    # Generate outputs
    os.makedirs(args.output, exist_ok=True)

    analyzer.generate_precision_recall_plots(args.output)

    if not args.plots_only:
        analyzer.generate_comparison_report(args.output)
        analyzer.generate_summary_report(args.output)
        analyzer.generate_json_report(args.output)

    print(f"\n✅ Analysis complete! Check results in: {args.output}")

if __name__ == "__main__":
    main()
