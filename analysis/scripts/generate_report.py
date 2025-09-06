#!/usr/bin/env python3
"""
Comprehensive Experiment Report Generator
Creates detailed HTML reports from descriptor comparison results
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
from datetime import datetime

class ReportGenerator:
    def __init__(self, results_folder, output_dir):
        self.results_folder = Path(results_folder)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_html_report(self):
        """Generate comprehensive HTML report"""

        # Load analysis results
        json_file = self.output_dir / "analysis_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                analysis_data = json.load(f)
        else:
            analysis_data = {"experiments": {}, "metadata": {}}

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Descriptor Comparison Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .experiment {{ border: 1px solid #bdc3c7; margin: 20px 0; padding: 20px; border-radius: 5px; }}
        .experiment h3 {{ margin-top: 0; color: #e74c3c; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .stat-card {{ background: #3498db; color: white; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .plot-card {{ border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }}
        .plot-card img {{ width: 100%; height: auto; }}
        .plot-title {{ background: #34495e; color: white; padding: 10px; margin: 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Descriptor Comparison Analysis Report</h1>

        <div class="metadata">
            <h2>📊 Analysis Overview</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Results Folder:</strong> {self.results_folder}</p>
            <p><strong>Experiments Analyzed:</strong> {len(analysis_data.get('experiments', {}))}</p>
        </div>

        <h2>📈 Visual Analysis</h2>
        <div class="plot-grid">
"""

        # Add plots
        plot_files = list(self.output_dir.glob("*.png"))
        for plot_file in sorted(plot_files):
            plot_name = plot_file.stem.replace('_', ' ').title()
            html_content += f"""
            <div class="plot-card">
                <h3 class="plot-title">{plot_name}</h3>
                <img src="{plot_file.name}" alt="{plot_name}">
            </div>
"""

        html_content += """
        </div>

        <h2>📋 Experiment Details</h2>
"""

        # Add experiment details
        for exp_name, exp_data in analysis_data.get('experiments', {}).items():
            html_content += f"""
        <div class="experiment">
            <h3>🧪 {exp_name}</h3>
            <p><strong>Total Comparisons:</strong> {exp_data.get('total_comparisons', 'N/A')}</p>
            <p><strong>Columns:</strong> {', '.join(exp_data.get('columns', []))}</p>

            <div class="stats">
"""

            # Add summary statistics
            if 'numeric_summary' in exp_data:
                for col_name, stats in exp_data['numeric_summary'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        html_content += f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['mean']:.3f}</div>
                    <div class="stat-label">{col_name} (Mean)</div>
                </div>
"""

            html_content += """
            </div>
        </div>
"""

        html_content += f"""

        <div class="footer">
            <p>Generated by Descriptor Comparison Analysis Tool</p>
            <p>Report created on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

        # Save HTML report
        html_file = self.output_dir / "analysis_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

        print(f"✅ HTML report generated: {html_file}")
        return html_file

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive experiment report')
    parser.add_argument('results_folder', help='Path to folder containing CSV results')
    parser.add_argument('--output', '-o', default='analysis/outputs',
                       help='Output directory for reports')

    args = parser.parse_args()

    print("=== Generating Experiment Report ===")

    generator = ReportGenerator(args.results_folder, args.output)
    html_file = generator.generate_html_report()

    print(f"\n🎉 Report generation complete!")
    print(f"📄 Open in browser: {html_file}")

if __name__ == "__main__":
    main()
