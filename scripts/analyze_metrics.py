#!/usr/bin/env python3
"""
Utility script to analyze and verify simulator metrics extraction.
This helps ensure we're getting the correct metrics from the simulation results.
"""

import os
import sys
import csv
import json
from pathlib import Path

# Add the project root to sys.path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from simulator.simulator import parse_simulator_output

def analyze_metrics_file(file_path):
    """Analyze a metrics CSV file and print stats."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Analyzing metrics file: {file_path}")
    
    # First, read all layer names
    layer_names = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = None
        layer_idx = None
        
        # Find the header row and layer index
        for i, row in enumerate(reader):
            if any("layername" in col.lower() for col in row):
                header = row
                for j, col in enumerate(row):
                    if "layername" in col.lower():
                        layer_idx = j
                        break
                break
        
        if header and layer_idx is not None:
            # Reset and read all layers
            f.seek(0)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > layer_idx:
                    layer_names.append(row[layer_idx])
    
    if not layer_names:
        print("No layers found or unable to parse the file.")
        return
    
    print(f"Found {len(layer_names)} layers in the file.")
    print("Sample layers:", layer_names[:5])
    
    # Test extraction for a few layers
    test_layers = layer_names[:5]
    for layer in test_layers:
        metrics = parse_simulator_output(file_path, layer)
        print(f"\nMetrics for {layer}:")
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    else:
        # Use the test metrics file by default
        metrics_file = os.path.join(project_root, "tests", "test_metric.csv")
    
    analyze_metrics_file(metrics_file)
    
    print("\nTo test with a different file, run:")
    print(f"  python {sys.argv[0]} path/to/metrics.csv")
