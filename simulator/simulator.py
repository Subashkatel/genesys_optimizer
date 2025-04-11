import os
import csv
import json
import time
import logging
import glob
import subprocess
import re
import threading
from utils.logging_utils import setup_logging

# Global semaphore to limit concurrent compilations/simulations
# This helps prevent overwhelming the system
MAX_CONCURRENT_RUNS = 4  # Adjust based on your system's capabilities
compile_semaphore = threading.Semaphore(MAX_CONCURRENT_RUNS)

logger = setup_logging("genesys_optimizer.simulator")

def check_output_readiness(full_output_dir, layer_name, timeout=60):
    """
    Monitor output directory until it's ready for simulation or timeout occurs.
    
    Args:
        full_output_dir: Absolute path to output directory
        layer_name: Name of the layer
        timeout: Maximum seconds to wait
        
    Returns:
        Boolean indicating if directory is ready
    """
    start_time = time.time()
    poll_interval = 0.5  # Start with 0.5 second between checks
    
    while (time.time() - start_time) < timeout:
        if os.path.exists(full_output_dir):
            # Check if directory is ready by verifying essential files exist
            json_files = glob.glob(os.path.join(full_output_dir, layer_name, "*.json"))
            
            if json_files:
                logger.info(f"Output directory ready for simulation: {full_output_dir}")
                return True
                
        # Adaptive polling interval - increase over time but cap at 2 seconds
        poll_interval = min(poll_interval * 1.2, 2.0)
        time.sleep(poll_interval)
    
    logger.error(f"Timed out waiting for output directory: {full_output_dir}")
    return False

def run_simulator(output_dir, layer_name, sim_path=None, metric_column=None, max_retries=2):
    """Run the simulator on a compiled layer and get performance metrics."""
    # Use the semaphore to limit concurrent operations
    with compile_semaphore:
        # Save current directory
        current_dir = os.getcwd()

        if not sim_path:
            logger.error("Simulator path not provided")
            return None
        
        if not os.path.isabs(output_dir):
            full_output_dir = os.path.abspath(os.path.join(current_dir, output_dir))
        else:
            full_output_dir = output_dir
        
        # Initial delay before checking for directory
        initial_delay = 5.0  # seconds
        logger.info(f"Waiting {initial_delay} seconds before checking for output directory")
        time.sleep(initial_delay)
        
        # Improved exponential backoff for directory readiness
        max_attempts = 15
        wait_time = 2.0
        total_wait_time = initial_delay
        
        for attempt in range(max_attempts):
            if os.path.exists(full_output_dir):
                # More thorough verification of directory readiness
                layer_dir = os.path.join(full_output_dir, layer_name)
                if os.path.exists(layer_dir):
                    # Check if the directory actually has content - ONLY check for layer JSON files
                    json_files = glob.glob(os.path.join(layer_dir, "*.json"))
                    
                    if json_files:
                        logger.info(f"Layer directory is ready with {len(json_files)} JSON files after {total_wait_time:.1f}s")
                        break
                    else:
                        logger.info(f"Layer directory exists but missing required JSON files")
            
            # Wait with exponential backoff
            logger.info(f"Waiting for output directory (attempt {attempt+1}/{max_attempts}): {full_output_dir}")
            time.sleep(wait_time)
            total_wait_time += wait_time
            wait_time = min(wait_time * 1.5, 10.0)
            
            if attempt == max_attempts - 1:
                logger.error(f"Output directory not ready after {max_attempts} attempts ({total_wait_time:.1f}s total): {full_output_dir}")
                if os.path.exists(full_output_dir):
                    # Log what's actually in the directory for debugging
                    layer_dir = os.path.join(full_output_dir, layer_name)
                    logger.error(f"Layer directory exists: {os.path.exists(layer_dir)}")
                    if os.path.exists(layer_dir):
                        files = os.listdir(layer_dir)
                        logger.error(f"Layer directory contents: {files}")
                return None
        
        # Create a unique output filename with unique timestamp to avoid collisions
        model_name = os.path.basename(full_output_dir)
        timestamp = int(time.time() * 1000)  # Milliseconds for uniqueness
        output_file = f"{model_name}_{timestamp}_simulation_results.csv"
        
        # Construct the simulator command - the "configs/" reference here is part of the simulator's 
        # command line interface, not necessarily a directory that needs to exist in our output folder
        cmd = [
            "python3", 
            "-m", 
            "genesys_sim.genesys", 
            "configs/", 
            full_output_dir, 
            "--mode", 
            "perf", 
            "--log_path", 
            output_file
        ]

        for attempt in range(max_retries):
            try:
                # Change to simulator directory first
                os.chdir(sim_path)
                logger.info(f"Running simulator (attempt {attempt + 1}/{max_retries}) from {sim_path}: {' '.join(cmd)}")
                
                # Run the simulator with timeout
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)  # Increased timeout
                
                # Output CSV is in the simulator directory
                output_csv = os.path.join(sim_path, output_file)

                os.chdir(current_dir)

                if not os.path.exists(output_csv):
                    logger.warning(f"Simulator output CSV file not found: {output_csv} (attempt {attempt +1})")
                    
                    # Try to find any output file that might have been generated
                    potential_files = glob.glob(os.path.join(sim_path, f"{model_name}*_simulation_results.csv"))
                    if potential_files:
                        latest_file = max(potential_files, key=os.path.getmtime)
                        logger.info(f"Found alternative output file: {latest_file}")
                        output_csv = latest_file
                    else:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Failed to run the simulator after {max_retries} attempts.")
                            return None
                        
                # Parse the results
                metrics = parse_simulator_output(output_csv, layer_name, metric_column)
                
                # Also look for layer-specific output files if standard parsing fails
                if metrics is None:
                    layer_output_files = find_layer_output_files(sim_path, full_output_dir, layer_name)
                    if layer_output_files:
                        logger.info(f"Trying to parse metrics from layer-specific output files")
                        metrics = parse_layer_output_files(layer_output_files, layer_name, metric_column)
                
                # Clean up the output file to avoid cluttering the simulator directory
                try:
                    os.remove(output_csv)
                except:
                    pass
                    
                return metrics
                
            except (subprocess.CalledProcessError, FileNotFoundError, IOError, subprocess.TimeoutExpired) as e:
                os.chdir(current_dir)
                
                logger.warning(f"Simulator attempt {attempt+1} failed: {str(e)}")
                if hasattr(e, 'stderr') and e.stderr:
                    logger.warning(f"Simulator error output: {e.stderr.decode()}")
                    
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("All simulator attempts failed")
                    return None
                    
        return None

def find_layer_output_files(sim_path, output_dir, layer_name):
    """
    Search for any output files that might contain metrics for the specific layer.
    
    Args:
        sim_path: Path to the simulator
        output_dir: Output directory for compilation results
        layer_name: Name of the layer
        
    Returns:
        List of paths to potential output files
    """
    potential_files = []
    
    # Look for CSV files in the simulator directory
    csv_files = glob.glob(os.path.join(sim_path, "*.csv"))
    for file in csv_files:
        if "_simulation_results" in file:
            potential_files.append(file)
    
    # Look for log or stats files in the output directory
    stats_files = glob.glob(os.path.join(output_dir, "*.log")) + \
                 glob.glob(os.path.join(output_dir, "*.stats")) + \
                 glob.glob(os.path.join(output_dir, "*.json"))
    potential_files.extend(stats_files)
    
    # Look for files specific to this layer
    layer_files = glob.glob(os.path.join(output_dir, layer_name, "*.json")) + \
                 glob.glob(os.path.join(output_dir, layer_name, "*.log")) + \
                 glob.glob(os.path.join(output_dir, layer_name, "*.stats"))
    potential_files.extend(layer_files)
    
    return potential_files

def parse_layer_output_files(file_paths, layer_name, metric_column=None):
    """
    Try to extract metrics from alternative output files.
    
    Args:
        file_paths: List of paths to potential output files
        layer_name: Name of the layer
        metric_column: Specific metric to extract
        
    Returns:
        Dictionary of metrics or specific metric value, or None if parsing failed
    """
    for file_path in file_paths:
        try:
            if file_path.endswith('.json'):
                # Try to parse as JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Look for metrics in different JSON structures
                if isinstance(data, dict):
                    # Check for layer name as a key
                    if layer_name in data:
                        layer_data = data[layer_name]
                        if isinstance(layer_data, dict):
                            tot_cycles = layer_data.get('totCycles') or layer_data.get('cycles')
                            tot_time = layer_data.get('totTime(us)') or layer_data.get('time') or layer_data.get('time_us')
                            
                            if tot_cycles or tot_time:
                                return create_metrics_dict(tot_cycles, tot_time, metric_column, layer_name)
                    
                    # Check for metrics at the top level
                    tot_cycles = data.get('totCycles') or data.get('cycles')
                    tot_time = data.get('totTime(us)') or data.get('time') or data.get('time_us')
                    
                    if tot_cycles or tot_time:
                        return create_metrics_dict(tot_cycles, tot_time, metric_column, layer_name)
                        
                    # Look for layers array
                    layers = data.get('layers') or data.get('layer_results')
                    if isinstance(layers, list):
                        for layer in layers:
                            if layer.get('name') == layer_name:
                                tot_cycles = layer.get('totCycles') or layer.get('cycles')
                                tot_time = layer.get('totTime(us)') or layer.get('time') or layer.get('time_us')
                                
                                if tot_cycles or tot_time:
                                    return create_metrics_dict(tot_cycles, tot_time, metric_column, layer_name)
            
            elif file_path.endswith('.log') or file_path.endswith('.stats'):
                # Try to parse log files for metrics using regex
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for patterns like "Layer <name>: cycles=<value>" or similar
                cycles_pattern = re.compile(rf'{layer_name}.*cycles[=:]\s*(\d+)', re.IGNORECASE)
                time_pattern = re.compile(rf'{layer_name}.*time[=:]\s*(\d+\.?\d*)', re.IGNORECASE)
                
                cycles_match = cycles_pattern.search(content)
                time_match = time_pattern.search(content)
                
                tot_cycles = float(cycles_match.group(1)) if cycles_match else None
                tot_time = float(time_match.group(1)) if time_match else None
                
                if tot_cycles or tot_time:
                    return create_metrics_dict(tot_cycles, tot_time, metric_column, layer_name)
                
        except Exception as e:
            logger.warning(f"Failed to parse metrics from {file_path}: {str(e)}")
    
    return None

def create_metrics_dict(tot_cycles, tot_time, metric_column, layer_name):
    """
    Create a metrics dictionary from extracted values.
    
    Args:
        tot_cycles: Total cycles value
        tot_time: Total time value
        metric_column: Specific metric requested
        layer_name: Layer name for logging
        
    Returns:
        Dictionary of metrics or specific metric value
    """
    metrics = {}
    
    if tot_cycles is not None:
        metrics["totCycles"] = float(tot_cycles)
        
    if tot_time is not None:
        metrics["totTime(us)"] = float(tot_time)
    
    if not metrics:
        return None
        
    logger.info(f"Found metrics for layer {layer_name}: {metrics}")
    
    # If a specific metric was requested and available, return just that
    if metric_column and metric_column.lower() in [key.lower() for key in metrics]:
        for key, value in metrics.items():
            if key.lower() == metric_column.lower():
                return value
    
    # Otherwise return all metrics
    return metrics

def parse_simulator_output(output_csv, layer_name, metric_column=None):
    """
    Parse the CSV output from the simulator with improved robustness.
    
    Args:
        output_csv: Path to the output CSV file
        layer_name: Name of the layer to extract metrics for
        metric_column: Specific metric column to return (default: None returns all metrics)
        
    Returns:
        Dictionary of metrics or specific metric value, or None if parsing failed
    """
    try:
        with open(output_csv, 'r') as f:
            # First try to find header by scanning for "layername" variations
            reader = csv.reader(f)
            header = None
            
            for row in reader:
                # Check if this row might be a header
                header_candidates = ["layername", "layer name", "name", "layer"]
                if any(any(candidate in col.lower() for candidate in header_candidates) for col in row):
                    header = row
                    break
            
            # If we couldn't find a header, reset and try a different approach
            if header is None:
                f.seek(0)
                reader = csv.reader(f)
                
                # Try the first non-empty row as header
                for row in reader:
                    if any(col.strip() for col in row):
                        header = row
                        break
            
            # Still no header? Return None
            if header is None:
                logger.error("Could not find a valid header row in CSV")
                return None
            
            # Find index of layername column (try multiple variations)
            layer_idx = None
            layer_col_names = ["layername", "layer name", "name", "layer"]
            
            for i, col in enumerate(header):
                if any(name in col.lower() for name in layer_col_names):
                    layer_idx = i
                    break
            
            if layer_idx is None:
                logger.error(f"Could not find layer name column in header: {header}")
                return None
            
            # Find indices for metrics columns (try multiple variations)
            metric_indices = {}
            layer_type_idx = None
            
            # First find the layer type index if it exists
            for i, col in enumerate(header):
                if "layertype" in col.lower() or "layer type" in col.lower():
                    layer_type_idx = i
                    break
            
            # Find all possible metric columns
            for i, col in enumerate(header):
                col_lower = col.lower().strip()
                
                # Standard metrics
                if "totcycles" in col_lower:
                    metric_indices["totCycles"] = i
                elif "time" in col_lower and ("us" in col_lower or "micro" in col_lower):
                    metric_indices["totTime(us)"] = i
                # SIMD-specific metrics
                elif "simdtotalcycles" in col_lower:
                    metric_indices["simdtotalCycles"] = i
                # Add more metrics as needed
                elif "memory" in col_lower or "mem" in col_lower:
                    metric_indices["memory"] = i
            
            if not metric_indices:
                logger.error(f"Could not find any metric columns in header: {header}")
                return None
            
            # Scan for the layer's row
            f.seek(0)
            next(reader)  # Skip header row
            
            for row in reader:
                if len(row) <= layer_idx:
                    continue
                    
                # Try either exact match or case-insensitive match
                row_layer = row[layer_idx]
                if row_layer == layer_name or row_layer.lower() == layer_name.lower():
                    metrics = {}
                    
                    # Determine if this is a SIMD layer
                    is_simd = False
                    if layer_type_idx is not None and len(row) > layer_type_idx:
                        is_simd = row[layer_type_idx].lower() == "simd"
                    
                    # Extract all available metrics
                    for metric_name, col_idx in metric_indices.items():
                        if col_idx < len(row) and row[col_idx].strip():
                            try:
                                metrics[metric_name] = float(row[col_idx])
                            except ValueError:
                                logger.warning(f"Non-numeric value '{row[col_idx]}' for metric {metric_name}")
                    
                    if metrics:
                        # Map SIMD-specific metrics to standard metric names if needed
                        if is_simd or "simdtotalCycles" in metrics:
                            # For SIMD layers, map simdtotalCycles to totCycles if it doesn't exist
                            if "simdtotalCycles" in metrics and "totCycles" not in metrics:
                                metrics["totCycles"] = metrics["simdtotalCycles"]
                                logger.info(f"Mapped simdtotalCycles to totCycles for SIMD layer {layer_name}")
                        
                        logger.info(f"Found metrics for layer {layer_name}: {metrics}")
                        
                        # If specific metric requested and available, return just that
                        if metric_column:
                            metric_column_lower = metric_column.lower()
                            
                            # Special handling for specific metric requests
                            if metric_column_lower == "totcycles":
                                # If totCycles was requested but only simdtotalCycles is available, return that
                                if "totCycles" in metrics:
                                    return metrics["totCycles"]
                                elif "simdtotalCycles" in metrics:
                                    return metrics["simdtotalCycles"]
                            
                            # Standard direct lookup
                            for key, value in metrics.items():
                                if key.lower() == metric_column_lower:
                                    return value
                        
                        # Otherwise return all metrics
                        return metrics
            
            logger.error(f"Layer '{layer_name}' not found in CSV data")
            
    except Exception as e:
        logger.error(f"Error parsing simulator output CSV {output_csv}: {str(e)}")
    
    return None