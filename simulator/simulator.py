"""Run the simulator on compiled model outputs."""

import os
import re
import time
import subprocess
from pathlib import Path
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.simulator")

def wait_for_directory_creation(directory_path, timeout=30, check_interval=0.5):
    """
    Wait for a directory to be created with a timeout.
    
    Args:
        directory_path: Path to the directory to wait for
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        
    Returns:
        True if directory exists, False if timeout reached
    """
    start_time = time.time()
    max_attempts = int(timeout / check_interval)
    
    for i in range(max_attempts):
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            return True
        
        time_elapsed = time.time() - start_time
        time_remaining = timeout - time_elapsed
        
        if time_remaining <= 0:
            break
            
        attempts_left = max_attempts - i
        logger.info(f"Waiting for output directory to be created: {directory_path} (attempts left: {attempts_left})")
        time.sleep(min(check_interval, time_remaining))
    
    return False

def run_simulator(output_dir, layer_name, sim_path, max_retries=1):
    """
    Run the simulator on the compiled model outputs.
    
    Args:
        output_dir: Directory containing compiled model outputs
        layer_name: Name of the layer to simulate
        sim_path: Path to the simulator
        max_retries: Maximum number of simulation retry attempts
        
    Returns:
        Dictionary of performance metrics or None if simulation failed
    """
    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        return None
        
    # Make sure sim_path is an absolute path and exists
    sim_path = os.path.abspath(sim_path)
    if not os.path.exists(sim_path):
        logger.error(f"Simulator path does not exist: {sim_path}")
        return None
    
    # Get directory name from output_dir
    dir_name = os.path.basename(output_dir)
    
    # Wait for the directory creation with timeout
    if not wait_for_directory_creation(output_dir, timeout=30):
        logger.error(f"Output directory does not exist after waiting: {output_dir}")
        return None
    
    # Set the results file name
    results_file = f"{dir_name}_simulation_results.csv"
    
    # Run simulator for up to max_retries attempts
    for i in range(max_retries):
        try:
            # Construct command - working directory should be same as output_dir parent
            cmd = [
                "python3", "-m", "genesys_sim.genesys", 
                "configs/", output_dir, 
                "--mode", "perf", 
                "--log_path", results_file
            ]
            
            logger.info(f"Running simulator (attempt {i+1}/{max_retries}) from {sim_path}: {' '.join(cmd)}")
            
            # Use the simulator path as the working directory
            result = subprocess.run(
                cmd, 
                cwd=sim_path, 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            # Check if simulation was successful
            if result.returncode != 0:
                logger.error(f"Simulator exited with error code {result.returncode}: {result.stderr}")
                if i < max_retries - 1:
                    logger.info(f"Retrying simulation for {layer_name}")
                continue
                
            # Parse results from the CSV file
            results_path = os.path.join(sim_path, results_file)
            if not os.path.exists(results_path):
                logger.error(f"Simulation results file not found: {results_path}")
                if i < max_retries - 1:
                    logger.info(f"Retrying simulation for {layer_name}")
                continue
                
            metrics = parse_simulation_results(results_path, layer_name)
            if metrics:
                logger.info(f"Found metrics for layer {layer_name}: {metrics}")
                return metrics
                
            logger.error(f"Failed to parse metrics for layer {layer_name}")
            if i < max_retries - 1:
                logger.info(f"Retrying simulation for {layer_name}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running simulator: {e.stderr}")
            if i < max_retries - 1:
                logger.info(f"Retrying simulation for {layer_name}")
            else:
                logger.error(f"Failed to run simulator for {layer_name} after {max_retries} attempts")
        except Exception as e:
            logger.error(f"Unexpected error running simulator: {str(e)}")
            if i < max_retries - 1:
                logger.info(f"Retrying simulation for {layer_name}")
            else:
                logger.error(f"Failed to run simulator for {layer_name} after {max_retries} attempts")
                
    return None

def parse_simulation_results(results_path, layer_name):
    """
    Parse simulation results from CSV file.
    
    Args:
        results_path: Path to the simulation results CSV file
        layer_name: Name of the layer to extract metrics for
        
    Returns:
        Dictionary of performance metrics or None if parsing failed
    """
    try:
        with open(results_path, 'r') as f:
            content = f.read()
        
        if not content.strip():
            logger.error(f"Simulation results file is empty: {results_path}")
            return None
            
        # Handle different CSV formats - some newer versions include headers
        lines = content.strip().split('\n')
        headers = None
        metrics = {}
        
        # Try to identify if the file has headers
        if ',' in lines[0] and not lines[0].strip().startswith('#'):
            potential_headers = lines[0].split(',')
            # Check if first item looks like a header
            if not potential_headers[0].replace('.', '').isdigit():
                headers = [h.strip() for h in potential_headers]
                data_lines = lines[1:]
            else:
                data_lines = lines
        else:
            data_lines = lines
            
        # If headers were found, use them to parse metrics
        if headers:
            for line in data_lines:
                values = [v.strip() for v in line.split(',')]
                if len(values) != len(headers):
                    continue
                    
                row_data = dict(zip(headers, values))
                
                # Look for the layer name or similar identifiers
                layer_id = row_data.get('Layer') or row_data.get('LayerName') or row_data.get('Name')
                if layer_id and layer_name in layer_id:
                    # Extract numeric metrics
                    for key, value in row_data.items():
                        try:
                            metrics[key] = float(value)
                        except (ValueError, TypeError):
                            metrics[key] = value
                    return metrics
        else:
            # Fallback to regex pattern matching for older formats
            # Example line: Layer1,Conv,28.0,28.0,12.0,6.0,6.0,12.0,1848.0,1.848
            pattern = fr"({layer_name}|Layer\d+),\w+,(?:[\d.]+,){{6}}([\d.]+),([\d.]+)"
            match = re.search(pattern, content)
            
            if match:
                metrics["totCycles"] = float(match.group(2))
                metrics["totTime(us)"] = float(match.group(3))
                return metrics
                
        # If we reach here, we couldn't find the layer's metrics
        logger.error(f"Could not find metrics for layer {layer_name} in {results_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing simulation results: {str(e)}")
        return None