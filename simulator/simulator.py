import os
import csv
import subprocess
import time
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.simulator")

def run_simulator(output_dir, layer_name, sim_path = None, metric_column=None, max_retries = 2):
    """run the simulator and a compiled layer and get performance metric.setup_logging

    """
    current_dir = os.getcwd()

    if sim_path is None:
        logger.error("Simulation path is not provided. Cannot run the simulator.")
        return None

    if not os.path.isabs(output_dir):
        full_output_dir = os.path.join(current_dir, output_dir)
    else:
        full_output_dir = output_dir

    model_name = os.path.basename(full_output_dir)
    output_file = f"{model_name}_simulation_results.csv"

    cmd = ["python3", "-m", "genesys_sim.genesys", "configs/", full_output_dir, "--mode", "perf", "--log_path", output_file]

    for attempt in range(max_retries):
        try:
            os.chdir(sim_path)
            logger.info(f"Running simulator (attempt {attempt + 1}/{max_retries}) from {sim_path}: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            output_csv = os.path.join(sim_path, output_file)

            os.chdir(current_dir)

            if not os.path.exists(output_csv):
                logger.warning(f"Simulator output CSV file not found: {output_csv} (attemt{attempt +1})")
                if attempt < max_retries -1:
                    wait_time = 2** attempt
                    logger.info(f"Retrying in {wait_time} secodds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to run the simulator after {max_retries} attempts.")
                    return None
            metrics = parse_simulation_output(output_csv, layer_name, metric_column)
            return metrics
        except (subprocess.CalledProcessError, FileNotFoundError, IOError) as e:
            
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



def parse_simulation_output(output_csv: str, layer_name: str, metric_column = None):
    """Parse the CSV output from the simulator 

    Args:
        output_csv (str): _description_
        layer_name (str): _description_
        metric_column (_type_, optional): _description_. Defaults to None.
    """

    try:
        with open(output_csv, 'r') as f:
            reader = csv.reader(f)
            header = None

            for row in reader:
                if any("layerName" in col for col in row):
                    header = row
                    break

            if header is None:
                logger.error("No header found in the CSV file.")
                return None
            
            try:
                layer_idx = next(i for i, col in enumerate(header) if "layerName" in col and layer_name in col)
            except StopIteration:
                logger.error(f"Column 'LayerName' not found for layer: {layer_name}")
                return None
            
            # find indices for the metric columns
            metric_indices ={}
            for col_name in ["totCycles", "totTime(us)"]:
                try:
                    idx = next(i for i, col in enumerate(header) if col.strip().lower() == col_name.lower())
                    metric_indices[col_name] = idx
                except StopIteration:
                    logger.warning(f"Metric column '{col_name}' not found in header")
            
            if not metric_indices:
                logger.error("No metric columns found in header")
                return None
            
            # look for the row that matches the given layer name
            for row in reader:
                if len(row) <= layer_idx:
                    continue
                if row[layer_idx] == layer_name:
                    metrics = {}
                    for metric_name, idx in metric_indices.items():
                        try:
                            metrics[metric_name] = float(row[idx])
                        except (ValueError, IndexError):
                            logger.warning(f"Could not convert {metric_name} to float for layer {layer_name}")
                            metrics[metric_name] = None
                    
                    logger.info(f"Found metrics for layer {layer_name}: {metrics}")
                    
                    # If a specific metric was requested, return that value
                    if metric_column and metric_column in metrics:
                        return metrics[metric_column]
                    
                    # Otherwise return all metrics
                    return metrics
            
            logger.error(f"Layer '{layer_name}' not found in CSV data")
            return None
        
    except (IOError, csv.Error) as e:
        logger.error(f"Error reading CSV file '{output_csv}': {str(e)}")
        return None