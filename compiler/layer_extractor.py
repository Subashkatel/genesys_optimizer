import os
import json
import glob
from utils.logging_utils import setup_logging
from pathlib import Path

logger = setup_logging("genesys_optimizer.layer_extractor")

def get_hardware_config_from_config_path(config_path):
    """
    Extract hardware configuration from config filename.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Hardware configuration string (e.g., "genesys32x32")
    """
    if config_path:
        config_filename = os.path.basename(config_path)
        if "32_" in config_filename:
            return "genesys32x32"
        elif "16_" in config_filename:
            return "genesys16x16"
    return None

def find_output_dir(base_dir, model_name, exp_name, config_path=None):
    """
    Find the correct output directory, accounting for hardware configuration.
    
    Args:
        base_dir: Base directory for output
        model_name: Name of the model
        exp_name: Experiment name
        config_path: Path to the hardware configuration file
        
    Returns:
        Path to the output directory if found, otherwise None
    """
    # Try with hardware configuration if provided
    if config_path:
        hw_config = get_hardware_config_from_config_path(config_path)
        if hw_config:
            dir_name = f"{model_name}_{hw_config}_{exp_name}"
            full_path = os.path.join(base_dir, dir_name)
            if os.path.isdir(full_path):
                logger.info(f"Found output directory with hardware config: {full_path}")
                return full_path
    
    # Try the standard path
    std_path = os.path.join(base_dir, f"{model_name}_{exp_name}")
    if os.path.isdir(std_path):
        return std_path
    
    # Try to find using glob pattern
    pattern = os.path.join(base_dir, f"{model_name}_*_{exp_name}")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    
    logger.error(f"Output directory does not exist: {os.path.join(base_dir, f'{model_name}_{exp_name}')}")
    return None

def extract_layers_info(output_dir: str, layer_name:str) -> list:
    """Extract layer information form the output directory given the layer name.

    Args:
        output_dir (str): _description_
        layer_name (str): _description_

    Returns:
        dict: _description_
    """

    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        print(f" output path does not exist: {output_dir}")
        return None
    
    layer_dir = os.path.join(output_dir, layer_name)
    
    if not os.path.exists(layer_dir):
        logger.error(f"Layer directory does not exist: {layer_dir}")
        return None
    
    json_files= list(Path(layer_dir).glob("*_json.json"))
    
    if not json_files:
        logger.error(f"No JSON file found in the layer directory: {layer_dir}")
        return None
    
    json_file = json_files[0]
    try:
        with open(json_file, "r") as f:
            layer_data = json.load(f)
        
        for op in layer_data.get("program", []):
            if isinstance(op, dict) and "operation" in op and "iterable_dimensions" in op:
                operation   = op.get("operation", "")
                instance_id = op.get("instance_id", 1)
                dimensions  = op.get("iterable_dimensions", {})
                tile_splits = op.get("tile_splits", {})

                return {
                    "operation":    operation,
                    "instance_id":  instance_id,
                    "dimensions":     dimensions,
                    "iterable_dimensions":   dimensions,
                    "tile_splits": tile_splits,
                    "tiling_key": f"{operation}_{instance_id}"
                }
            
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {json_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while reading the JSON file {json_file}: {e}")
        return None
    

def get_all_layers(output_dir: str) -> dict:
    """Get all layers from the output directory.

    Args:
        output_dir (str): _description_

    Returns:
        dict: _description_
    """
    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        return []

    return [dir for dir in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, dir))]

def filter_layers_by_pattern(all_layers: list, patterns: list) -> list:
    """filter the layers with matching pattern and return the filtered layers.

    Args:
        all_layers (dict): _description_
        pattern (str): _description_

    Returns:
        dict: _description_
    """

    if not all_layers:
        logger.error("No layers found to filter.")
        return None
    
    if not patterns:
        logger.error("No pattern provided for filtering.")
        return all_layers
    
    return [layer for layer in all_layers if any(pattern in layer for pattern in patterns)]

def filter_layers_by_operation(all_layers: list, output_dir: str, operations: list) -> list:
    """Filter the layers by operations and return the filtered layers.

    Args:
        all_layers (list): _description_
        operations (list): _description_

    Returns:
        list: _description_
    """
    if not all_layers:
        logger.error("No layers found to filter.")
        return None
    
    if not operations:
        logger.error("No operations provided for filtering.")
        return [(layer, extract_layers_info(output_dir, layer)) for layer in all_layers]
    
    filtered_layers = []
    for layer in all_layers:
        layer_info = extract_layers_info(output_dir, layer)
        if layer_info and layer_info.get("operation") in operations:
            filtered_layers.append((layer, layer_info))

    return filtered_layers



