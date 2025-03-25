import os
import json
from utils.logging_utils import setup_logging
from pathlib import Path

logger = setup_logging("genesys_optimizer.layer_extractor")

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
                    "dimensions":   dimensions,
                    "current_tile_splits": tile_splits,
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
        return {}

    return[dir for dir in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, dir))]

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