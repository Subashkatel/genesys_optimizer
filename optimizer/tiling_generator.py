"""Generate tiling configurations for model layers."""

import itertools
import logging
from utils.math_utils import get_factors
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.tiling_generator")

def generate_all_tiling_configs(layer_info):
    """
    Generate ALL possible tiling configurations for a layer without any sampling.
    
    Args:
        layer_info: Dictionary containing layer information
        
    Returns:
        List of all possible tiling configurations
    """
    dimensions = layer_info["dimensions"]
    
    # factors for each dimension that has size > 1
    factor_dict = {}
    for dim, size in dimensions.items():
        if size > 1:
            factor_dict[dim] = get_factors(size)
        else:
            factor_dict[dim] = [1]
    
    # calculates the total possible configurations
    total_configs = 1
    for factors in factor_dict.values():
        total_configs *= len(factors)
    
    logger.info(f"Generating all {total_configs} possible tiling configurations")
    
    # generates all possible combinations
    dims = list(dimensions.keys())
    factor_lists = [factor_dict[dim] for dim in dims]
    
    configs = []
    for split_combination in itertools.product(*factor_lists):
        config = {dim: split for dim, split in zip(dims, split_combination)}
        configs.append(config)
    
    return configs

def generate_tiling_configs(layer_info, max_configs=None):
    """
    Generate tiling configurations for a layer.
    If max_configs is None or negative, generate all possible configurations.
    Otherwise, use sampling strategy to limit configurations.
    

    """
    if max_configs is None or max_configs < 0:
        return generate_all_tiling_configs(layer_info)
    
    dimensions = layer_info["dimensions"]
    configs = []
    
    # generates factors for each dimension
    factor_dict = {}
    for dim, size in dimensions.items():
        factor_dict[dim] = get_factors(size)
    
    # Prioritize dimensions to optimize - use current tile splits as a hint
    current_splits = layer_info.get("current_tile_splits", {})
    
    # Determine which dimensions are currently tiled (have values > 1)
    tiled_dims = [dim for dim, value in current_splits.items() 
                 if value > 1 and dim in dimensions]
    
    # If no dimensions are currently tiled, use heuristics to find good candidates
    dims_to_optimize = select_dimensions_to_optimize(dimensions, tiled_dims)
    
    # If we have dimensions to optimize
    if dims_to_optimize:
        # Get factors for each dimension to optimize
        dimension_factors = [factor_dict[dim] for dim in dims_to_optimize]
        
        # Calculate total possible configurations
        total_configs = 1
        for factors in dimension_factors:
            total_configs *= len(factors)
        
        # If too many configurations, sample strategically
        if total_configs > max_configs:
            configs = sample_tiling_configs(dimensions, dims_to_optimize, factor_dict, 
                                           current_splits, max_configs)
        else:
            # If reasonable number of configurations, generate all combinations
            configs = generate_all_combinations(dimensions, dims_to_optimize, dimension_factors)
    else:
        # If no dimensions to optimize, just use default configuration
        configs.append({dim: 1 for dim in dimensions.keys()})
    
    return configs

def select_dimensions_to_optimize(dimensions, tiled_dims, max_dims=3):
    """
    Select which dimensions to optimize based on heuristics.
    
    Args:
        dimensions: Dictionary of dimension names and sizes
        tiled_dims: List of dimensions that are already tiled
        max_dims: Maximum number of dimensions to optimize
        
    Returns:
        List of dimension names to optimize
    """
    dims_to_optimize = []
    
    if tiled_dims:
        # Prioritize dimensions that are already tiled
        dims_to_optimize = tiled_dims
    else:
        # Choose dimensions based on size - larger dimensions often benefit more from tiling
        # Skip dimensions with size 1 as they don't benefit from tiling
        sizeable_dims = [(dim, size) for dim, size in dimensions.items() if size > 1]
        
        # Sort dimensions by size in descending order
        sorted_dims = sorted(sizeable_dims, key=lambda x: x[1], reverse=True)
        
        # Take the top few largest dimensions
        dims_to_optimize = [dim for dim, _ in sorted_dims[:max_dims]]
    
    # Always include at least one dimension if possible
    if not dims_to_optimize and dimensions:
        # Pick any dimension except those with size 1
        for dim, size in dimensions.items():
            if size > 1:
                dims_to_optimize.append(dim)
                break
                
    return dims_to_optimize

def sample_tiling_configs(dimensions, dims_to_optimize, factor_dict, current_splits, max_configs):
    """
    Sample a representative set of tiling configurations.
    
    Args:
        dimensions: Dictionary of dimension names and sizes
        dims_to_optimize: List of dimensions to optimize
        factor_dict: Dictionary mapping dimensions to their factors
        current_splits: Dictionary of current tile splits
        max_configs: Maximum number of configurations to generate
        
    Returns:
        List of tiling configurations
    """
    import random
    logger.info(f"Sampling {max_configs} tiling configurations")
    configs = []
    
    # Always include the current configuration
    current_config = {dim: current_splits.get(dim, 1) for dim in dimensions.keys()}
    configs.append(current_config)
    
    # Include a configuration with all splits set to 1 (baseline)
    default_config = {dim: 1 for dim in dimensions.keys()}
    if default_config != current_config:
        configs.append(default_config)
    
    # Add configurations with max tiling for each dimension
    for dim in dims_to_optimize:
        max_factor = max(factor_dict[dim])
        config = {d: 1 for d in dimensions.keys()}
        config[dim] = max_factor
        if config not in configs:
            configs.append(config)
    
    # Add some configurations with median factors
    for dim in dims_to_optimize:
        factors = factor_dict[dim]
        if len(factors) > 2:
            median_factor = factors[len(factors) // 2]
            config = {d: 1 for d in dimensions.keys()}
            config[dim] = median_factor
            if config not in configs:
                configs.append(config)
    
    # If we still need more configurations, add random combinations
    attempts = 0
    while len(configs) < max_configs and attempts < max_configs * 3:
        attempts += 1
        config = {dim: 1 for dim in dimensions.keys()}
        
        # Randomly choose which dimensions to tile in this configuration
        dims_to_tile = random.sample(
            dims_to_optimize, 
            k=random.randint(1, min(len(dims_to_optimize), 2))
        )
        
        for dim in dims_to_tile:
            factors = factor_dict[dim]
            config[dim] = random.choice(factors)
        
        # Only add if unique
        if config not in configs:
            configs.append(config)
            
    return configs

def generate_all_combinations(dimensions, dims_to_optimize, dimension_factors):
    """
    Generate all combinations of tiling configurations.
    
    Args:
        dimensions: Dictionary of dimension names and sizes
        dims_to_optimize: List of dimensions to optimize
        dimension_factors: List of factors for each dimension to optimize
        
    Returns:
        List of tiling configurations
    """
    configs = []
    for splits in itertools.product(*dimension_factors):
        config = {dim: 1 for dim in dimensions.keys()}
        for idx, dim in enumerate(dims_to_optimize):
            config[dim] = splits[idx]
        configs.append(config)
    return configs