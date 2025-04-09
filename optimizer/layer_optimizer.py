"""Optimize tiling for individual model layers."""

import os
import logging
import concurrent.futures
from compiler.model_compiler import compile_model
from simulator.simulator import run_simulator
from optimizer.tiling_generator import generate_tiling_configs, generate_all_tiling_configs
from optimizer.checkpoint import CheckpointManager
from optimizer.cache import OptimizationCache
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.layer_optimizer")

def optimize_layer(model_path, layer_name, layer_info, output_dir, sim_path, 
                   metric="totCycles", max_configs=10, compile_retries=3, sim_retries=2,
                   model_name=None, optimization_cache=None):
    """
    Optimize tiling configuration for a single layer.
    
    Args:
        model_path: Path to the model file
        layer_name: Name of the layer to optimize
        layer_info: Dictionary containing layer information
        output_dir: Base output directory
        sim_path: Path to the simulator
        metric: Performance metric to optimize
        max_configs: Maximum number of configurations to test (-1 for exhaustive)
        compile_retries: Maximum number of compilation retry attempts
        sim_retries: Maximum number of simulator retry attempts
        model_name: Name of the model (derived from model_path if None)
        optimization_cache: Cache for optimization results (optional)
        
    Returns:
        Tuple of (layer_name, best_config, best_metric, tiling_key)
    """
    if model_name is None:
        model_name = os.path.basename(model_path).split('.')[0]
        
    logger.info(f"Optimizing layer: {layer_name}")
    
    # Make sure output_dir is an absolute path
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    # Check if we have a cached result for a similar layer
    if optimization_cache:
        cached_result = optimization_cache.get_cached_result(layer_info)
        if cached_result:
            logger.info(f"Using cached result for layer {layer_name} from a similar layer")
            cached_config = cached_result.get("best_config")
            cached_metric = cached_result.get("best_metric")
            tiling_key = layer_info["tiling_key"]
            
            # Verify the cached configuration is valid for this layer
            can_use_cached = True
            for dim, value in cached_config.items():
                if dim not in layer_info["dimensions"]:
                    can_use_cached = False
                    break
                if value > layer_info["dimensions"][dim]:
                    can_use_cached = False
                    break
            
            if can_use_cached:
                logger.info(f"Cached configuration is valid for this layer: {cached_config}")
                return (layer_name, cached_config, cached_metric, tiling_key)
            else:
                logger.info("Cached configuration is not valid for this layer, proceeding with optimization")
    
    # Generate tiling configurations - use exhaustive search if max_configs is -1
    if max_configs < 0:
        logger.info(f"Performing exhaustive search for layer {layer_name}")
        tiling_configs = generate_all_tiling_configs(layer_info)
    else:
        tiling_configs = generate_tiling_configs(layer_info, max_configs)
    
    logger.info(f"Testing {len(tiling_configs)} tiling configurations for {layer_name}")
    
    # Track best configuration (minimization)
    best_metric_value = float('inf')
    best_config = None
    
    # Test each configuration
    tiling_key = layer_info["tiling_key"]
    
    for i, tile_splits in enumerate(tiling_configs):
        logger.info(f"Testing configuration {i+1}/{len(tiling_configs)} for {layer_name}: {tile_splits}")
        
        # Create tiling config for this layer using its JSON key
        tiling_config = {tiling_key: {"1": tile_splits}}
        
        test_exp_name = f"{layer_name}_test_{i}"
        try:
            compile_success = compile_model(model_path, test_exp_name, 
                                          tiling_config, 
                                          max_retries=compile_retries)
            if not compile_success:
                logger.warning(f"Compilation failed for {layer_name} configuration {i+1} after all retries")
                continue
            
            # Create the absolute path to the test output directory
            test_output_dir = os.path.join(output_dir, f"{model_name}_{test_exp_name}")
            
            # Run the simulator with the absolute path
            metrics = run_simulator(test_output_dir, layer_name, sim_path, 
                                   max_retries=sim_retries)
            
            if metrics is None:
                logger.warning(f"Failed to get metrics for {layer_name} configuration {i+1} after all retries")
                continue
            
            # Extract the requested metric
            metric_value = metrics.get(metric) if isinstance(metrics, dict) else metrics
            
            if metric_value is None:
                logger.error(f"Requested metric {metric} not found in returned metrics for {layer_name}")
                continue
            
            # Check if this configuration is better (lower metric value)
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_config = tile_splits
                logger.info(f"New best configuration for {layer_name} with {metric} = {best_metric_value}")
        except Exception as e:
            logger.error(f"Error testing configuration {i+1} for {layer_name}: {str(e)}")
            continue
    
    if best_config is not None:
        logger.info(f"Best configuration for {layer_name}: {best_config} with {metric} = {best_metric_value}")
        
        # Add the result to the cache for future similar layers
        if optimization_cache:
            result = {
                "best_config": best_config,
                "best_metric": best_metric_value,
                "tiling_key": tiling_key
            }
            optimization_cache.add_to_cache(layer_info, result)
            logger.info(f"Added optimization result for {layer_name} to cache")
            
        return (layer_name, best_config, best_metric_value, tiling_key)
    else:
        logger.warning(f"No valid configuration found for {layer_name}")
        return (layer_name, None, None, tiling_key)

def optimize_layers_parallel(model_path, layers_info, output_dir, sim_path, 
                           metric="totCycles", max_configs_per_layer=10, 
                           compile_retries=3, sim_retries=2, max_workers=None,
                           checkpoint_dir="checkpoints", checkpoint_interval=300,
                           enable_caching=True, cache_dir="layer_cache"):
    """
    Optimize multiple layers in parallel using a thread pool.
    
    Args:
        model_path: Path to the model file
        layers_info: List of (layer_name, layer_info) tuples
        output_dir: Base output directory
        sim_path: Path to the simulator
        metric: Performance metric to optimize
        max_configs_per_layer: Maximum number of configurations to test per layer
        compile_retries: Maximum number of compilation retry attempts
        sim_retries: Maximum number of simulator retry attempts
        max_workers: Maximum number of parallel workers (None = use CPU count)
        checkpoint_dir: Directory to store checkpoint files
        checkpoint_interval: Interval in seconds between checkpoint saves
        enable_caching: Whether to use caching for similar layers
        cache_dir: Directory to store layer cache files
        
    Returns:
        Dictionary mapping layer names to optimization results
    """
    model_name = os.path.basename(model_path).split('.')[0]
    
    # Initialize checkpoint manager - only for saving progress, no need to load
    checkpoint_manager = CheckpointManager(model_name, checkpoint_dir)
    checkpoint_manager.set_save_interval(checkpoint_interval)
    
    # Initialize layer cache if enabled
    optimization_cache = None
    if enable_caching:
        optimization_cache = OptimizationCache(model_name, cache_dir)
        logger.info("Layer similarity caching enabled")
    
    # Initialize empty results dictionary
    results = {}
    
    # Calculate total configurations to be tested across all layers
    total_configs = 0
    for _, layer_info in layers_info:
        if max_configs_per_layer < 0:
            # For exhaustive search, estimate the total configurations
            dims_with_factors = sum(1 for _, size in layer_info["dimensions"].items() if size > 1)
            # Rough estimate - actual numbers will be displayed during execution
            total_configs += 2 ** dims_with_factors  
        else:
            total_configs += min(max_configs_per_layer, 
                                len(generate_tiling_configs(layer_info, max_configs_per_layer)))
    
    logger.info(f"Total configurations to test: ~{total_configs} (approximate)")
    
    # Define a function to optimize a single layer for use with ThreadPoolExecutor
    def optimize_layer_wrapper(layer_data):
        layer_name, layer_info = layer_data
        return optimize_layer(
            model_path=model_path,
            layer_name=layer_name,
            layer_info=layer_info,
            output_dir=output_dir,
            sim_path=sim_path,
            metric=metric,
            max_configs=max_configs_per_layer,
            compile_retries=compile_retries,
            sim_retries=sim_retries,
            model_name=model_name,
            optimization_cache=optimization_cache
        )
    
    # Use ThreadPoolExecutor to parallelize layer optimization
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all optimization tasks
        future_to_layer = {
            executor.submit(optimize_layer_wrapper, (layer_name, layer_info)): layer_name
            for layer_name, layer_info in layers_info
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_layer):
            layer_name = future_to_layer[future]
            try:
                layer_name, best_config, best_metric, tiling_key = future.result()
                if best_config is not None:
                    layer_result = {
                        "best_config": best_config,
                        "best_metric": best_metric,
                        "tiling_key": tiling_key
                    }
                    # Update results dictionary and checkpoint
                    results[layer_name] = layer_result
                    checkpoint_manager.update_layer_result(layer_name, layer_result)
                    logger.info(f"Layer {layer_name} optimization completed and saved to checkpoint")
            except Exception as e:
                logger.error(f"Layer optimization for {layer_name} failed with error: {str(e)}")
    
    # Ensure final checkpoint is saved
    checkpoint_manager.save_checkpoint(force=True)
    logger.info(f"Final checkpoint saved with {len(results)} optimized layers")
    
    return results

def build_final_tiling_config(optimization_results):
    """
    Build the final tiling configuration from optimization results.
    
    Args:
        optimization_results: Dictionary mapping layer names to optimization results
        
    Returns:
        Dictionary containing the final tiling configuration
    """
    final_config = {}
    
    for layer_name, result in optimization_results.items():
        if "best_config" in result and result["best_config"] is not None:
            tiling_key = result["tiling_key"]
            final_config[tiling_key] = {"1": result["best_config"]}
    
    return final_config