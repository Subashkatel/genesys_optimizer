"""Optimize tiling for individual model layers."""

import os
import logging
import concurrent.futures
import time
import queue
from compiler.model_compiler import compile_model
from simulator.simulator import run_simulator
from optimizer.tiling_generator import generate_tiling_configs, generate_all_tiling_configs
from optimizer.checkpoint import CheckpointManager
from optimizer.cache import OptimizationCache
from utils.logging_utils import setup_logging
from compiler.layer_extractor import get_hardware_config_from_config_path, find_output_dir

logger = setup_logging("genesys_optimizer.layer_optimizer")

def optimize_layer(model_path, layer_name, layer_info, output_dir, sim_path, 
                   metric="totCycles", max_configs=10, compile_retries=3, sim_retries=2,
                   model_name=None, optimization_cache=None, hardware_config=None, config_path=None):
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
        hardware_config: Hardware configuration string (e.g., "genesys32x32")
        config_path: Path to the hardware configuration file
        
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
            # Fix parameter order - was passing parameters in wrong positions
            compile_success = compile_model(
                model_path=model_path,
                config_path=config_path,  # Add the config path
                experiment_name=test_exp_name,
                tiling_config=tiling_config,
                max_retries=compile_retries
            )
            
            if not compile_success:
                logger.warning(f"Compilation failed for {layer_name} configuration {i+1} after all retries")
                continue
            
            # Create the absolute path to the test output directory - including hardware config
            if hardware_config:
                dir_name = f"{model_name}_{hardware_config}_{test_exp_name}"
            else:
                dir_name = f"{model_name}_{test_exp_name}"
            
            test_output_dir = os.path.join(output_dir, dir_name)
            
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
                           checkpoint_dir="checkpoints", checkpoint_interval=60,
                           enable_caching=True, cache_dir="layer_cache",
                           config_path=None):
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
        config_path: Path to the hardware configuration file
        
    Returns:
        Dictionary mapping layer names to optimization results
    """
    model_name = os.path.basename(model_path).split('.')[0]
    
    # Make checkpoint and cache directories absolute if they're relative
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.abspath(cache_dir)
    
    # Extract hardware configuration from config path if available
    hardware_config = None
    if config_path:
        hardware_config = get_hardware_config_from_config_path(config_path)
        logger.info(f"Using hardware configuration: {hardware_config}")
    
    # Initialize checkpoint manager with frequent saves
    checkpoint_manager = CheckpointManager(model_name, checkpoint_dir)
    checkpoint_manager.set_save_interval(checkpoint_interval)
    logger.info(f"Using checkpoint directory: {checkpoint_dir} with interval {checkpoint_interval}s")
    
    # Initialize layer cache if enabled
    optimization_cache = None
    if enable_caching:
        optimization_cache = OptimizationCache(model_name, cache_dir)
        logger.info(f"Layer similarity caching enabled using directory: {cache_dir}")
    
    # Initialize results dictionary with any existing checkpoint data
    results = checkpoint_manager.load_checkpoint()
    has_checkpoint = bool(results)
    
    if has_checkpoint:
        logger.info(f"Resuming optimization from checkpoint with {len(results)} completed layers")
        # Filter out layers that are already optimized
        layers_info = [(name, info) for name, info in layers_info if name not in results]
        logger.info(f"Remaining layers to optimize: {len(layers_info)}")
    
    # Group layers by operation type for better cache locality
    grouped_layers = {}
    for layer_name, layer_info in layers_info:
        op_type = layer_info.get("operation")
        if op_type not in grouped_layers:
            grouped_layers[op_type] = []
        grouped_layers[op_type].append((layer_name, layer_info))
    
    # Sort operations by estimated complexity (number of configs)
    operation_complexity = []
    for op_type, op_layers in grouped_layers.items():
        total_configs = sum(
            max_configs_per_layer if max_configs_per_layer > 0 else 
            len(generate_all_tiling_configs(layer_info)) 
            for _, layer_info in op_layers
        )
        operation_complexity.append((op_type, total_configs))
    
    # Sort operations by complexity in descending order
    sorted_operations = sorted(operation_complexity, key=lambda x: x[1], reverse=True)
    
    # Create a prioritized list of layers
    prioritized_layers = []
    for op_type, _ in sorted_operations:
        prioritized_layers.extend(grouped_layers[op_type])
    
    # Now use a shared queue for all workers for better load balancing
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Add all layer optimization tasks to the queue with improved prioritization
    total_tasks = 0
    
    # Group similar configurations together for better cache utilization
    # This helps reduce thrashing of the compilation system
    config_groups = {}
    
    for layer_idx, (layer_name, layer_info) in enumerate(prioritized_layers):
        # Generate configurations for this layer
        if max_configs_per_layer < 0:
            configs = generate_all_tiling_configs(layer_info)
        else:
            configs = generate_tiling_configs(layer_info, max_configs_per_layer)
        
        # Group configurations by their "size complexity"
        for i, tile_splits in enumerate(configs):
            # Create a complexity key based on the configuration
            complexity = sum(val for val in tile_splits.values())
            if complexity not in config_groups:
                config_groups[complexity] = []
            
            config_groups[complexity].append({
                'layer_name': layer_name,
                'layer_info': layer_info,
                'tile_splits': tile_splits,
                'config_idx': i,
                'total_configs': len(configs),
                'priority': layer_idx  # Lower is higher priority
            })
            total_tasks += 1
    
    # Now add tasks to queue in order of complexity (simplest first)
    # This helps ensure we get some results quickly
    for complexity in sorted(config_groups.keys()):
        for task in config_groups[complexity]:
            task_queue.put(task)
    
    logger.info(f"Added {total_tasks} configuration tasks to the queue")
    
    # Track the best configuration for each layer
    layer_best_configs = {}
    
    # Worker function to process tasks from the queue
    def worker():
        while True:
            try:
                # Get a task from the queue
                task = task_queue.get(block=False)
                layer_name = task['layer_name']
                layer_info = task['layer_info']
                tile_splits = task['tile_splits']
                config_idx = task['config_idx']
                total_configs = task['total_configs']
                
                logger.info(f"Testing configuration {config_idx+1}/{total_configs} for {layer_name}: {tile_splits}")
                
                # Create tiling config for this layer
                tiling_key = layer_info["tiling_key"]
                tiling_config = {tiling_key: {"1": tile_splits}}
                
                test_exp_name = f"{layer_name}_test_{config_idx}"
                
                try:
                    # Compile the model with this configuration
                    compile_success = compile_model(
                        model_path=model_path,
                        config_path=config_path,
                        experiment_name=test_exp_name,
                        tiling_config=tiling_config,
                        max_retries=compile_retries
                    )
                    
                    if not compile_success:
                        logger.warning(f"Compilation failed for {layer_name} configuration {config_idx+1}")
                        task_queue.task_done()
                        continue
                    
                    # Create the output directory path
                    if hardware_config:
                        dir_name = f"{model_name}_{hardware_config}_{test_exp_name}"
                    else:
                        dir_name = f"{model_name}_{test_exp_name}"
                    
                    test_output_dir = os.path.join(output_dir, dir_name)
                    
                    # Run the simulator
                    metrics = run_simulator(test_output_dir, layer_name, sim_path, max_retries=sim_retries)
                    
                    if metrics is None:
                        logger.warning(f"Failed to get metrics for {layer_name} configuration {config_idx+1}")
                        task_queue.task_done()
                        continue
                    
                    # Extract the metric value
                    metric_value = metrics.get(metric) if isinstance(metrics, dict) else metrics
                    
                    if metric_value is None:
                        logger.error(f"Requested metric {metric} not found for {layer_name}")
                        task_queue.task_done()
                        continue
                    
                    # Add result to result queue
                    result_queue.put({
                        'layer_name': layer_name,
                        'tiling_key': tiling_key,
                        'tile_splits': tile_splits,
                        'metric_value': metric_value
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing {layer_name} configuration {config_idx+1}: {str(e)}")
                
                task_queue.task_done()
                
            except queue.Empty:
                # No more tasks in queue
                break
    
    # Process result queue and update best configurations
    def result_processor():
        last_checkpoint_save = time.time()
        
        while True:
            try:
                # Process any available results
                result = result_queue.get(block=False)
                layer_name = result['layer_name']
                tiling_key = result['tiling_key']
                tile_splits = result['tile_splits']
                metric_value = result['metric_value']
                
                # Check if this is better than current best
                if layer_name not in layer_best_configs or metric_value < layer_best_configs[layer_name]['best_metric']:
                    logger.info(f"New best configuration for {layer_name} with {metric} = {metric_value}")
                    layer_best_configs[layer_name] = {
                        'best_config': tile_splits,
                        'best_metric': metric_value,
                        'tiling_key': tiling_key
                    }
                    
                    # Update results dictionary
                    results[layer_name] = layer_best_configs[layer_name]
                    
                    # Save checkpoint on each improvement
                    checkpoint_manager.update_layer_result(layer_name, layer_best_configs[layer_name])
                    last_checkpoint_save = time.time()
                
                result_queue.task_done()
                
            except queue.Empty:
                # No results to process, check if we should save checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_save >= checkpoint_interval and layer_best_configs:
                    checkpoint_manager.save_checkpoint(force=True)
                    last_checkpoint_save = current_time
                
                # Check if we're done (no tasks and no results pending)
                if task_queue.empty() and result_queue.empty():
                    break
                
                # Brief pause to avoid CPU spinning
                time.sleep(0.1)
    
    # Calculate number of workers to use - handle the case when max_workers is None
    if max_workers is None:
        # Default is min(32, os.cpu_count() + 4)
        cpu_count = os.cpu_count() or 4
        # Use fewer workers to avoid system overload
        recommended_workers = min(8, max(1, cpu_count // 2))
        actual_workers = recommended_workers
    else:
        actual_workers = max_workers
    
    logger.info(f"Using {actual_workers} worker threads for optimization")
    
    # Use ThreadPoolExecutor for workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        # Start workers - reserve 1 thread for the result processor
        worker_count = max(1, actual_workers - 1)
        workers = [executor.submit(worker) for _ in range(worker_count)]
        
        # Start result processor in the executor too
        result_processor_future = executor.submit(result_processor)
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(workers + [result_processor_future]):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker thread failed with error: {str(e)}")
    
    # Cache the optimized configurations
    if enable_caching:
        for layer_name, best_result in layer_best_configs.items():
            for layer_name_orig, layer_info_orig in layers_info:
                if layer_name_orig == layer_name:
                    optimization_cache.add_to_cache(layer_info_orig, best_result)
                    break
    
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