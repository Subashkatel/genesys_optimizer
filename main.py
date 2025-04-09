#!/usr/bin/env python3
"""
GeneSys Optimizer - Main Application Entry Point

This script optimizes tiling configurations for neural network models
on the GeneSys architecture.
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path

from utils.logging_utils import setup_logging
from compiler.model_compiler import prepare_model, compile_model
from compiler.layer_extractor import (
    get_all_layers, 
    filter_layers_by_pattern, 
    filter_layers_by_operation,
    extract_layers_info
)
from optimizer.layer_optimizer import optimize_layers_parallel, build_final_tiling_config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize tiling splits for a neural network model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model ONNX file')
    parser.add_argument('--metric', type=str, default='totCycles', help='Performance metric to optimize (e.g., totCycles or totTime(us))')
    parser.add_argument('--output_dir', type=str, default='genesys_compiler_output',
                        help='Output directory for compilation results')
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--sim_path', type=str, required=True, help='Path to the simulator')
    parser.add_argument('--layers', type=str, nargs='*', help='Specific layers to optimize (optional)')
    parser.add_argument('--max_configs_per_layer', type=int, default=10, 
                        help='Maximum number of configurations to test per layer (use -1 for exhaustive search)')
    parser.add_argument('--exhaustive', action='store_true',
                        help='Try all possible tiling configurations (equivalent to --max_configs_per_layer -1)')
    parser.add_argument('--operation_types', type=str, nargs='*', 
                        default=None, help='Operation types to optimize')
    parser.add_argument('--compile_retries', type=int, default=3,
                        help='Number of compilation retry attempts for transient errors')
    parser.add_argument('--sim_retries', type=int, default=2,
                        help='Number of simulator retry attempts for transient errors')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: number of CPUs)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (default: console only)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to store checkpoint files')
    parser.add_argument('--checkpoint_interval', type=int, default=300,
                        help='Interval in seconds between checkpoint saves')
    parser.add_argument('--enable_caching', action='store_true', default=True,
                        help='Enable caching of optimization results for similar layers')
    parser.add_argument('--disable_caching', action='store_true',
                        help='Disable caching of optimization results for similar layers')
    parser.add_argument('--cache_dir', type=str, default='layer_cache',
                        help='Directory to store layer cache files')
    parser.add_argument('--clear_cache', action='store_true',
                        help='Clear the layer cache before starting')
    
    args = parser.parse_args()
    
    # Handle exhaustive search option
    if args.exhaustive:
        args.max_configs_per_layer = -1
        
    # Handle caching options
    if args.disable_caching:
        args.enable_caching = False
        
    return args

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging("genesys_optimizer", log_file=args.log_file, level=log_level)
    
    # Record start time for benchmarking
    start_time = time.time()
    
    # Extract model name from path
    model_name = os.path.basename(args.model_path).split('.')[0]
    
    logger.info(f"Starting optimization for model: {model_name}")
    logger.info(f"Using simulator path: {args.sim_path}")
    
    # Log whether we're doing exhaustive search
    if args.max_configs_per_layer < 0:
        logger.info("Running in EXHAUSTIVE mode - will try all possible tiling configurations")
    else:
        logger.info(f"Running with max {args.max_configs_per_layer} configurations per layer")
    
    # Log caching status
    if args.enable_caching:
        logger.info(f"Layer similarity caching is ENABLED (using directory: {args.cache_dir})")
        if args.clear_cache:
            from optimizer.cache import OptimizationCache
            cache = OptimizationCache(model_name, args.cache_dir)
            cache.clear_cache()
            logger.info("Cleared existing layer cache")
    else:
        logger.info("Layer similarity caching is DISABLED")
    
    # Step 1: Compile the model with default settings
    exp_name = "default"
    logger.info("Step 1: Preparing and compiling model with default settings")
    prepare_model(args.model_path, max_retries=args.compile_retries)
    compile_success = compile_model(args.model_path, args.config_path, experiment_name=exp_name, max_retries=args.compile_retries)
    
    if not compile_success:
        logger.error("Initial compilation failed after all retries. Exiting.")
        return 1
    
    # Step 2: Get a list of all layers in the compiled model
    logger.info("Step 2: Identifying layers to optimize")
    default_output_dir = os.path.join(args.output_dir, f"{model_name}_{exp_name}")
    all_layers = get_all_layers(default_output_dir)
    
    # Filter layers if specified
    if args.layers:
        layers = filter_layers_by_pattern(all_layers, args.layers)
        logger.info(f"Filtered to {len(layers)} specified layers")
    else:
        # Get all layers but filter by operation type if specified
        if args.operation_types:
            layer_info_pairs = filter_layers_by_operation(
                all_layers, default_output_dir, args.operation_types
            )
            layers = [layer for layer, _ in layer_info_pairs]
            logger.info(f"Filtered to {len(layers)} layers with operations: {args.operation_types}")
        else:
            layers = all_layers
            logger.info(f"Processing all {len(layers)} layers")
    
    # Step 3: For each layer, extract info and prepare for optimization
    logger.info("Step 3: Extracting layer information")
    layers_info = []
    for layer in layers:
        info = extract_layers_info(default_output_dir, layer)
        if info:
            layers_info.append((layer, info))
        else:
            logger.warning(f"Skipping layer {layer} - couldn't extract info")
    
    # Step 4: Optimize layers in parallel
    logger.info(f"Step 4: Optimizing {len(layers_info)} layers in parallel with {args.max_workers or 'auto'} workers")
    optimization_results = optimize_layers_parallel(
        model_path=args.model_path,
        layers_info=layers_info,
        output_dir=args.output_dir,
        sim_path=args.sim_path,
        metric=args.metric,
        max_configs_per_layer=args.max_configs_per_layer,
        compile_retries=args.compile_retries,
        sim_retries=args.sim_retries,
        max_workers=args.max_workers,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        enable_caching=args.enable_caching,
        cache_dir=args.cache_dir
    )
    
    # Step 5: Build final tiling configuration
    logger.info("Step 5: Building final tiling configuration")
    final_tiling_config = build_final_tiling_config(optimization_results)
    
    # Step 6: Save the results
    logger.info("Step 6: Saving optimization results")
    with open(f"{model_name}_tiling_optimization_results.json", 'w') as f:
        json.dump(optimization_results, f, indent=4)
    
    with open(f"{model_name}_optimal_tiling.json", 'w') as f:
        json.dump(final_tiling_config, f, indent=4)
    
    # Step 7: Compile the model with the optimal tiling configuration
    logger.info("Step 7: Compiling model with optimal tiling configuration")
    final_exp_name = "optimized_tiling"
    compile_success = compile_model(
        model_path=args.model_path,
        config_path=args.config_path,
        experiment_name=final_exp_name,  
        tiling_config=final_tiling_config, 
        max_retries=args.compile_retries
    )
    
    if compile_success:
        logger.info(f"Successfully compiled model with optimized tiling configuration")
        logger.info(f"Output directory: {model_name}_{final_exp_name}")
    else:
        logger.error("Failed to compile model with optimized tiling configuration")
    
    # Report total execution time
    execution_time = time.time() - start_time
    logger.info(f"Total optimization time: {execution_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    exit(main())