"""Checkpoint system for saving optimization progress."""

import os
import json
import logging
import time
from datetime import datetime
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.checkpoint")

class CheckpointManager:
    """
    Manages checkpoints for the optimization process to enable recovery from crashes.
    This is a simple system that saves the current state of optimization periodically.
    """
    
    def __init__(self, model_name, output_dir="checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            model_name: Name of the model being optimized
            output_dir: Directory to store checkpoint files
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.checkpoint_file = os.path.join(output_dir, f"{model_name}_checkpoint.json")
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes by default
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def save_checkpoint(self, force=False):
        """
        Save the current optimization results to the checkpoint file.
        
        Args:
            force: Force saving regardless of time interval
            
        Returns:
            True if the checkpoint was saved, False otherwise
        """
        current_time = time.time()
        
        # Only save if forced or if enough time has passed since last save
        if force or (current_time - self.last_save_time > self.save_interval):
            try:
                # Ensure the output directory exists
                os.makedirs(self.output_dir, exist_ok=True)
                
                checkpoint_data = {
                    "model_name": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "results": self.results
                }
                
                # Save to a temporary file first, then rename to avoid corruption
                temp_file = f"{self.checkpoint_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                # Atomic replacement of the checkpoint file
                os.replace(temp_file, self.checkpoint_file)
                
                self.last_save_time = current_time
                logger.info(f"Progress checkpoint saved with {len(self.results)} completed layers")
                return True
            except (IOError, OSError) as e:
                logger.error(f"Failed to save checkpoint: {str(e)}")
                return False
        
        return False
    
    def has_layer_result(self, layer_name):
        """
        Check if a layer has already been optimized.
        
        Args:
            layer_name: Name of the layer to check
            
        Returns:
            True if the layer has already been optimized, False otherwise
        """
        return layer_name in self.results
    
    def get_layer_result(self, layer_name):
        """
        Get the optimization result for a layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            The optimization result or None if not found
        """
        return self.results.get(layer_name)
    
    def update_layer_result(self, layer_name, result):
        """
        Update the optimization result for a layer and save the checkpoint if needed.
        
        Args:
            layer_name: Name of the layer
            result: Optimization result to save
            
        Returns:
            True if the checkpoint was saved, False otherwise
        """
        self.results[layer_name] = result
        # Force save in tests, but in normal operation will check save_interval
        return self.save_checkpoint(force=True)
    
    def get_all_results(self):
        """
        Get all optimization results.
        
        Returns:
            Dictionary of all layer optimization results
        """
        return self.results.copy()
    
    def set_save_interval(self, seconds):
        """
        Set the interval between automatic checkpoint saves.
        
        Args:
            seconds: Interval in seconds between saves
        """
        self.save_interval = seconds