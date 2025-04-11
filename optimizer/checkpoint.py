"""
Checkpoint management system for saving and restoring optimizer state.
This allows recovery after interruptions and resuming long-running optimizations.
"""

import os
import json
import time
import threading
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.checkpoint")

class CheckpointManager:
    """
    Manages saving and loading optimization checkpoints.
    
    Provides periodic saving of checkpoint files to allow resuming optimization
    if the process is interrupted.
    """
    
    def __init__(self, model_name, checkpoint_dir="checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            model_name: Name of the model being optimized
            checkpoint_dir: Directory to store checkpoint files
        """
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.json")
        self.layer_results = {}
        self.save_interval = 60  # Default: 1 minute (60 seconds) - more frequent than before
        self.last_save_time = time.time()
        self.save_timer = None
        self.lock = threading.Lock()
        self.saves_count = 0  # Track how many saves have been performed
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Also create a backup directory for rotated checkpoints
        self.backup_dir = os.path.join(checkpoint_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def set_save_interval(self, interval_seconds):
        """Set the interval between automatic checkpoint saves."""
        self.save_interval = interval_seconds
        logger.info(f"Checkpoint save interval set to {interval_seconds} seconds")
        
    def update_layer_result(self, layer_name, result):
        """
        Update the result for a given layer and save immediately.
        
        Args:
            layer_name: Name of the layer
            result: Optimization result for the layer
        """
        with self.lock:
            self.layer_results[layer_name] = result
            
            # Save on every layer completion for maximum safety
            self.save_checkpoint()
    
    def _rotate_checkpoints(self):
        """Rotate checkpoint files to maintain a history."""
        # Maximum number of backups to keep
        max_backups = 5
        
        # Create a backup with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"{self.model_name}_checkpoint_{timestamp}.json")
        
        try:
            if os.path.exists(self.checkpoint_file):
                # Copy current checkpoint to backup
                with open(self.checkpoint_file, 'r') as src:
                    with open(backup_file, 'w') as dst:
                        dst.write(src.read())
                
                # Get list of backup files and keep only the most recent
                backup_files = sorted([
                    os.path.join(self.backup_dir, f) for f in os.listdir(self.backup_dir)
                    if f.startswith(f"{self.model_name}_checkpoint_")
                ])
                
                # Remove oldest backups if we have too many
                while len(backup_files) > max_backups:
                    oldest = backup_files.pop(0)
                    os.remove(oldest)
        except Exception as e:
            logger.warning(f"Failed to rotate checkpoint backups: {str(e)}")
    
    def save_checkpoint(self, force=False):
        """
        Save the current optimization state to a checkpoint file.
        
        Args:
            force: If True, save immediately regardless of interval timer
            
        Returns:
            True if save was successful, False otherwise
        """
        with self.lock:
            # Only save if we have results to save or force is True
            if not self.layer_results and not force:
                return True
                
            current_time = time.time()
            elapsed = current_time - self.last_save_time
            
            # Check if enough time has passed since last save or force is True
            if force or elapsed >= self.save_interval:
                try:
                    # Every 10 saves, rotate the checkpoint files
                    if self.saves_count % 10 == 0:
                        self._rotate_checkpoints()
                    
                    # Ensure checkpoint directory exists
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    
                    # Checkpoint data structure
                    checkpoint_data = {
                        "model_name": self.model_name,
                        "timestamp": current_time,
                        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "completed_layers": len(self.layer_results),
                        "layer_results": self.layer_results
                    }
                    
                    # Write to temporary file first to avoid corrupting existing checkpoint
                    temp_file = f"{self.checkpoint_file}.tmp"
                    with open(temp_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                    
                    # Replace old checkpoint with new one (atomic operation on most filesystems)
                    os.replace(temp_file, self.checkpoint_file)
                    
                    self.last_save_time = current_time
                    self.saves_count += 1
                    
                    logger.info(f"Progress checkpoint saved with {len(self.layer_results)} completed layers")
                    return True
                except (IOError, OSError) as e:
                    logger.error(f"Failed to save checkpoint: {str(e)}")
                    return False
            
            return True
    
    def load_checkpoint(self):
        """
        Load the latest checkpoint if available.
        
        Returns:
            Dictionary of layer results from checkpoint, or empty dict if no checkpoint
        """
        if not os.path.exists(self.checkpoint_file):
            logger.info("No checkpoint file found. Starting fresh optimization.")
            return {}
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                
            self.layer_results = checkpoint_data.get("layer_results", {})
            logger.info(f"Loaded checkpoint with {len(self.layer_results)} completed layers from {checkpoint_data.get('date', 'unknown date')}")
            return self.layer_results
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint file: {str(e)}")
            return {}
        
    def clear_checkpoint(self):
        """
        Clear the checkpoint file.
        
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("Checkpoint file cleared")
                return True
            except OSError as e:
                logger.error(f"Failed to delete checkpoint file: {str(e)}")
                return False
        return True