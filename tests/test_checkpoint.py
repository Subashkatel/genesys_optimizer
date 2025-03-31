import os
import json
import shutil
import unittest
import time
from optimizer.checkpoint import CheckpointManager

class TestCheckpointManager(unittest.TestCase):
    """Test the CheckpointManager functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_checkpoints")
        # Clear test directory if it exists
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.model_name = "test_model"
        self.checkpoint_manager = CheckpointManager(self.model_name, self.test_dir)
    
    def tearDown(self):
        """Clean up after the test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_checkpoint_file_creation(self):
        """Test if the checkpoint file is created."""
        # Force save a checkpoint
        self.checkpoint_manager.save_checkpoint(force=True)
        
        # Check if the file exists
        checkpoint_file = os.path.join(self.test_dir, f"{self.model_name}_checkpoint.json")
        self.assertTrue(os.path.exists(checkpoint_file), 
                        f"Checkpoint file was not created at {checkpoint_file}")
        
        # Check file permissions
        self.assertTrue(os.access(checkpoint_file, os.R_OK), 
                        "Checkpoint file is not readable")
        self.assertTrue(os.access(checkpoint_file, os.W_OK), 
                        "Checkpoint file is not writable")
    
    def test_update_and_retrieve_results(self):
        """Test updating and retrieving layer results."""
        layer_name = "test_layer"
        result = {
            "best_config": {"dim1": 2, "dim2": 3},
            "best_metric": 123.45,
            "tiling_key": "conv_1"
        }
        
        # Update the result
        self.checkpoint_manager.update_layer_result(layer_name, result)
        
        # Check if the layer exists in results
        self.assertTrue(self.checkpoint_manager.has_layer_result(layer_name),
                        f"Layer {layer_name} not found in results")
        
        # Retrieve and verify the result
        retrieved_result = self.checkpoint_manager.get_layer_result(layer_name)
        self.assertEqual(retrieved_result, result,
                         "Retrieved result does not match the original")
        
        # Verify the checkpoint file exists and contains the data
        checkpoint_file = os.path.join(self.test_dir, f"{self.model_name}_checkpoint.json")
        self.assertTrue(os.path.exists(checkpoint_file), 
                        "Checkpoint file was not created")
        
        # Verify file content
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            self.assertIn("results", data, "Results key not found in checkpoint data")
            self.assertIn(layer_name, data["results"], 
                          f"Layer {layer_name} not found in checkpoint data")
            self.assertEqual(data["results"][layer_name], result, 
                             "Saved result does not match the original")
    
    def test_get_all_results(self):
        """Test getting all optimization results."""
        # Add multiple results
        results = {
            "layer1": {
                "best_config": {"dim1": 2, "dim2": 3},
                "best_metric": 123.45,
                "tiling_key": "conv_1"
            },
            "layer2": {
                "best_config": {"dim1": 4, "dim2": 5},
                "best_metric": 678.9,
                "tiling_key": "conv_2"
            }
        }
        
        for layer_name, result in results.items():
            self.checkpoint_manager.update_layer_result(layer_name, result)
        
        # Get all results and verify
        all_results = self.checkpoint_manager.get_all_results()
        self.assertEqual(len(all_results), len(results),
                         "Number of results doesn't match expected")
        
        for layer_name, result in results.items():
            self.assertIn(layer_name, all_results,
                          f"Layer {layer_name} not found in all results")
            self.assertEqual(all_results[layer_name], result,
                             f"Result for {layer_name} doesn't match expected")
    
    def test_save_interval(self):
        """Test the automatic save interval functionality."""
        # Set a short save interval for testing
        self.checkpoint_manager.set_save_interval(1)  # 1 second
        
        # Add a result but don't force save - update_layer_result now forces save
        layer_name = "test_layer"
        result = {
            "best_config": {"dim1": 2, "dim2": 3},
            "best_metric": 123.45,
            "tiling_key": "conv_1"
        }
        
        self.checkpoint_manager.update_layer_result(layer_name, result)
        
        # Check if file exists (should be created because update_layer_result forces save)
        checkpoint_file = os.path.join(self.test_dir, f"{self.model_name}_checkpoint.json")
        self.assertTrue(os.path.exists(checkpoint_file), 
                        "Checkpoint should have been saved")
        
        # Wait for the save interval to pass
        time.sleep(1.5)
        
        # Add another result
        layer_name2 = "test_layer2"
        result2 = {
            "best_config": {"dim1": 4, "dim2": 5},
            "best_metric": 678.9,
            "tiling_key": "conv_2"
        }
        
        self.checkpoint_manager.update_layer_result(layer_name2, result2)
        
        # Verify the file exists and has the latest data
        self.assertTrue(os.path.exists(checkpoint_file), 
                        "Checkpoint file should exist after interval-based save")
        
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            self.assertIn(layer_name2, data["results"], 
                          "Second layer not found in saved checkpoint")
    
    def test_file_permissions(self):
        """Test if there are any file permission issues."""
        import tempfile
        
        # Create a temporary directory with known permissions
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Try creating a checkpoint manager with this directory
                temp_manager = CheckpointManager(self.model_name, temp_dir)
                
                # Try saving a checkpoint
                result = temp_manager.save_checkpoint(force=True)
                
                # Check the result
                self.assertTrue(result, "Failed to save checkpoint in temporary directory")
                
                # Check if file exists
                checkpoint_file = os.path.join(temp_dir, f"{self.model_name}_checkpoint.json")
                self.assertTrue(os.path.exists(checkpoint_file), 
                                "Checkpoint file not created in temporary directory")
                
                # Print file info for debugging
                file_stat = os.stat(checkpoint_file)
                print(f"File permissions: {oct(file_stat.st_mode)}")
                print(f"File owner: {file_stat.st_uid}")
                print(f"Current user: {os.getuid()}")
            except Exception as e:
                self.fail(f"Exception during permission test: {str(e)}")

if __name__ == '__main__':
    unittest.main()
