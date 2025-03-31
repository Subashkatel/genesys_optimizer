import os
import json
import shutil
import unittest
from optimizer.cache import OptimizationCache

class TestOptimizationCache(unittest.TestCase):
    """Test the OptimizationCache functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_cache")
        # Clear test directory if it exists
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.model_name = "test_model"
        self.cache = OptimizationCache(self.model_name, self.test_dir)
    
    def tearDown(self):
        """Clean up after the test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_cache_file_creation(self):
        """Test if the cache file is created."""
        # Add something to the cache
        layer_info = {
            "operation": "conv",
            "instance_id": 1,
            "dimensions": {"H": 224, "W": 224, "C": 3}
        }
        result = {
            "best_config": {"H": 32, "W": 32, "C": 1},
            "best_metric": 100.0,
            "tiling_key": "conv_1"
        }
        
        self.cache.add_to_cache(layer_info, result)
        
        # Check if the file exists
        cache_file = os.path.join(self.test_dir, f"{self.model_name}_layer_cache.json")
        self.assertTrue(os.path.exists(cache_file), 
                        f"Cache file was not created at {cache_file}")
        
        # Check file content
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            
        # The cache should not be empty
        self.assertGreater(len(cache_data), 0, "Cache file was created but is empty")
        
        # Check file permissions
        self.assertTrue(os.access(cache_file, os.R_OK), 
                        "Cache file is not readable")
        self.assertTrue(os.access(cache_file, os.W_OK), 
                        "Cache file is not writable")
    
    def test_add_and_retrieve_from_cache(self):
        """Test adding and retrieving items from the cache."""
        # Create a layer info
        layer_info = {
            "operation": "conv",
            "instance_id": 1,
            "dimensions": {"H": 224, "W": 224, "C": 3}
        }
        
        # Create a result
        result = {
            "best_config": {"H": 32, "W": 32, "C": 1},
            "best_metric": 100.0,
            "tiling_key": "conv_1"
        }
        
        # Add to cache
        self.cache.add_to_cache(layer_info, result)
        
        # Retrieve from cache
        cached_result = self.cache.get_cached_result(layer_info)
        
        # Check if the retrieved result matches the original
        self.assertIsNotNone(cached_result, "Failed to retrieve result from cache")
        self.assertEqual(cached_result, result, 
                         "Retrieved result doesn't match the original")
    
    def test_fingerprinting(self):
        """Test that layers with the same structure get the same fingerprint."""
        # Create two layers with the same structure but different names
        layer_info1 = {
            "operation": "conv",
            "instance_id": 1,
            "dimensions": {"H": 224, "W": 224, "C": 3}
        }
        
        layer_info2 = {
            "operation": "conv",
            "instance_id": 2,  # Different instance ID
            "dimensions": {"H": 224, "W": 224, "C": 3}  # Same dimensions
        }
        
        # Get fingerprints
        fingerprint1 = self.cache.generate_layer_fingerprint(layer_info1)
        fingerprint2 = self.cache.generate_layer_fingerprint(layer_info2)
        
        # They should be the same due to how fingerprinting works (we exclude instance_id)
        self.assertEqual(fingerprint1, fingerprint2, 
                         "Fingerprints for layers with same structure don't match")
        
        # Now create a different layer
        layer_info3 = {
            "operation": "conv",
            "instance_id": 3,
            "dimensions": {"H": 112, "W": 112, "C": 64}  # Different dimensions
        }
        
        fingerprint3 = self.cache.generate_layer_fingerprint(layer_info3)
        
        # This should have a different fingerprint
        self.assertNotEqual(fingerprint1, fingerprint3,
                           "Fingerprints for different layers shouldn't match")
    
    def test_cache_loading(self):
        """Test that cache is loaded from disk correctly."""
        # Add something to the cache
        layer_info = {
            "operation": "conv",
            "instance_id": 1,
            "dimensions": {"H": 224, "W": 224, "C": 3}
        }
        
        result = {
            "best_config": {"H": 32, "W": 32, "C": 1},
            "best_metric": 100.0,
            "tiling_key": "conv_1"
        }
        
        self.cache.add_to_cache(layer_info, result)
        
        # Create a new cache instance (should load from disk)
        new_cache = OptimizationCache(self.model_name, self.test_dir)
        
        # Try to retrieve the previously cached result
        cached_result = new_cache.get_cached_result(layer_info)
        
        # Check if the retrieved result matches the original
        self.assertIsNotNone(cached_result, "Failed to load and retrieve result from cache file")
        self.assertEqual(cached_result, result, 
                         "Retrieved result from loaded cache doesn't match the original")
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add something to the cache
        layer_info = {
            "operation": "conv",
            "instance_id": 1,
            "dimensions": {"H": 224, "W": 224, "C": 3}
        }
        
        result = {
            "best_config": {"H": 32, "W": 32, "C": 1},
            "best_metric": 100.0,
            "tiling_key": "conv_1"
        }
        
        self.cache.add_to_cache(layer_info, result)
        
        # Verify it's in the cache
        self.assertIsNotNone(self.cache.get_cached_result(layer_info),
                            "Item not found in cache before clearing")
        
        # Clear the cache
        self.cache.clear_cache()
        
        # Verify the item is no longer in the cache
        self.assertIsNone(self.cache.get_cached_result(layer_info),
                         "Item still found in cache after clearing")
        
        # Verify the cache file is gone
        cache_file = os.path.join(self.test_dir, f"{self.model_name}_layer_cache.json")
        self.assertFalse(os.path.exists(cache_file),
                        "Cache file still exists after clearing")
    
    def test_similar_layers_in_different_models(self):
        """Test that similar layers in different models are cached separately."""
        layer_info = {
            "operation": "conv",
            "instance_id": 1,
            "dimensions": {"H": 224, "W": 224, "C": 3}
        }
        
        result = {
            "best_config": {"H": 32, "W": 32, "C": 1},
            "best_metric": 100.0,
            "tiling_key": "conv_1"
        }
        
        # Add to first model's cache
        self.cache.add_to_cache(layer_info, result)
        
        # Create a cache for a different model
        model2_cache = OptimizationCache("different_model", self.test_dir)
        
        # Same layer info should not be found in the second model's cache
        self.assertIsNone(model2_cache.get_cached_result(layer_info),
                         "Layer incorrectly found in different model's cache")
        
        # Add to second model's cache with different result
        result2 = {
            "best_config": {"H": 16, "W": 16, "C": 1},
            "best_metric": 200.0,
            "tiling_key": "conv_1"
        }
        model2_cache.add_to_cache(layer_info, result2)
        
        # Verify each cache has correct, different results
        self.assertEqual(self.cache.get_cached_result(layer_info), result,
                        "First model's cache has incorrect result")
        self.assertEqual(model2_cache.get_cached_result(layer_info), result2,
                        "Second model's cache has incorrect result")

    def test_file_permissions(self):
        """Test if there are any file permission issues."""
        import tempfile
        
        # Create a temporary directory with known permissions
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Try creating a cache with this directory
                temp_cache = OptimizationCache(self.model_name, temp_dir)
                
                # Try adding to cache
                layer_info = {
                    "operation": "conv",
                    "instance_id": 1,
                    "dimensions": {"H": 224, "W": 224, "C": 3}
                }
                
                result = {
                    "best_config": {"H": 32, "W": 32, "C": 1},
                    "best_metric": 100.0,
                    "tiling_key": "conv_1"
                }
                
                success = temp_cache.add_to_cache(layer_info, result)
                
                # Check the result
                self.assertTrue(success, "Failed to add to cache in temporary directory")
                
                # Check if file exists
                cache_file = os.path.join(temp_dir, f"{self.model_name}_layer_cache.json")
                self.assertTrue(os.path.exists(cache_file), 
                                "Cache file not created in temporary directory")
                
                # Print file info for debugging
                file_stat = os.stat(cache_file)
                print(f"File permissions: {oct(file_stat.st_mode)}")
                print(f"File owner: {file_stat.st_uid}")
                print(f"Current user: {os.getuid()}")
            except Exception as e:
                self.fail(f"Exception during permission test: {str(e)}")

if __name__ == '__main__':
    unittest.main()
