import unittest
from optimizer.tiling_generator import (
    generate_all_tiling_configs,
    generate_tiling_configs,
    select_dimensions_to_optimize
)

class TestTilingGenerator(unittest.TestCase):
    """Test the tiling configuration generator."""
    
    def setUp(self):
        """Set up test data."""
        # Simple layer info for testing
        self.simple_layer = {
            "operation": "conv",
            "instance_id": 1,
            "dimensions": {
                "H": 16,
                "W": 16,
                "C": 3
            },
            "current_tile_splits": {
                "H": 1,
                "W": 1,
                "C": 1
            }
        }
        
        # More complex layer with more dimensions
        self.complex_layer = {
            "operation": "conv_bias",
            "instance_id": 1,
            "dimensions": {
                "OC": 64,
                "N": 1,
                "IC": 32,
                "KH": 3,
                "KW": 3,
                "OH": 56,
                "OW": 56
            },
            "current_tile_splits": {
                "OC": 1,
                "N": 1,
                "IC": 1,
                "KH": 1,
                "KW": 1,
                "OH": 14,
                "OW": 14
            }
        }
    
    def test_generate_all_tiling_configs(self):
        """Test generating all possible tiling configurations."""
        # Test with simple layer
        configs = generate_all_tiling_configs(self.simple_layer)
        
        # Calculate expected number of configs: 
        # H has factors [1, 2, 4, 8, 16] = 5 options
        # W has factors [1, 2, 4, 8, 16] = 5 options
        # C has factors [1, 3] = 2 options
        # Total: 5 * 5 * 2 = 50 configurations
        self.assertEqual(len(configs), 50, "Wrong number of configurations generated")
        
        # Verify some specific configurations exist
        expected_configs = [
            {"H": 1, "W": 1, "C": 1},  # No tiling
            {"H": 16, "W": 16, "C": 3},  # Full tiling
            {"H": 4, "W": 4, "C": 1}   # Partial tiling
        ]
        
        for config in expected_configs:
            self.assertIn(config, configs, f"Expected configuration {config} not found")
    
    def test_generate_tiling_configs_with_limit(self):
        """Test generating a limited number of configurations."""
        # Test with complex layer
        max_configs = 10
        configs = generate_tiling_configs(self.complex_layer, max_configs)
        
        # Verify we didn't exceed the limit
        self.assertLessEqual(len(configs), max_configs, 
                             f"Generated more than {max_configs} configurations")
        
        # Verify all configurations have valid dimensions
        dimensions = self.complex_layer["dimensions"].keys()
        for config in configs:
            # Each config should have the same dimensions as the layer
            self.assertEqual(set(config.keys()), set(dimensions), 
                             "Generated config has incorrect dimensions")
    
    def test_select_dimensions_to_optimize(self):
        """Test selection of dimensions to optimize."""
        # Test with already tiled dimensions
        tiled_dims = ["OH", "OW"]
        selected = select_dimensions_to_optimize(self.complex_layer["dimensions"], tiled_dims)
        
        # Should prioritize already tiled dimensions
        self.assertEqual(selected, tiled_dims, 
                         "Failed to prioritize already tiled dimensions")
        
        # Test without tiled dimensions - should choose largest dimensions
        selected = select_dimensions_to_optimize(self.complex_layer["dimensions"], [])
        
        # Should not exceed max_dims (default is 3)
        self.assertLessEqual(len(selected), 3, "Selected too many dimensions")

    def test_all_configurations(self):
        """Test generating all configurations for a complex layer."""
        configs = generate_all_tiling_configs(self.complex_layer)
        
        # Verify we have a reasonable number of configurations
        self.assertGreater(len(configs), 0, "No configurations generated")
        
        # Check that each config has the correct keys
        for config in configs:
            self.assertIn("OC", config)
            self.assertIn("N", config)
            self.assertIn("IC", config)
            self.assertIn("KH", config)
            self.assertIn("KW", config)
            self.assertIn("OH", config)
            self.assertIn("OW", config)

if __name__ == '__main__':
    unittest.main()
