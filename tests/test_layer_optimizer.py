import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from optimizer.layer_optimizer import (
    optimize_layer,
    optimize_layers_parallel,
    build_final_tiling_config
)

class TestLayerOptimizer(unittest.TestCase):
    """Test the layer optimization functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = "/path/to/model.onnx"  # Mock path
        self.sim_path = "/path/to/simulator"  # Mock path
        
        # Sample layer info
        self.layer_name = "test_layer"
        self.layer_info = {
            "operation": "conv",
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
            },
            "tiling_key": "conv_1"
        }
        
        # Sample tiling configurations
        self.tiling_configs = [
            {"OC": 1, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 1},  # Baseline
            {"OC": 64, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 1}, # Max OC
            {"OC": 1, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 56, "OW": 1}, # Max OH
            {"OC": 1, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 56}  # Max OW
        ]
        
        # Sample optimization results
        self.optimization_results = {
            "layer1": {
                "best_config": {"OC": 1, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 28, "OW": 28},
                "best_metric": 1000.0,
                "tiling_key": "conv_1"
            },
            "layer2": {
                "best_config": {"OC": 32, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 1},
                "best_metric": 800.0,
                "tiling_key": "conv_2"
            }
        }
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('optimizer.layer_optimizer.generate_tiling_configs')
    @patch('optimizer.layer_optimizer.compile_model')
    @patch('optimizer.layer_optimizer.run_simulator')
    def test_optimize_layer(self, mock_simulator, mock_compile, mock_generate_configs):
        """Test optimizing a single layer."""
        # Setup mocks
        mock_generate_configs.return_value = self.tiling_configs
        mock_compile.return_value = True
        
        # Simulator returns different metrics for different configs
        metrics = [
            {"totCycles": 2000, "totTime(us)": 5.0},    # Baseline
            {"totCycles": 1500, "totTime(us)": 3.75},   # Max OC (best)
            {"totCycles": 1800, "totTime(us)": 4.5},    # Max OH
            {"totCycles": 1600, "totTime(us)": 4.0}     # Max OW
        ]
        mock_simulator.side_effect = metrics
        
        # Run optimization
        result = optimize_layer(
            model_path=self.model_path,
            layer_name=self.layer_name,
            layer_info=self.layer_info,
            output_dir=self.temp_dir,
            sim_path=self.sim_path,
            metric="totCycles",
            max_configs=4
        )
        
        # Verify the expected functions were called
        mock_generate_configs.assert_called_once_with(self.layer_info, 4)
        self.assertEqual(mock_compile.call_count, 4)
        self.assertEqual(mock_simulator.call_count, 4)
        
        # Verify correct best configuration was selected
        layer_name, best_config, best_metric, tiling_key = result
        self.assertEqual(layer_name, self.layer_name)
        self.assertEqual(best_config, self.tiling_configs[1])  # Max OC was best
        self.assertEqual(best_metric, 1500)  # Best metric
        self.assertEqual(tiling_key, "conv_1")
    
    @patch('optimizer.layer_optimizer.optimize_layer')
    def test_optimize_layers_parallel(self, mock_optimize_layer):
        """Test optimizing multiple layers in parallel."""
        # Setup mock
        mock_optimize_layer.side_effect = [
            ("layer1", {"OC": 1, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 28, "OW": 28}, 1000.0, "conv_1"),
            ("layer2", {"OC": 32, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 1}, 800.0, "conv_2")
        ]
        
        # Prepare layers info
        layers_info = [
            ("layer1", self.layer_info),
            ("layer2", self.layer_info.copy())  # Deep copy to avoid issues
        ]
        layers_info[1][1]["tiling_key"] = "conv_2"  # Change key for second layer
        
        # Run optimization
        results = optimize_layers_parallel(
            model_path=self.model_path,
            layers_info=layers_info,
            output_dir=self.temp_dir,
            sim_path=self.sim_path,
            max_configs_per_layer=4,
            checkpoint_dir=self.temp_dir,
            cache_dir=self.temp_dir
        )
        
        # Verify optimize_layer was called for each layer
        self.assertEqual(mock_optimize_layer.call_count, 2)
        
        # Verify results are as expected
        self.assertEqual(len(results), 2)
        self.assertIn("layer1", results)
        self.assertIn("layer2", results)
        self.assertEqual(results["layer1"]["best_metric"], 1000.0)
        self.assertEqual(results["layer2"]["best_metric"], 800.0)
    
    @patch('optimizer.layer_optimizer.OptimizationCache')
    @patch('optimizer.layer_optimizer.generate_tiling_configs')
    @patch('optimizer.layer_optimizer.compile_model')
    @patch('optimizer.layer_optimizer.run_simulator')
    def test_optimize_layer_with_cache(self, mock_simulator, mock_compile, 
                                    mock_generate_configs, mock_cache_class):
        """Test optimizing a layer using the cache."""
        # Setup cache mock to return a cached result
        mock_cache = MagicMock()
        mock_cache_class.return_value = mock_cache
        cached_result = {
            "best_config": {"OC": 32, "N": 1, "IC": 1, "KH": 1, "KW": 1, "OH": 1, "OW": 1},
            "best_metric": 1200.0,
            "tiling_key": "conv_1"
        }
        mock_cache.get_cached_result.return_value = cached_result
        
        # Run optimization
        result = optimize_layer(
            model_path=self.model_path,
            layer_name=self.layer_name,
            layer_info=self.layer_info,
            output_dir=self.temp_dir,
            sim_path=self.sim_path,
            metric="totCycles",
            max_configs=4,
            optimization_cache=mock_cache
        )
        
        # Verify the cache was checked
        mock_cache.get_cached_result.assert_called_once_with(self.layer_info)
        
        # Verify other functions were not called (optimization skipped)
        mock_generate_configs.assert_not_called()
        mock_compile.assert_not_called()
        mock_simulator.assert_not_called()
        
        # Verify the cached result was returned
        layer_name, best_config, best_metric, tiling_key = result
        self.assertEqual(layer_name, self.layer_name)
        self.assertEqual(best_config, cached_result["best_config"])
        self.assertEqual(best_metric, cached_result["best_metric"])
        self.assertEqual(tiling_key, cached_result["tiling_key"])
    
    def test_build_final_tiling_config(self):
        """Test building the final tiling configuration."""
        # Build the final config
        final_config = build_final_tiling_config(self.optimization_results)
        
        # Verify the structure is correct
        self.assertIn("conv_1", final_config)
        self.assertIn("conv_2", final_config)
        self.assertIn("1", final_config["conv_1"])
        self.assertIn("1", final_config["conv_2"])
        
        # Verify the tile splits are correct
        self.assertEqual(
            final_config["conv_1"]["1"], 
            self.optimization_results["layer1"]["best_config"]
        )
        self.assertEqual(
            final_config["conv_2"]["1"], 
            self.optimization_results["layer2"]["best_config"]
        )

if __name__ == '__main__':
    unittest.main()
