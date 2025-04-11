import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from simulator.simulator import run_simulator, parse_simulator_output, parse_layer_output_files

class TestSimulator(unittest.TestCase):
    """Test the simulator integration."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        self.layer_name = "test_layer"
        self.sim_path = "/path/to/simulator"  # Mock path
        
        # Path to the test metrics file
        self.test_metrics_file = os.path.join(
            os.path.dirname(__file__), 
            "test_metric.csv"
        )
        
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    @patch('simulator.simulator.subprocess.run')
    @patch('simulator.simulator.os.chdir')
    @patch('simulator.simulator.os.path.exists')
    @patch('simulator.simulator.parse_simulator_output')
    def test_run_simulator_success(self, mock_parse, mock_exists, mock_chdir, mock_run):
        """Test successful simulator run."""
        # Setup mocks
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(stdout=b"Simulation complete", stderr=b"")
        mock_parse.return_value = {"totCycles": 1000, "totTime(us)": 2.5}
        
        # Run the simulator
        result = run_simulator(
            output_dir=self.temp_dir,
            layer_name=self.layer_name,
            sim_path=self.sim_path
        )
        
        # Verify simulator was called correctly
        mock_chdir.assert_any_call(self.sim_path)
        mock_run.assert_called_once()
        
        # Verify parse_simulator_output was called
        mock_parse.assert_called_once()
        
        # Verify correct result was returned
        self.assertEqual(result, {"totCycles": 1000, "totTime(us)": 2.5})
    
    def test_parse_simulator_output_with_test_metrics(self):
        """Test parsing metrics from the test metrics file."""
        # Skip if test metrics file doesn't exist
        if not os.path.exists(self.test_metrics_file):
            self.skipTest(f"Test metrics file not found: {self.test_metrics_file}")
            
        # Test with different layers from the metrics file
        test_cases = [
            # Layer name, expected totCycles, expected totTime
            ("layer88_conv_bias89", 108798.0, 108.798),
            ("layer3_conv_bias4", 1731191.0, 1731.191),
            ("layer64_conv_bias65", 3336382.0, 3336.382),
            ("layer8_elem_add4d4d9", 512000.0, 512.0),
            ("layer49_elem_add4d4d50", 128000.0, 128.0),
        ]
        
        for layer_name, expected_cycles, expected_time in test_cases:
            # Test full metrics dictionary
            metrics = parse_simulator_output(self.test_metrics_file, layer_name)
            self.assertIsNotNone(metrics, f"Failed to extract metrics for {layer_name}")
            self.assertIn("totCycles", metrics, f"totCycles not found for {layer_name}")
            self.assertIn("totTime(us)", metrics, f"totTime(us) not found for {layer_name}")
            self.assertAlmostEqual(metrics["totCycles"], expected_cycles, places=1, 
                                 msg=f"Wrong totCycles value for {layer_name}")
            self.assertAlmostEqual(metrics["totTime(us)"], expected_time, places=3,
                                 msg=f"Wrong totTime(us) value for {layer_name}")
            
            # Test specific metric extraction
            cycles = parse_simulator_output(self.test_metrics_file, layer_name, "totCycles")
            self.assertEqual(cycles, expected_cycles, f"Wrong totCycles when requesting specific metric for {layer_name}")
            
            time_us = parse_simulator_output(self.test_metrics_file, layer_name, "totTime(us)")
            self.assertEqual(time_us, expected_time, f"Wrong totTime(us) when requesting specific metric for {layer_name}")
    
    def test_parse_simulator_output_nonexistent_layer(self):
        """Test parsing metrics for a layer that doesn't exist."""
        # Skip if test metrics file doesn't exist
        if not os.path.exists(self.test_metrics_file):
            self.skipTest(f"Test metrics file not found: {self.test_metrics_file}")
        
        metrics = parse_simulator_output(self.test_metrics_file, "nonexistent_layer")
        self.assertIsNone(metrics, "Should return None for nonexistent layer")
    
    def test_parse_simulator_output_case_insensitive(self):
        """Test that layer name matching is case-insensitive."""
        # Skip if test metrics file doesn't exist
        if not os.path.exists(self.test_metrics_file):
            self.skipTest(f"Test metrics file not found: {self.test_metrics_file}")
        
        # Original is "layer88_conv_bias89", try different case
        metrics = parse_simulator_output(self.test_metrics_file, "Layer88_Conv_Bias89")
        self.assertIsNotNone(metrics, "Case-insensitive matching failed")
        self.assertAlmostEqual(metrics["totCycles"], 108798.0, places=1)
    
    def test_parse_layer_output_files(self):
        """Test parsing metrics from various file formats."""
        # Create a temporary JSON file
        json_file = os.path.join(self.temp_dir, "layer_metrics.json")
        with open(json_file, 'w') as f:
            json.dump({
                "test_layer": {
                    "totCycles": 5000,
                    "totTime(us)": 10.5
                }
            }, f)
        
        # Create a temporary log file
        log_file = os.path.join(self.temp_dir, "layer_metrics.log")
        with open(log_file, 'w') as f:
            f.write("Layer test_layer: cycles=6000, time=12.5ms\n")
        
        # Test parsing from JSON
        metrics = parse_layer_output_files([json_file], "test_layer")
        self.assertIsNotNone(metrics, "Failed to parse JSON file")
        self.assertEqual(metrics["totCycles"], 5000)
        
        # Test parsing from log
        metrics = parse_layer_output_files([log_file], "test_layer")
        self.assertIsNotNone(metrics, "Failed to parse log file")
        self.assertEqual(metrics["totCycles"], 6000)

if __name__ == '__main__':
    unittest.main()
