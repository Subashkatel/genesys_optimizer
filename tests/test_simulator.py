import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from simulator.simulator import run_simulator, parse_simulator_output

class TestSimulator(unittest.TestCase):
    """Test the simulator integration."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        self.layer_name = "test_layer"
        self.sim_path = "/path/to/simulator"  # Mock path
        
        # Sample CSV data that the simulator might produce
        self.sample_csv_data = [
            ["index", "layerName", "totCycles", "totTime(us)", "utilization"],
            ["0", "test_layer", "1200", "8.5", "0.75"]
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
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
    
    def test_parse_simulator_output_valid_csv(self):
        """Test parsing a valid simulator output CSV."""
        # Create a temporary CSV file
        csv_path = os.path.join(self.temp_dir, "simulation_results.csv")
        with open(csv_path, 'w') as f:
            for row in self.sample_csv_data:
                f.write(','.join(row) + '\n')
        
        # Parse the CSV
        result = parse_simulator_output(csv_path, self.layer_name)
        
        # Verify correct metrics were extracted
        self.assertEqual(result, {"totCycles": 1200.0, "totTime(us)": 8.5})

if __name__ == '__main__':
    unittest.main()
