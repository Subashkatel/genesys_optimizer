import unittest
import tempfile
import csv
from unittest.mock import patch, MagicMock, mock_open
from simulator.simulator import run_simulator, parse_simulation_output
from utils.logging_utils import setup_logging
import os

logger = setup_logging(name="test_simulator", level="INFO")

class TestSimulator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.sample_csv_content = (
            "some header row\n"
            "another header row\n"
            "layerName,totCycles,totTime(us),otherMetric\n"
            "layer1,1000,2.5,100\n"
            "layer2,2000,5.0,200\n"
        )

        @patch("simulator.simulator.subprocess.run")
        @patch("simulator.simulator.os.chdir")
        @patch("simulator.simulator.os.path.exists")
        def test_run_simulator_success(self, mock_exists, mock_chdir, mock_run):
            # csv file check
            mock_exists.return_value = True

            # Mock the subprocess
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run_return_value = mock_process

            # patch for parse_simulator_output
            with patch("simulator.simulator.parse_simulator_output") as mock_parse:
                mock_parse.return_value = {"totCycles": 1000, "totTime(us)": 2.5}

                # running the simulator 
                result = run_simulator(
                    output_dir="test_output",
                    layer_name="layer1",
                    sim_path="/path/to/simulator",
                    max_retries=1
                )

                # check calls 

                mock_chdir.assert_called_once_with("/path/to/simulator")
                mock_run.assert_called_once()
                mock_parse.assert_called_once()

                # check the result
                self.asserEqual(result, {"totCycles": 1000, "totTime(us)": 2.5})
