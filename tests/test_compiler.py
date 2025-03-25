import unittest
from compiler.model_compiler import prepare_model, compile_model
from utils.logging_utils import setup_logging

# Setup logging with a proper level and name
logger = setup_logging(name="test_compiler", level="INFO")

class TestModelCompiler(unittest.TestCase):
    def setUp(self):
        # set up any necessary state befor the tests
        logger.info("Setting up the test environment...")
        self.model_name = "supported_2.onnx"
        self.model_dir = "/Users/subash/Desktop/sudachi/Research/HadisLab/GeneSys.codelets"
        self.prepared_dir = "/Users/subash/Desktop/sudachi/Research/HadisLab/genesys_optimizer/prepared_models"
        self.model_path = f"{self.model_dir}/{self.model_name}"
        # relative path
        self.prepared_model_path = f"{self.prepared_dir}/{self.model_name}"
        self.experiment_name = "test_experiment"
        self.tiling_config = {
            "tile_size": 16,
            "max_tiles": 4
        }
        self.fuse = False
        self.max_tries = 2
        # prepare the model before each test
        logger.info("Preparing the model for testing...")
        self.assertTrue(prepare_model(self.model_path, self.max_tries), f"Failed to prepare the model: {self.model_path}")
        logger.info("Model prepared successfully")

    def test_compile_model(self):
        # Test if the model compiles successfully"
        logger.info("Testing model compilation...")
        # print(f"Model path: {self.model_path}")
        # print(f"Prepared model path: {self.prepared_model_path}")
        result = compile_model(self.prepared_model_path, self.experiment_name, self.tiling_config, self.fuse, self.max_tries)
        self.assertTrue(result, "Model compilation failed")
        logger.info("Model compilation test completed")

if __name__ == "__main__":
    unittest.main()
