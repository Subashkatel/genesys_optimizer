import unittest
from compiler.layer_extractor import extract_layers_info, get_all_layers
from utils.logging_utils import setup_logging

logger = setup_logging(name="test_layer_extractor", level="INFO")
class TestLayerExtractor(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up the test environment...")
        self.output_dir = "/Users/subash/Desktop/sudachi/Research/HadisLab/genesys_optimizer/genesys_compiler_output/resnet18_genesys16x16_test_experiment"
        self.layer_name = "layer0_conv_bias1"

        self.expected_output = {
            "operation": "conv_bias",
            "instance_id": 1,
            "tile_splits": {
                "OC": 1,
                "N": 1,
                "IC": 1,
                "KH": 1,
                "KW": 1,
                "OH": 112,
                "OW": 2
            },
            "iterable_dimensions": {
                "OC": 64,
                "N": 1,
                "IC": 64,
                "KH": 7,
                "KW": 7,
                "OH": 112,
                "OW": 112
            },
            "tiling_key": "conv_bias_1"
        }

    def test_extract_layers_info(self):
        logger.info("Testing layer extraction...")
        result = extract_layers_info(self.output_dir, self.layer_name)
        layers = get_all_layers(self.output_dir)
        logger.info(f"Extracted layer info: {result}")
        self.assertEqual(result.get("operation"), self.expected_output["operation"])
        self.assertEqual(result.get("instance_id"), self.expected_output["instance_id"])
        self.assertEqual(result.get("tile_splits"), self.expected_output["tile_splits"])
        self.assertEqual(result.get("iterable_dimensions"), self.expected_output["iterable_dimensions"])
        self.assertEqual(result.get("tiling_key"), self.expected_output["tiling_key"])
        logger.info("Layer extraction test completed")
        
        self.assertEqual(len(layers), 48)
        logger.info(f"Total layers extracted: {len(layers)}")



if __name__ == "__main__":
    unittest.main()
    