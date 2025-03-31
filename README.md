# GeneSys Optimizer

A tool to optimize tiling configurations for models on the GeneSys.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/genesys-optimizer.git
cd genesys-optimizer

# Go to the genesys_compiler dir and activate the virtual environment
cd ..
cd genesys_compiler
source venv/bin/activate  

# Install the package
pip install -e .
```

## Usage

```bash
# Basic usage
python -m genesys_optimizer.main --model_path /path/to/model.onnx --sim_path /path/to/simulator

# Current Usage
python3 main.py --model_path /Users/Name/GeneSys.codelets/resnet18.onnx --sim_path /Users/Name/GeneSys.sim --max_configs_per_layer 5 --checkpoint_dir my_checkpoints --enable_caching --cache_dir model_cache --max_workers 2

```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Path to the ONNX model file to optimize | (Required) |
| `--sim_path` | Path to the GeneSys simulator directory | (Required) |
| `--metric` | Performance metric to optimize (e.g., totCycles, totTime(us)) | `totCycles` |
| `--output_dir` | Directory where compilation results are stored | `genesys_compiler_output` |
| `--layers` | Specific layer names to optimize (if not specified, all layers are optimized) | (All layers) |
| `--operation_types` | Types of operations to optimize (e.g., conv, matmul) | (All operations) |
| `--max_configs_per_layer` | Maximum number of tiling configurations to test per layer | 10 |
| `--exhaustive` | Test all possible tiling configurations (equivalent to setting max_configs_per_layer to -1) | False |
| `--compile_retries` | Number of times to retry compilation if it fails | 3 |
| `--sim_retries` | Number of times to retry simulation if it fails | 2 |
| `--max_workers` | Maximum number of parallel optimization workers | (CPU count) |
| `--log_file` | File to write log messages to (if not specified, logs to console only) | (Console only) |
| `--log_level` | Verbosity of log messages (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `--checkpoint_dir` | Directory to store checkpoint files for crash recovery | `checkpoints` |
| `--checkpoint_interval` | How often to save checkpoints (in seconds) | 300 |
| `--enable_caching` | Enable caching of optimization results for similar layers | True |
| `--disable_caching` | Disable the layer similarity caching system | False |
| `--cache_dir` | Directory to store the layer cache files | `layer_cache` |
| `--clear_cache` | Clear the existing layer cache before starting optimization | False |

## Tests

Run the tests with:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test modules
python -m unittest tests.test_tiling_generator
python -m unittest tests.test_simulator
python -m unittest tests.test_layer_optimizer
python -m unittest tests.test_cache
python -m unittest tests.test_checkpoint
```

## Output Files

The optimizer produces several output files:
- `<model_name>_tiling_optimization_results.json`: Complete optimization results for each layer
- `<model_name>_optimal_tiling.json`: Final tiling configuration to use with the GeneSys compiler
- Log file (if specified with --log_file)
- Cache file (in the cache directory)
- Checkpoint files (in the checkpoint directory)