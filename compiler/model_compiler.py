import os
import json
import time
import subprocess
from utils.logging_utils import setup_logging

logger = setup_logging("genesys_optimizer.compiler")

def prepare_model(model_path, max_retries = 0) -> bool:
    """Prepare the model for compilation.

    Args:
        model_path (_type_): _description_
        max_tries (int, optional): _description_. Defaults to 0.
    """
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False

    cmd = ["prepare-model", "-m", model_path]

    for i in range(max_retries):
        try:
            logger.info(f"Preparing the model: {model_path} (try {i +1} of {max_retries})")
            logger.info(f"command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            logger.info("Model preparation completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while preparing the model: {e.stderr}")
            if i < max_retries - 1:
                logger.info(f"Retrying preparation of the model: {model_path} (try {i + 2} of {max_retries})")
            else:
                logger.error(f"Failed to prepare the model after {max_retries} attempts.")
                return False
    return False


def compile_model(model_path, config_path, experiment_name, tiling_config=None, fuse=False, max_retries = 1) -> bool:
    """Compile the model after the model has been prepared for compilation.

    Args:
        model_path (_type_): _description_
        max_tries (int, optional): _description_. Defaults to 1.
    """
    model_path = os.path.abspath(model_path)
    config_path = os.path.abspath(config_path) if config_path else None

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False
    
    if config_path:
        cmd = ["compile-genesys", "-m", model_path, "-c", config_path, "-e", experiment_name]
    else:
        cmd = ["compile-genesys", "-m", model_path, "-e", experiment_name]
    
    tiling_file_path = None
    # Create an absolute path to tiling_config directory
    tiling_config_dir = os.path.abspath("tiling_config")
    
    # check if tiling_config dir exists if not create it
    if not os.path.exists(tiling_config_dir):
        os.makedirs(tiling_config_dir)

    if tiling_config:
        # Use absolute path for the tiling config file
        tiling_file_name = f"tiling_{experiment_name}.json"
        tiling_file_path = os.path.join(tiling_config_dir, tiling_file_name)
        
        with open(tiling_file_path, "w") as f:
            json.dump(tiling_config, f, indent=4)
        
        # Use absolute path in the command
        cmd.extend(["-t", tiling_file_path])
    

    if fuse:
        # if fuse is true add -f to the command
        cmd.append("-f")
    
    for i in range(max_retries):
        try:
            logger.info(f"Compiling the model: {model_path} (try {i +1} of {max_retries})")
            logger.info(f"command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            logger.info("Model compilation completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while compiling the model: {e.stderr}")
            if i < max_retries - 1:
                logger.info(f"Retrying compilation of the model: {model_path} (try {i + 2} of {max_retries})")
            else:
                logger.error(f" Failed to compile the model after {max_retries} attempts.")
                return False
    return False

