import os
import json
import time
import subprocess
import logging
import sys

logger = logging.getLogger("genesys_optimizer.compiler")

def prepare_model(model_path, max_tries = 1) -> bool:
    """Prepare the model for compilation.

    Args:
        model_path (_type_): _description_
        max_tries (int, optional): _description_. Defaults to 0.
    """

    cmd = ["prepare-model", "-m", model_path]

    for i in range(max_tries):
        try:
            logger.info(f"Preparing the model: {model_path} (try {i +1} of {max_tries})")
            logger.info(f"command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            logger.info("Model preparation completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while preparing the model: {e.stderr}")
            if i < max_tries - 1:
                logger.info(f"Retrying preparation of the model: {model_path} (try {i + 2} of {max_tries})")
            else:
                logger.error(f"Failed to prepare the model after {max_tries} attempts.")
                return False
    return False


def compile_model(model_path, experiment_name, tiling_config=None, fuse=False, max_tries = 1) -> bool:
    """Compile the model after the model has been prepared for compilation.

    Args:
        model_path (_type_): _description_
        max_tries (int, optional): _description_. Defaults to 1.
    """

    cmd = ["compile-genesys", "-m", model_path, "-e", experiment_name]
    
    config_path = None
    # check if tilling_config dir exists if not create it
    if not os.path.exists("tiling_config"):
        os.makedirs("tiling_config")

    if tiling_config:
        # go inside the tiling_config directory to create the config file
        config_path = os.path.join("tiling_config", f"tiling_{experiment_name}.json")
        with open(config_path, "w") as f:
            json.dump(tiling_config, f, indent=4)
        cmd.extend(["-t", config_path])
    

    if fuse:
        # if fuse is true add -f to the command
        cmd.append("-f")
    
    for i in range(max_tries):
        try:
            logger.info(f"Compiling the model: {model_path} (try {i +1} of {max_tries})")
            logger.info(f"command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            logger.info("Model compilation completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while compiling the model: {e.stderr}")
            if i < max_tries - 1:
                logger.info(f"Retrying compilation of the model: {model_path} (try {i + 2} of {max_tries})")
            else:
                logger.error(f" Failed to compile the model after {max_tries} attempts.")
                return False
    return False

