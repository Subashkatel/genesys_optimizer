import logging
import sys

def setup_logging(name="genesys_optimizer", level=logging.INFO, log_file=None):
    """Setting up a logger with proper formating

    Args:
        name (str, optional): _description_. Defaults to "genesys_optimizer".
        level (_type_, optional): _description_. Defaults to logging.INFO.
        log_file (_type_, optional): _description_. Defaults to None.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # setup console handler - ch
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # setup file handler - fh
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger