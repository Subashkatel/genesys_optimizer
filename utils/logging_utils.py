import logging
import os
import sys

# Track which loggers have been configured
_configured_loggers = set()

def setup_logging(logger_name="genesys_optimizer", log_file=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        logger_name: Name of the logger
        log_file: Path to log file (if None, log to console only)
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Check if this logger has already been configured
    if logger_name in _configured_loggers:
        return logging.getLogger(logger_name)
    
    # Add to configured loggers set
    _configured_loggers.add(logger_name)
    
    # Get logger
    logger = logging.getLogger(logger_name)
    
    # Only configure if the logger doesn't already have handlers
    if not logger.handlers:
        # Set logging level
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log file is specified
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to the root logger to avoid duplicate logs
        logger.propagate = False
    
    return logger

def get_logger(name):
    """
    Get a logger by name. If the logger doesn't exist, it will be created.
    
    Args:
        name: Logger name
        
    Returns:
        Logger object
    """
    if name in _configured_loggers:
        return logging.getLogger(name)
    else:
        # For modules that need a logger but don't need to configure it
        # This ensures we track this logger in our _configured_loggers set
        _configured_loggers.add(name)
        logger = logging.getLogger(name)
        logger.propagate = False
        return logger

def reset_logging_config():
    """
    Reset all logging configuration.
    Useful for testing or when you need to reconfigure logging.
    """
    # Reset our tracking set
    _configured_loggers.clear()
    
    # Reset the logging module's internal state
    logging.shutdown()
    logging.root.handlers = []
    
    # Clear handlers from all loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)