# Life_Engine\src\utils\logger_setup.py

"""
Handles the setup of the application's logging system.
"""
import logging
import sys

def setup_logging(config: dict):
    """
    Configures the root logger based on the simulation config.

    Args:
        config (dict): The 'logging' section of the config.json file.
    """
    log_level = config.get("level", "INFO").upper()
    log_format = config.get("format")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # TODO: Add file handler in a future step as per Rule 2.2

    # We return a standard LoggerAdapter. It will automatically add the
    # dictionary's contents to all log records.
    # We initialize 'step' to 'INIT' for messages logged before the loop starts.
    adapter = logging.LoggerAdapter(logging.getLogger(), {'step': 'INIT'})
    return adapter