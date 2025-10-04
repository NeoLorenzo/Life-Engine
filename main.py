# Life_Engine\main.py

"""
Main entry point for the Life-Engine simulation.

This script handles:
1. Loading simulation parameters from config.json.
2. Setting up the logging system.
3. Initializing and running the main simulation engine.
"""
import json
import logging
import numpy as np
import random

from src.utils.logger_setup import setup_logging
from src.engine import Simulation
from src.renderer import Renderer

def main():
    """Loads config, sets up logging, and starts the simulation."""
    # 1. Load Configuration (Rule 1.2)
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create one.")
        return

    # 2. Setup Logging (Rule 2)
    logger = setup_logging(config["logging"])

    # 3. Control Randomness (Rule 12)
    seed = config["simulation"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Master seed set to {seed}.")

    # 4. Initialize and Run Simulation
    renderer = Renderer()
    try:
        sim = Simulation(config, logger, renderer)
        sim.run()
    except Exception as e:
        # Catch exceptions from the core loop (Rule 2.7)
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
    finally:
        renderer.close()
        logging.shutdown()


if __name__ == "__main__":
    main()