# Life_Engine\src\engine.py

"""
Defines the Simulation engine, which runs the main simulation loop.
"""
import logging
from src.environment import Environment
from src.renderer import Renderer

class Simulation:
    def __init__(self, config: dict, logger: logging.LoggerAdapter, renderer: Renderer):
        self.config = config
        self.logger = logger
        self.renderer = renderer
        # The number of steps is no longer fixed, so we remove it.
        self.environment = Environment(config["environment"])

    def run(self):
        """Executes the main simulation loop."""
        self.logger.info("Simulation starting. Press ESC or close the window to exit.")
        
        running = True
        step = 0
        while running:
            self.logger.extra['step'] = step
            
            running = self.renderer.handle_events()

            # --- Perception and Action Step ---
            for agent in self.environment.agents:
                agent.perceive_and_act(self.environment, self.logger)

            # --- State Update Step ---
            self.environment.update()

            # --- Rendering Step ---
            self.renderer.draw(self.environment, step)

            # --- Logging Step ---
            for agent in self.environment.agents:
                pos = agent.position
                # This is a high-frequency message, better suited for DEBUG level.
                self.logger.debug(
                    f"Agent {agent.id} updated position to ({pos[0]:.2f}, {pos[1]:.2f})"
                )
            
            step += 1

        self.logger.extra['step'] = 'FINAL'
        self.logger.info("Simulation finished.")