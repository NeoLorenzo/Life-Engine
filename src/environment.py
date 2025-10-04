# Life_Engine\src\environment.py

"""
Defines the Environment class, which contains and manages all agents and food.
"""
import numpy as np
from src.agent import Agent

class Food:
    """A simple class for food items."""
    def __init__(self, id: int, position: list[float]):
        self.id = id
        self.position = np.array(position, dtype=np.float64)

class Environment:
    def __init__(self, config: dict):
        self.agents = [
            Agent(
                id=conf["id"],
                position=conf["initial_position"],
                velocity=conf["velocity"],
                max_speed=conf["max_speed"],
                max_force=conf["max_force"],
                friction_strength=conf["friction_strength"],
                vision_range=conf["vision_range"],
                field_of_view=conf["field_of_view"],
                wander_distance=conf["wander_distance"],
                wander_radius=conf["wander_radius"],
                head_scan_angle=conf["head_scan_angle"],
                head_scan_speed=conf["head_scan_speed"],
                head_turn_speed=conf["head_turn_speed"]
            )
            for conf in config["agents"]
        ]
        self.food = [
            Food(
                id=conf["id"],
                position=conf["position"]
            )
            for conf in config.get("food", [])
        ]

    def update(self):
        """Updates the state of all agents in the environment."""
        for agent in self.agents:
            agent.update()