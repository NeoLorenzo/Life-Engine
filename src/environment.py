# Life_Engine\src\environment.py

"""
Defines the Environment class, which contains and manages all agents and food.
"""
import numpy as np
from src.agent import Agent
import constants
import logging

class Food:
    """A simple class for food items."""
    def __init__(self, id: int, position: list[float]):
        self.id = id
        self.position = np.array(position, dtype=np.float64)

class Environment:
    def __init__(self, config: dict):
        self.agents = []
        agent_props = config["agent_properties"]
        
        for i in range(config["agent_quantity"]):
            # Generate random initial position and velocity (Rule 12)
            initial_pos = [
                np.random.uniform(agent_props["collision_radius"], constants.SCREEN_WIDTH - agent_props["collision_radius"]),
                np.random.uniform(agent_props["collision_radius"], constants.SCREEN_HEIGHT - agent_props["collision_radius"])
            ]
            initial_vel = np.random.rand(2) * 2 - 1 # Random vector between [-1, 1]
            
            agent = Agent(
                id=i,
                position=initial_pos,
                velocity=initial_vel,
                # Unpack the shared properties from the config
                **agent_props
            )
            self.agents.append(agent)

        self.food = [
            Food(
                id=conf["id"],
                position=conf["position"]
            )
            for conf in config.get("food", [])
        ]

    def _resolve_collisions(self, logger: logging.LoggerAdapter):
        """Handles agent-agent and agent-boundary collisions."""
        # Agent-Agent collision
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1 = self.agents[i]
                agent2 = self.agents[j]
                
                delta = agent1.position - agent2.position
                dist_sq = np.dot(delta, delta)
                min_dist = agent1.radius + agent2.radius
                
                if dist_sq < min_dist ** 2:
                    logger.debug(f"Collision detected and resolved between Agent {agent1.id} and Agent {agent2.id}")
                    dist = np.sqrt(dist_sq)
                    normal = delta / dist
                    overlap = min_dist - dist
                    
                    # Separate the agents
                    agent1.position += normal * overlap / 2
                    agent2.position -= normal * overlap / 2
                    
                    # Elastic collision response (simplified)
                    v1n = np.dot(agent1.velocity, normal)
                    v2n = np.dot(agent2.velocity, normal)
                    agent1.velocity += normal * (v2n - v1n)
                    agent2.velocity += normal * (v1n - v2n)

        # Agent-Boundary collision
        for agent in self.agents:
            collided = False
            if agent.position[0] < agent.radius:
                agent.position[0] = agent.radius
                agent.velocity[0] *= -0.9 # Reflect with damping
                collided = True
            elif agent.position[0] > constants.SCREEN_WIDTH - agent.radius:
                agent.position[0] = constants.SCREEN_WIDTH - agent.radius
                agent.velocity[0] *= -0.9
                collided = True
            
            if agent.position[1] < agent.radius:
                agent.position[1] = agent.radius
                agent.velocity[1] *= -0.9
                collided = True
            elif agent.position[1] > constants.SCREEN_HEIGHT - agent.radius:
                agent.position[1] = constants.SCREEN_HEIGHT - agent.radius
                agent.velocity[1] *= -0.9
                collided = True
            
            if collided:
                logger.debug(f"Agent {agent.id} collided with boundary.")

    def update(self, logger: logging.LoggerAdapter):
        """Updates the state of all agents in the environment."""
        for agent in self.agents:
            agent.update()
        
        self._resolve_collisions(logger)