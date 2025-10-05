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

class Obstacle:
    """A simple class for static obstacles."""
    def __init__(self, id: int, position: list[float], radius: float):
        self.id = id
        self.position = np.array(position, dtype=np.float64)
        self.radius = radius

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

        # Procedural Obstacle Generation
        self.obstacles = []
        obstacle_config = config.get("obstacles", {})
        for i in range(obstacle_config.get("quantity", 0)):
            self.obstacles.append(
                Obstacle(
                    id=i,
                    position=[
                        np.random.uniform(0, constants.SCREEN_WIDTH),
                        np.random.uniform(0, constants.SCREEN_HEIGHT)
                    ],
                    radius=np.random.uniform(
                        obstacle_config.get("min_radius", 10),
                        obstacle_config.get("max_radius", 30)
                    )
                )
            )

        # Procedural Food Generation
        self.food = []
        food_config = config.get("food", {})
        for i in range(food_config.get("quantity", 0)):
            self.food.append(
                Food(
                    id=i,
                    position=[
                        np.random.uniform(0, constants.SCREEN_WIDTH),
                        np.random.uniform(0, constants.SCREEN_HEIGHT)
                    ]
                )
            )

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
                
                if dist_sq < min_dist ** 2 and dist_sq > 1e-6:
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

        # Agent-Obstacle collision (NEW)
        for agent in self.agents:
            for obstacle in self.obstacles:
                delta = agent.position - obstacle.position
                dist_sq = np.dot(delta, delta)
                min_dist = agent.radius + obstacle.radius

                if dist_sq < min_dist ** 2:
                    logger.debug(f"Agent {agent.id} collided with obstacle {obstacle.id}.")
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-6 else 1.0
                    normal = delta / dist if dist > 1e-6 else np.array([1.0, 0.0])
                    overlap = min_dist - dist

                    # Push the agent out of the obstacle
                    agent.position += normal * overlap

                    # Reflect the agent's velocity
                    v_dot_n = np.dot(agent.velocity, normal)
                    agent.velocity -= 2 * v_dot_n * normal
                    agent.velocity *= 0.8 # Damping to lose energy on bounce

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