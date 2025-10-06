# Life_Engine\src\environment.py

"""
Defines the Environment class, which contains and manages all agents and food.
"""
import numpy as np
from src.agent import Agent
import constants
import logging
import math

class Apple:
    """A simple class for apple items."""
    def __init__(self, id: int, position: list[float], parent_tree: 'AppleTree'):
        self.id = id
        self.position = np.array(position, dtype=np.float64)
        self.parent_tree = parent_tree

class AppleTree:
    """A class for apple trees that can spawn apples."""
    def __init__(self, id: int, position: list[float], trunk_radius: float, canopy_radius: float):
        self.id = id
        self.position = np.array(position, dtype=np.float64)
        self.trunk_radius = trunk_radius
        self.canopy_radius = canopy_radius
        self.spawned_apples = [] # This will hold references to apple objects
        self.trunk_obstacle = None # Will hold a reference to the physical obstacle

class Obstacle:
    """A simple class for static obstacles."""
    def __init__(self, id: int, position: list[float], radius: float):
        self.id = id
        self.position = np.array(position, dtype=np.float64)
        self.radius = radius

class Environment:
    def __init__(self, config: dict):
        self.config = config # Store the config for later use
        
        # --- Day/Night Cycle Initialization ---
        cycle_config = config.get("day_night_cycle", {})
        self.cycle_duration = cycle_config.get("cycle_duration_steps", 24000) # Default to 24k steps
        self.day_duration_percent = cycle_config.get("day_duration_percent", 0.5)
        self.is_day = True # Start during the day
        
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
                properties=agent_props
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

        # Procedural Apple Tree Generation
        self.trees = []
        tree_config = config.get("apple_trees", {})
        obstacle_id_offset = len(self.obstacles) # Start obstacle IDs after existing ones
        for i in range(tree_config.get("quantity", 0)):
            tree_pos = [
                np.random.uniform(0, constants.SCREEN_WIDTH),
                np.random.uniform(0, constants.SCREEN_HEIGHT)
            ]
            trunk_rad = np.random.uniform(
                tree_config.get("min_radius", 15) * 0.2, # Trunk is a fraction of canopy
                tree_config.get("max_radius", 30) * 0.3
            )
            
            # Create the visual tree object
            tree = AppleTree(
                id=i,
                position=tree_pos,
                trunk_radius=trunk_rad,
                canopy_radius=np.random.uniform(
                    tree_config.get("min_radius", 15),
                    tree_config.get("max_radius", 30)
                )
            )
            self.trees.append(tree)

            # Create a corresponding physical obstacle for the trunk
            trunk_obstacle = Obstacle(
                id=obstacle_id_offset + i,
                position=tree_pos,
                radius=trunk_rad
            )
            self.obstacles.append(trunk_obstacle)
            tree.trunk_obstacle = trunk_obstacle # Link the tree to its obstacle

        # Procedural Initial Apple Generation
        self.apples = []
        apple_config = config.get("initial_apples", {})
        for i in range(apple_config.get("quantity", 0)):
            self.apples.append(
                Apple(
                    id=i,
                    position=[
                        np.random.uniform(0, constants.SCREEN_WIDTH),
                        np.random.uniform(0, constants.SCREEN_HEIGHT)
                    ]
                )
            )
        self.next_apple_id = len(self.apples)

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

    @property
    def light_level(self) -> float:
        """
        Calculates the current ambient light level (0.0 to 1.0) based on the time of day.
        Uses a cosine function for a smooth transition.
        """
        phase = (self.world_step % self.cycle_duration) / self.cycle_duration
        # A cosine wave shifted and scaled to be in the [0, 1] range.
        # It's 1.0 at phase 0.25 (noon) and 0.0 at phase 0.75 (midnight).
        light = (math.cos((phase - 0.25) * 2 * math.pi) + 1) / 2.0
        return light

    def update(self, logger: logging.LoggerAdapter, step: int):
        """Updates the state of all agents and objects in the environment."""
        self.world_step = step
        
        # --- Day/Night Cycle Update ---
        was_day = self.is_day
        # Transition to night when the sun sets past the horizon
        night_start_phase = self.day_duration_percent / 2 + 0.25
        day_start_phase = 1.0 - (self.day_duration_percent / 2) + 0.25
        
        current_phase = (self.world_step % self.cycle_duration) / self.cycle_duration
        
        # Handle phase wrapping around 1.0 for day start
        if day_start_phase >= 1.0:
            day_start_phase -= 1.0
            self.is_day = current_phase >= day_start_phase or current_phase < night_start_phase
        else:
            self.is_day = day_start_phase <= current_phase < night_start_phase

        if was_day and not self.is_day:
            logger.info("The sky darkens. Night has begun.")
        elif not was_day and self.is_day:
            logger.info("The sun rises. Day has begun.")

        # --- Apple Spawning Step ---
        # Note: This logic will be expanded later to remove eaten apples from tree lists.
        tree_config = self.config.get("apple_trees", {})
        max_apples = tree_config.get("max_apples", 5)
        spawn_rate = tree_config.get("apple_spawn_rate", 0.005)
        spawn_radius_multiplier = tree_config.get("spawn_radius_multiplier", 1.5)

        for tree in self.trees:
            if len(tree.spawned_apples) < max_apples and np.random.rand() < spawn_rate:
                # Calculate a random spawn position within the tree's spawn radius
                angle = np.random.uniform(0, 2 * np.pi)
                # Spawn between the trunk's edge and the canopy's spawn radius
                min_spawn_rad = tree.trunk_radius + constants.APPLE_RADIUS
                max_spawn_rad = tree.canopy_radius * spawn_radius_multiplier
                radius = np.random.uniform(min_spawn_rad, max_spawn_rad)
                spawn_pos = tree.position + np.array([np.cos(angle), np.sin(angle)]) * radius
                
                # Create and register the new apple
                new_apple = Apple(id=self.next_apple_id, position=spawn_pos, parent_tree=tree)
                self.apples.append(new_apple)
                tree.spawned_apples.append(new_apple) # Tree keeps a reference
                self.next_apple_id += 1
                logger.debug(f"Tree {tree.id} spawned Apple {new_apple.id}.")

        # --- Agent State Update ---
        for agent in self.agents:
            agent.update(self, logger, step)
        
        self._resolve_collisions(logger)