# Life_Engine\src\agent.py

"""
Defines the Agent class, representing a single entity in the simulation.

Data Contract:
- Inputs:
    - id (int): A unique identifier for the agent.
    - position (np.ndarray): A 2D vector [x, y] representing the agent's location.
    - velocity (np.ndarray): A 2D vector [vx, vy] representing the agent's speed and direction.
    - max_speed (float): The maximum speed the agent can travel in one step.
- Outputs:
    - The agent's state is modified in place via its methods.
- Invariants:
    - Position and velocity must always be 2-element NumPy arrays of type float64.
"""
import numpy as np
import logging
import math
import constants
import random
# We need to import the Environment class for type hinting, but this can
# create a circular dependency. We use a string hint ('Environment') to avoid this.

class Agent:
    def __init__(self, id: int, position: list[float], velocity: list[float], properties: dict):
        self.id = id
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        
        # Unpack properties from the dictionary using .get() for safety
        self.radius = properties.get("collision_radius", 8.0)
        self.color = random.choice(constants.AGENT_COLORS)
        self.flocking_range = properties.get("flocking_range", 150.0)
        self.flocking_strength = properties.get("flocking_strength", 0.5)
        self.acceleration = np.zeros(2, dtype=np.float64)
        
        initial_heading = self.velocity / np.linalg.norm(self.velocity) if np.linalg.norm(self.velocity) > 0 else np.array([1.0, 0.0])
        self.heading_vector = initial_heading
        
        self.base_max_speed = properties.get("max_speed", 1.5)
        self.max_speed = self.base_max_speed
        self.max_force = properties.get("max_force", 0.05)
        self.head_turn_speed = properties.get("head_turn_speed", 0.1)
        self.friction_strength = properties.get("friction_strength", 0.99)
        self.base_vision_range = properties.get("vision_range", 150)
        self.base_large_vision_range = properties.get("large_vision_range", 300)
        self.vision_clarity_in_darkness = properties.get("vision_clarity_in_darkness_percent", 0.1)
        self.current_vision_range = self.base_vision_range
        self.current_large_vision_range = self.base_large_vision_range
        self.wander_distance = properties.get("wander_distance", 100)
        self.wander_radius = properties.get("wander_radius", 50)
        self.wander_target = None
        
        self.max_stamina = properties.get("max_stamina", 100.0)
        self.stamina = self.max_stamina
        self.stamina_consumption_rate = properties.get("stamina_consumption_rate", 0.05)
        self.stamina_regeneration_rate = properties.get("stamina_regeneration_rate", 0.4)
        self.tired_threshold = self.max_stamina * properties.get("tired_threshold_percent", 0.4)
        self.rest_threshold = self.max_stamina * properties.get("rest_threshold_percent", 0.1)
        self.wake_threshold = self.max_stamina * properties.get("wake_threshold_percent", 0.95)
        
        self.max_hunger = properties.get("max_hunger", 2500.0)
        self.hunger = properties.get("initial_hunger", 2000.0)
        self.hunger_decay_rate = properties.get("hunger_decay_rate", 0.1)
        self.hunger_per_apple = properties.get("hunger_per_apple", 500.0)
        self.hungry_threshold = self.max_hunger * properties.get("hungry_threshold_percent", 0.6)

        # --- Vigilance System ---
        vigilance_system = properties.get("vigilance_system", {})
        self.max_vigilance = vigilance_system.get("max_vigilance", 100.0)
        self.vigilance = vigilance_system.get("initial_vigilance", 100.0)
        self.vigilance_decay_day = vigilance_system.get("decay_rate_day", 0.075)
        self.vigilance_decay_night = vigilance_system.get("decay_rate_night", 0.15)
        self.vigilance_regeneration_rate = vigilance_system.get("regeneration_rate", 0.5)
        self.sleep_threshold = self.max_vigilance * vigilance_system.get("sleep_threshold_percent", 0.1)
        self.wake_vigilance_threshold = self.max_vigilance * vigilance_system.get("wake_threshold_percent", 0.95)
        self.stamina_vigilance_floor = vigilance_system.get("stamina_vigilance_floor_percent", 0.2)
        
        self.head_scan_angle_rad = math.radians(properties.get("head_scan_angle", 80))
        self.head_scan_speed = properties.get("head_scan_speed", 0.05)
        self.scan_phase = 0.0
        
        self.state = "wandering"
        # New states: "seeking_tree", "searching_for_apple", "moving_to_apple", "picking_up", "eating", "dead", "sleeping", "lying_down", "getting_up"
        self.target_entity = None
        self.search_anchor_point = None
        self.search_angle = np.random.uniform(0, 2 * np.pi)
        self.animation_timer = 0
        self.gait_phase = 0.0
        self.gait_speed = 0.2

        # --- Short-Term Memory ---
        self.memory_persistence_steps = properties.get("memory_persistence_steps", 90)
        self.memory_timer = 0
        self.last_known_position = None
        self.is_operating_from_memory = False # Flag for the renderer
        
        self.field_of_view_rad = math.radians(properties.get("field_of_view", 120))
        self.cos_fov_half = math.cos(self.field_of_view_rad / 2.0)
        self.large_field_of_view_rad = math.radians(properties.get("large_field_of_view", 160))

        # --- Context Steering Architecture ---
        self.map_resolution = properties.get("context_map_resolution", 32)
        self.threat_range = properties.get("threat_detection_range", 60.0)
        self.danger_avoidance_strength = properties.get("danger_avoidance_strength", 4.0)
        self.interest_map = np.zeros(self.map_resolution)
        self.danger_map = np.zeros(self.map_resolution)
        # Pre-calculate direction vectors for each slot in the map
        angles = np.linspace(0, 2 * np.pi, self.map_resolution, endpoint=False)
        self.direction_vectors = np.c_[np.cos(angles), np.sin(angles)]

    @property
    def forward_vector(self) -> np.ndarray:
        return self.heading_vector

    def _apply_force(self, force: np.ndarray):
        self.acceleration += force

    def _seek(self, target_pos: np.ndarray) -> np.ndarray:
        desired_velocity = target_pos - self.position
        norm = np.linalg.norm(desired_velocity)
        if norm > 0:
            desired_velocity = (desired_velocity / norm) * self.max_speed
            steering_force = desired_velocity - self.velocity
            norm_steering = np.linalg.norm(steering_force)
            if norm_steering > self.max_force:
                steering_force = (steering_force / norm_steering) * self.max_force
            return steering_force
        return np.zeros(2)

    def _is_entity_visible(self, entity: object) -> bool:
        """Checks if a given entity is within the agent's field of vision."""
        # Use large vision range for trees, standard for others (e.g., apples)
        vision_range = self.current_large_vision_range if hasattr(entity, 'canopy_radius') else self.current_vision_range
        
        vec_to_entity = entity.position - self.position
        dist_sq = np.dot(vec_to_entity, vec_to_entity)

        # 1. Check if the entity is within vision range
        if dist_sq > vision_range**2:
            return False

        # 2. Check if the entity is within the field of view
        if dist_sq > 1e-6: # Avoid division by zero if on top of the entity
            vec_to_entity_normalized = vec_to_entity / np.sqrt(dist_sq)
            dot_product = np.dot(self.heading_vector, vec_to_entity_normalized)
            
            # Determine correct FOV based on entity type
            cos_fov = self.cos_fov_half # Default for standard vision
            if hasattr(entity, 'canopy_radius'):
                # This is a tree, so we need to calculate the cos of the large FOV
                # We do it here to avoid storing another variable in __init__
                cos_large_fov_half = math.cos(self.large_field_of_view_rad / 2.0)
                cos_fov = cos_large_fov_half

            if dot_product < cos_fov:
                return False
                
        # NOTE: This is where a future raycast for occlusion would go.
        # For now, if it's in range and in the cone, it's considered visible.
        return True

    def _evaluate_context(self, environment: 'Environment'):
        """Populates the interest and danger maps based on the environment."""
        self.interest_map.fill(0)
        self.danger_map.fill(0)

        # --- Evaluate DANGER ---
        # Obstacles
        for obstacle in environment.obstacles:
            # If moving to an apple, ignore the danger from its parent tree's trunk
            if self.state == "moving_to_apple" and self.target_entity is not None:
                if obstacle is self.target_entity.parent_tree.trunk_obstacle:
                    continue # Skip this specific obstacle

            vec_to_obstacle = obstacle.position - self.position
            dist_sq = np.dot(vec_to_obstacle, vec_to_obstacle)
            if dist_sq < (self.threat_range + obstacle.radius)**2:
                dist = np.sqrt(dist_sq)
                # The closer the obstacle, the higher the danger
                danger_value = 1.0 - (dist / (self.threat_range + obstacle.radius))
                # Project danger onto the map
                dot_products = np.dot(self.direction_vectors, vec_to_obstacle / dist)
                self.danger_map += np.maximum(0, dot_products) * (danger_value ** 2)
        
        # Other Agents
        for other_agent in environment.agents:
            if other_agent.id == self.id:
                continue
            
            vec_to_agent = other_agent.position - self.position
            dist_sq = np.dot(vec_to_agent, vec_to_agent)
            
            # Only consider agents within the threat detection range
            if dist_sq < self.threat_range**2 and dist_sq > 1e-6:
                dist = np.sqrt(dist_sq)
                # Danger is inversely proportional to distance, cubed for sharper falloff
                danger_value = 1.0 - (dist / self.threat_range)
                
                # Project this danger onto the map
                dot_products = np.dot(self.direction_vectors, vec_to_agent / dist)
                self.danger_map += np.maximum(0, dot_products) * (danger_value ** 3)

        # Boundaries
        # If the agent is very close to an apple, it should ignore boundary danger to retrieve it.
        ignore_boundary_danger = False
        if self.state == "moving_to_apple" and self.target_entity is not None:
            dist_to_target_sq = np.dot(self.target_entity.position - self.position, self.target_entity.position - self.position)
            # Ignore danger if within half the threat range of the target apple.
            if dist_to_target_sq < (self.threat_range * 0.5)**2:
                ignore_boundary_danger = True

        if not ignore_boundary_danger:
            boundary_points = [
                (self.position[0], 0), (self.position[0], constants.SCREEN_HEIGHT),
                (0, self.position[1]), (constants.SCREEN_WIDTH, self.position[1])
            ]
            for point in boundary_points:
                vec_to_boundary = np.array(point) - self.position
                dist = np.linalg.norm(vec_to_boundary)
                if dist < self.threat_range:
                    danger_value = 1.0 - (dist / self.threat_range)
                    dot_products = np.dot(self.direction_vectors, vec_to_boundary / dist)
                    self.danger_map += np.maximum(0, dot_products) * (danger_value ** 2)

        # --- Evaluate INTEREST ---
        # Flocking (Cohesion)
        nearby_peers = []
        center_of_mass = np.zeros(2, dtype=np.float64)
        for peer in environment.agents:
            if peer.id == self.id or peer.color != self.color:
                continue
            dist_sq = np.dot(peer.position - self.position, peer.position - self.position)
            if dist_sq < self.flocking_range**2:
                nearby_peers.append(peer.position)
        
        if nearby_peers:
            center_of_mass = np.mean(nearby_peers, axis=0)
            vec_to_com = center_of_mass - self.position
            norm = np.linalg.norm(vec_to_com)
            if norm > 0:
                dot_products = np.dot(self.direction_vectors, vec_to_com / norm)
                self.interest_map += np.maximum(0, dot_products) * self.flocking_strength

        # --- Hunger-Driven Interest ---
        is_hungry = self.hunger < self.hungry_threshold
        interest_found = False

        # If the agent has a specific target, that is its ONLY interest.
        if self.state in ["moving_to_apple", "seeking_tree"] and self.target_entity is not None:
            interest_found = True
            vec_to_target = self.target_entity.position - self.position
            norm = np.linalg.norm(vec_to_target)
            if norm > 0:
                dot_products = np.dot(self.direction_vectors, vec_to_target / norm)
                # Use a high interest value for committed targets
                self.interest_map += np.maximum(0, dot_products) * 1.0
        
        # Otherwise, if hungry, look for new targets.
        elif is_hungry:
            # 1. Look for nearby apples (highest priority)
            visible_apples = []
            for apple in environment.apples:
                vec_to_apple = apple.position - self.position
                if np.dot(vec_to_apple, vec_to_apple) < self.current_vision_range**2:
                    visible_apples.append(apple)
            
            if visible_apples:
                interest_found = True
                closest_apple = min(visible_apples, key=lambda a: np.linalg.norm(a.position - self.position))
                vec_to_target = closest_apple.position - self.position
                norm = np.linalg.norm(vec_to_target)
                if norm > 0:
                    dot_products = np.dot(self.direction_vectors, vec_to_target / norm)
                    self.interest_map += np.maximum(0, dot_products) * 1.0 # Max interest

            # 2. If no apples are seen, look for distant trees (unless already searching)
            if not interest_found and self.state != "searching_for_apple":
                visible_trees = []
                for tree in environment.trees:
                    vec_to_tree = tree.position - self.position
                    if np.dot(vec_to_tree, vec_to_tree) < self.current_large_vision_range**2:
                        visible_trees.append(tree)
                
                if visible_trees:
                    interest_found = True
                    closest_tree = min(visible_trees, key=lambda t: np.linalg.norm(t.position - self.position))
                    vec_to_target = closest_tree.position - self.position
                    norm = np.linalg.norm(vec_to_target)
                    if norm > 0:
                        dot_products = np.dot(self.direction_vectors, vec_to_target / norm)
                        self.interest_map += np.maximum(0, dot_products) * 0.8 # High interest

        # 3. If no survival interest is found, determine how to wander
        if not interest_found:
            # Special circular wander when searching near a tree
            if self.state == "searching_for_apple" and self.search_anchor_point is not None:
                if self.wander_target is None or np.linalg.norm(self.position - self.wander_target) < self.wander_radius * 0.5:
                    self.search_angle += np.random.uniform(0.5, 1.5) # Increment angle
                    search_radius = self.wander_radius * 1.5
                    target_offset = np.array([np.cos(self.search_angle), np.sin(self.search_angle)]) * search_radius
                    self.wander_target = self.search_anchor_point + target_offset
            # Standard forward-projected wander
            else:
                if self.wander_target is None or np.linalg.norm(self.position - self.wander_target) < self.wander_radius:
                    circle_center = self.position + self.heading_vector * self.wander_distance
                    angle = np.random.uniform(0, 2 * math.pi)
                    displacement = np.array([math.cos(angle), math.sin(angle)]) * self.wander_radius
                    self.wander_target = np.clip(circle_center + displacement, [0,0], [constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT])
            
            vec_to_target = self.wander_target - self.position
            norm = np.linalg.norm(vec_to_target)
            if norm > 0:
                dot_products = np.dot(self.direction_vectors, vec_to_target / norm)
                self.interest_map += np.maximum(0, dot_products) * 0.3 # Lower interest

    def perceive_and_act(self, environment: 'Environment', logger: logging.LoggerAdapter):
        """The agent's 'brain'. Decides what to do based on its state and environment."""
        # --- Universal State Transitions (Vigilance/Sleeping) ---
        if self.state == "sleeping" and self.vigilance >= self.wake_vigilance_threshold:
            self.state = "getting_up"
            self.animation_timer = constants.SLEEP_ANIMATION_DURATION
            self.stamina = self.max_stamina # Restore stamina when the decision to wake is made
            logger.info(f"Agent {self.id} is getting up. Stamina restored.")
            return

        is_sleep_deprived = self.vigilance < self.sleep_threshold
        can_sleep = self.state not in ["sitting_down", "resting", "sleeping", "picking_up", "eating", "dead", "lying_down", "getting_up"]
        if is_sleep_deprived and can_sleep:
            self.state = "lying_down"
            self.animation_timer = constants.SLEEP_ANIMATION_DURATION
            logger.info(f"Agent {self.id} is lying down to sleep.")
            return

        # --- Universal State Transitions (Stamina/Resting) ---
        if self.state == "resting" and self.stamina >= self.wake_threshold:
            self.state = "standing_up"
            self.animation_timer = constants.SIT_STAND_ANIMATION_DURATION
            logger.info(f"Agent {self.id} has recovered. Standing up.")
            return
        
        is_tired = self.stamina < self.rest_threshold
        can_rest = self.state in ["wandering", "seeking_tree", "searching_for_apple"]
        if is_tired and can_rest:
            self.state = "sitting_down"
            self.animation_timer = constants.SIT_STAND_ANIMATION_DURATION
            logger.info(f"Agent {self.id} is exhausted. Sitting down.")
            return

        # States that block perception/action
        blocking_states = ["sitting_down", "standing_up", "resting", "picking_up", "eating", "dead", "sleeping", "lying_down", "getting_up"]
        if self.state in blocking_states:
            return

        # --- State-Specific Perception and Action ---
        is_hungry = self.hunger < self.hungry_threshold

        # --- Wandering State ---
        if self.state == "wandering" and is_hungry:
            self.max_speed = self.base_max_speed # Ensure speed is reset
            # Look for apples first, using the proper visibility check
            visible_apples = [a for a in environment.apples if self._is_entity_visible(a)]
            if visible_apples:
                self.target_entity = min(visible_apples, key=lambda a: np.linalg.norm(a.position - self.position))
                self.state = "moving_to_apple"
                # Immediately set the memory upon acquiring a target
                self.last_known_position = self.target_entity.position.copy()
                logger.info(f"Agent {self.id} is hungry and sees an apple. Moving to it.")
            else:
                # If no apples, look for trees, using the proper visibility check
                visible_trees = [t for t in environment.trees if self._is_entity_visible(t)]
                if visible_trees:
                    self.target_entity = min(visible_trees, key=lambda t: np.linalg.norm(t.position - self.position))
                    self.state = "seeking_tree"
                    logger.info(f"Agent {self.id} is hungry and sees a tree. Moving towards it.")

        # --- Seeking Tree State ---
        elif self.state == "seeking_tree":
            if self.target_entity is None or not hasattr(self.target_entity, 'canopy_radius'):
                self.state = "wandering" # Target is invalid
                return

            # --- Memory Update Logic ---
            is_visible = self._is_entity_visible(self.target_entity)
            if is_visible:
                self.last_known_position = self.target_entity.position.copy()
                self.memory_timer = self.memory_persistence_steps
                if self.is_operating_from_memory:
                    self.is_operating_from_memory = False
                    logger.info(f"Agent {self.id} re-acquired target tree {self.target_entity.id}.")
            elif self.memory_timer > 0 and not self.is_operating_from_memory:
                self.is_operating_from_memory = True
                logger.info(f"Agent {self.id} lost sight of target tree {self.target_entity.id}. Operating from memory.")
            # --- End Memory Logic ---
            
            dist_to_tree = np.linalg.norm(self.target_entity.position - self.position)
            if dist_to_tree < self.target_entity.canopy_radius * 1.2: # Arrived at the tree
                self.state = "searching_for_apple"
                logger.info(f"Agent {self.id} reached the tree. Now searching for apples.")
                self.search_anchor_point = self.target_entity.position
                self.target_entity = None # Clear tree target

        # --- Searching for Apple State ---
        elif self.state == "searching_for_apple":
            self.max_speed = self.base_max_speed # Ensure speed is reset
            # Use the proper visibility check
            visible_apples = [a for a in environment.apples if self._is_entity_visible(a)]
            if visible_apples:
                self.target_entity = min(visible_apples, key=lambda a: np.linalg.norm(a.position - self.position))
                self.state = "moving_to_apple"
                logger.info(f"Agent {self.id} found an apple under the tree.")
            elif not is_hungry: # No longer hungry
                self.state = "wandering"

        # --- Moving to Apple State (with Arrival Behavior) ---
        elif self.state == "moving_to_apple":
            if self.target_entity is None or self.target_entity not in environment.apples:
                self.state = "searching_for_apple" # Target was eaten by someone else
                self.target_entity = None
                self.last_known_position = None
                return

            # --- Memory Update Logic ---
            is_visible = self._is_entity_visible(self.target_entity)
            if is_visible:
                self.last_known_position = self.target_entity.position.copy()
                self.memory_timer = self.memory_persistence_steps
                if self.is_operating_from_memory:
                    self.is_operating_from_memory = False
                    logger.info(f"Agent {self.id} re-acquired target apple {self.target_entity.id}.")
            elif self.memory_timer > 0 and not self.is_operating_from_memory:
                self.is_operating_from_memory = True
                logger.info(f"Agent {self.id} lost sight of target apple {self.target_entity.id}. Operating from memory.")
            # --- End Memory Logic ---
            
            dist_to_apple = np.linalg.norm(self.target_entity.position - self.position)

            # --- Arrival Logic ---
            slowing_radius = 50.0 # The radius within which the agent starts to slow down.
            if dist_to_apple < slowing_radius:
                # Linearly scale speed from max_speed down to a minimum value
                min_speed = 0.2
                speed_scale = max(min_speed, (dist_to_apple / slowing_radius))
                self.max_speed = self.base_max_speed * speed_scale
            else:
                self.max_speed = self.base_max_speed
            # --- End Arrival Logic ---

            if dist_to_apple < self.radius + 2: # Close enough to pick up
                self.state = "picking_up"
                self.animation_timer = constants.PICKUP_ANIMATION_DURATION
                logger.info(f"Agent {self.id} is picking up an apple.")
                self.velocity *= 0 # Stop moving
                self.max_speed = self.base_max_speed # Reset for next time
                return

        # --- Core Steering Logic (applies to all moving states) ---
        self._evaluate_context(environment)
        current_vel_norm = self.velocity / np.linalg.norm(self.velocity) if np.linalg.norm(self.velocity) > 0 else self.heading_vector
        momentum_bonus = np.maximum(0, np.dot(self.direction_vectors, current_vel_norm)) * 0.1
        final_map = self.interest_map - (self.danger_map * self.danger_avoidance_strength) + momentum_bonus
        
        best_direction_index = np.argmax(final_map)
        best_direction_vector = self.direction_vectors[best_direction_index]
        
        steering_target = self.position + best_direction_vector * 100
        self._apply_force(self._seek(steering_target))
        
        self.heading_vector = (self.heading_vector * 0.9) + (best_direction_vector * 0.1)
        self.heading_vector /= np.linalg.norm(self.heading_vector)

    def update(self, environment: 'Environment', logger: logging.LoggerAdapter, step: int):
        """Updates the agent's state based on physics."""
        # Dead agents do nothing.
        if self.state == "dead":
            return

        # --- Update Short-Term Memory Timer ---
        if self.memory_timer > 0:
            self.memory_timer -= 1

        # --- Update Sensory Acuity based on Environment ---
        light_level = environment.light_level
        vision_factor = self.vision_clarity_in_darkness + (1.0 - self.vision_clarity_in_darkness) * light_level
        
        # Vision is disabled completely when sleeping or in related states
        if self.state in ["sleeping", "lying_down", "getting_up"]:
            vision_factor = 0.0

        self.current_vision_range = self.base_vision_range * vision_factor
        self.current_large_vision_range = self.base_large_vision_range * vision_factor
        
        if step % 100 == 0:
            logger.debug(f"Agent {self.id} light_level {light_level:.2f} -> vision_range {self.current_vision_range:.1f}")

        # --- Vigilance Update ---
        if self.state == "sleeping":
            self.vigilance = min(self.max_vigilance, self.vigilance + self.vigilance_regeneration_rate)
            if step % 100 == 0: # Throttle logging (Rule 2.4)
                logger.debug(f"Agent {self.id} sleeping. Vigilance is now {self.vigilance:.2f}")
        else:
            light_level = environment.light_level
            # Linearly interpolate between night and day decay rates
            effective_decay_rate = self.vigilance_decay_night * (1.0 - light_level) + self.vigilance_decay_day * light_level
            self.vigilance = max(0, self.vigilance - effective_decay_rate)
            if step % 100 == 0: # Throttle logging (Rule 2.4)
                logger.debug(f"Agent {self.id} vigilance is {self.vigilance:.2f} (decay rate: {effective_decay_rate:.3f})")

        if self.state == "sleeping":
            self.velocity *= 0
            self.acceleration.fill(0)
            return

        if self.state == "resting":
            self.stamina = min(self.max_stamina, self.stamina + self.stamina_regeneration_rate)
            self.velocity *= 0
            self.acceleration.fill(0)
            return

        if self.state in ["sitting_down", "standing_up", "lying_down", "getting_up"]:
            self.velocity *= 0.8 # Slow to a stop while animating
            self.acceleration.fill(0)
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                if self.state == "sitting_down":
                    self.state = "resting"
                elif self.state == "standing_up":
                    self.state = "wandering"
                elif self.state == "lying_down":
                    self.state = "sleeping"
                elif self.state == "getting_up":
                    self.state = "wandering"
            return
        
        if self.state == "picking_up":
            self.velocity *= 0.8
            self.acceleration.fill(0)
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                # Action: remove apple from world
                if self.target_entity in environment.apples:
                    environment.apples.remove(self.target_entity)
                    # Also remove from parent tree's list to allow respawning
                    for tree in environment.trees:
                        if self.target_entity in tree.spawned_apples:
                            tree.spawned_apples.remove(self.target_entity)
                            break
                self.target_entity = None
                self.state = "eating"
                self.animation_timer = constants.EAT_ANIMATION_DURATION
            return

        if self.state == "eating":
            self.velocity *= 0.8
            self.acceleration.fill(0)
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.hunger = min(self.max_hunger, self.hunger + self.hunger_per_apple)
                logger.info(f"Agent {self.id} finished eating. Hunger is now {self.hunger:.1f}.")
                self.state = "wandering"
                self.last_known_position = None # Clear memory after eating
            return

        # --- Hunger and Death ---
        self.hunger = max(0, self.hunger - self.hunger_decay_rate)
        if self.hunger <= 0 and self.state != "dead":
            self.state = "dead"
            self.velocity.fill(0)
            self.acceleration.fill(0)
            logger.info(f"Agent {self.id} has died of starvation.")
            return # Stop further processing for this step

        # --- Stamina Consumption and Speed Reduction ---
        # Vigilance now caps the maximum possible stamina.
        vigilance_factor = self.vigilance / self.max_vigilance
        stamina_cap_factor = self.stamina_vigilance_floor + (1.0 - self.stamina_vigilance_floor) * vigilance_factor
        current_max_stamina = self.max_stamina * stamina_cap_factor
        self.stamina = min(self.stamina, current_max_stamina)
        
        if step % 100 == 0:
            logger.debug(f"Agent {self.id} vigilance {self.vigilance:.1f} -> stamina cap {current_max_stamina:.1f}")

        current_speed = np.linalg.norm(self.velocity)
        stamina_consumed = current_speed * self.stamina_consumption_rate
        self.stamina = max(0, self.stamina - stamina_consumed)

        # The tired threshold is now relative to the current maximum stamina.
        current_tired_threshold = current_max_stamina * (self.tired_threshold / self.max_stamina)
        if self.stamina < current_tired_threshold:
            # Reduce max speed proportionally to remaining stamina below the threshold
            speed_multiplier = self.stamina / current_tired_threshold if current_tired_threshold > 0 else 0
            self.max_speed = self.base_max_speed * max(0.1, speed_multiplier) # Ensure a small minimum speed
        else:
            self.max_speed = self.base_max_speed # Restore full speed

        # --- Standard Physics Update ---
        self.gait_phase += current_speed * self.gait_speed
        self.velocity *= self.friction_strength
        self.velocity += self.acceleration
        
        norm = np.linalg.norm(self.velocity)
        if norm > self.max_speed:
            self.velocity = (self.velocity / norm) * self.max_speed
            
        self.position += self.velocity
        self.acceleration = np.zeros(2, dtype=np.float64)