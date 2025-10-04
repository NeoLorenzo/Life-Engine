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
# We need to import the Environment class for type hinting, but this can
# create a circular dependency. We use a string hint ('Environment') to avoid this.

class Agent:
    def __init__(self, id: int, position: list[float], velocity: list[float], collision_radius: float, max_stamina: float, stamina_consumption_rate: float, stamina_regeneration_rate: float, tired_threshold_percent: float, rest_threshold_percent: float, wake_threshold_percent: float, max_speed: float, max_force: float, friction_strength: float, vision_range: float, field_of_view: float, wander_distance: float, wander_radius: float, head_scan_angle: float, head_scan_speed: float, head_turn_speed: float):
        self.id = id
        self.radius = collision_radius
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.acceleration = np.zeros(2, dtype=np.float64)
        
        # Initialize heading to match initial velocity direction
        initial_heading = self.velocity / np.linalg.norm(self.velocity) if np.linalg.norm(self.velocity) > 0 else np.array([1.0, 0.0])
        self.heading_vector = initial_heading
        
        self.base_max_speed = max_speed # The agent's top speed when fully rested
        self.max_speed = max_speed
        self.max_force = max_force
        self.head_turn_speed = head_turn_speed
        self.friction_strength = friction_strength
        self.vision_range = vision_range
        self.wander_distance = wander_distance
        self.wander_radius = wander_radius
        self.wander_target = None
        
        # Stamina attributes
        self.max_stamina = max_stamina
        self.stamina = max_stamina
        self.stamina_consumption_rate = stamina_consumption_rate
        self.stamina_regeneration_rate = stamina_regeneration_rate
        self.tired_threshold = max_stamina * tired_threshold_percent
        self.rest_threshold = max_stamina * rest_threshold_percent
        self.wake_threshold = max_stamina * wake_threshold_percent
        
        self.head_scan_angle_rad = math.radians(head_scan_angle)
        self.head_scan_speed = head_scan_speed
        self.scan_phase = 0.0 # Start with head facing forward
        
        self.state = "wandering"
        self.gait_phase = 0.0
        self.gait_speed = 0.2
        
        self.field_of_view_rad = math.radians(field_of_view)
        self.cos_fov_half = math.cos(self.field_of_view_rad / 2.0)

    @property
    def forward_vector(self) -> np.ndarray:
        """The direction the agent's HEAD is facing."""
        return self.heading_vector

    def _apply_force(self, force: np.ndarray):
        self.acceleration += force

    def _seek(self, target_pos: np.ndarray) -> np.ndarray:
        """Calculates the steering force to move towards a target."""
        desired_velocity = target_pos - self.position
        desired_velocity /= np.linalg.norm(desired_velocity)
        desired_velocity *= self.max_speed
        
        steering_force = desired_velocity - self.velocity
        
        # Limit the steering force to max_force
        norm = np.linalg.norm(steering_force)
        if norm > self.max_force:
            steering_force = (steering_force / norm) * self.max_force
            
        return steering_force

    def perceive_and_act(self, environment: 'Environment', logger: logging.LoggerAdapter):
        """Perceives the environment and calculates forces to apply."""
        # --- State Transition Logic ---
        if self.state == "resting":
            if self.stamina > self.wake_threshold:
                self.state = "wandering"
                logger.info(f"Agent {self.id} has recovered. Resuming wandering.")
            else:
                return # Do nothing while resting

        if self.stamina < self.rest_threshold:
            self.state = "resting"
            logger.info(f"Agent {self.id} is exhausted. Entering resting state.")
            return # Stop acting for this frame

        # --- Perception Logic (runs only if not resting) ---
        visible_food = []
        for food in environment.food:
            vec_to_food = food.position - self.position
            dist_sq = np.dot(vec_to_food, vec_to_food)
            if dist_sq > self.vision_range ** 2: continue
            norm_vec_to_food = np.linalg.norm(vec_to_food)
            if norm_vec_to_food > 0:
                cos_angle = np.dot(self.forward_vector, vec_to_food / norm_vec_to_food)
                if cos_angle < self.cos_fov_half: continue
            visible_food.append((dist_sq, food))

        # --- Action Logic (runs only if not resting) ---
        target_heading = self.heading_vector
        if not visible_food:
            self.state = "wandering"
            if self.wander_target is None or np.linalg.norm(self.position - self.wander_target) < self.wander_radius:
                circle_center = self.position + self.heading_vector * self.wander_distance
                angle = np.random.uniform(0, 2 * math.pi)
                displacement = np.array([math.cos(angle), math.sin(angle)]) * self.wander_radius
                new_target = circle_center + displacement
                new_target[0] = np.clip(new_target[0], 0, constants.SCREEN_WIDTH)
                new_target[1] = np.clip(new_target[1], 0, constants.SCREEN_HEIGHT)
                self.wander_target = new_target
                logger.debug(f"Agent {self.id} generating new wander target at ({self.wander_target[0]:.1f}, {self.wander_target[1]:.1f})")
            
            seek_force = self._seek(self.wander_target)
            self._apply_force(seek_force)

            self.scan_phase += self.head_scan_speed
            scan_offset_angle = math.sin(self.scan_phase) * self.head_scan_angle_rad / 2.0
            body_direction = self.velocity / np.linalg.norm(self.velocity) if np.linalg.norm(self.velocity) > 0.1 else self.heading_vector
            c, s = math.cos(scan_offset_angle), math.sin(scan_offset_angle)
            rotation_matrix = np.array([[c, -s], [s, c]])
            target_heading = rotation_matrix @ body_direction
        else:
            self.state = "seeking"
            self.wander_target = None 
            visible_food.sort(key=lambda x: x[0])
            _, closest_food = visible_food[0]
            logger.debug(f"Agent {self.id} head locked on food {closest_food.id}.")
            
            target_heading = closest_food.position - self.position
            seek_force = self._seek(closest_food.position)
            self._apply_force(seek_force)

        norm = np.linalg.norm(target_heading)
        if norm > 0:
            target_heading /= norm
            self.heading_vector = (self.heading_vector * (1.0 - self.head_turn_speed)) + (target_heading * self.head_turn_speed)
            self.heading_vector /= np.linalg.norm(self.heading_vector)

    def update(self):
        """Updates the agent's state based on physics."""
        if self.state == "resting":
            self.stamina = min(self.max_stamina, self.stamina + self.stamina_regeneration_rate)
            self.velocity *= 0 # Come to a full stop
            self.acceleration = np.zeros(2, dtype=np.float64)
            return

        # --- Stamina Consumption and Speed Reduction ---
        current_speed = np.linalg.norm(self.velocity)
        stamina_consumed = current_speed * self.stamina_consumption_rate
        self.stamina = max(0, self.stamina - stamina_consumed)

        if self.stamina < self.tired_threshold:
            # Reduce max speed proportionally to remaining stamina below the threshold
            speed_multiplier = self.stamina / self.tired_threshold
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