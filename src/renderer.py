# Life_Engine\src\renderer.py

"""
Handles all visualization for the simulation using Pygame.

Data Contract:
- Inputs:
    - An 'environment' object which contains a list of agents to be drawn.
      Each agent must have a 'position' attribute (np.ndarray) and a 'color'
      and 'radius' attribute.
- Outputs:
    - Renders the state of the environment to the screen.
- Side Effects:
    - Initializes and manages a Pygame window.
"""
import pygame
import constants
from src.environment import Environment
import math
import numpy as np

class Renderer:
    """Manages the Pygame window and draws the simulation state."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Life-Engine")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock() # To control the frame rate

    def _draw_rotated_rect(self, surface, rect_dims, center_pos, angle_deg, color):
        """Helper function to draw a rotated rectangle, centered correctly."""
        rect_surface = pygame.Surface(rect_dims, pygame.SRCALPHA)
        rect_surface.fill(color)
        rotated_surface = pygame.transform.rotate(rect_surface, angle_deg)
        rect = rotated_surface.get_rect(center=center_pos)
        surface.blit(rotated_surface, rect.topleft)

    def draw(self, environment: Environment, step: int):
        """
        Draws the entire environment, including all agents and food.

        Args:
            environment (Environment): The simulation environment to render.
            step (int): The current simulation step, for display.
        """
        # 1. Fill the background
        self.screen.fill(constants.BACKGROUND_COLOR)

        # 2. Draw all obstacles
        for obstacle in environment.obstacles:
            pygame.draw.circle(
                self.screen,
                constants.OBSTACLE_COLOR,
                obstacle.position.astype(int),
                int(obstacle.radius)
            )

        # 3. Draw all food items
        for food in environment.food:
            pygame.draw.circle(
                self.screen,
                constants.FOOD_COLOR,
                food.position.astype(int),
                constants.FOOD_RADIUS
            )

        # 3. Draw all agents and their vision cones
        for agent in environment.agents:
            # --- 1. Calculate Head Position ---
            # We need the head's position for both the cone and the head itself.
            head_pos = agent.position + agent.heading_vector * (constants.TORSO_DEPTH)

            # --- 2. Draw Vision Cone from the Head ---
            center_point = head_pos # Use the head's position as the origin.
            forward_angle = math.atan2(agent.forward_vector[1], agent.forward_vector[0])
            start_angle = forward_angle - agent.field_of_view_rad / 2
            end_angle = forward_angle + agent.field_of_view_rad / 2
            
            points = [center_point]
            num_segments = 20
            for i in range(num_segments + 1):
                angle = start_angle + (end_angle - start_angle) * i / num_segments
                x = center_point[0] + agent.vision_range * math.cos(angle)
                y = center_point[1] + agent.vision_range * math.sin(angle)
                points.append((x, y))
            
            cone_surface = pygame.Surface((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(cone_surface, constants.VISION_CONE_COLOR, points)
            self.screen.blit(cone_surface, (0, 0))

            # --- 2. Define Orientations and Vectors ---
            body_angle_rad = math.atan2(agent.velocity[1], agent.velocity[0]) if np.linalg.norm(agent.velocity) > 0.1 else math.atan2(agent.heading_vector[1], agent.heading_vector[0])
            body_angle_deg = math.degrees(body_angle_rad)
            
            body_forward_vector = np.array([math.cos(body_angle_rad), math.sin(body_angle_rad)])
            body_right_vector = np.array([-math.sin(body_angle_rad), math.cos(body_angle_rad)])
            
            # The torso color is now the agent's assigned group color.
            # We can create a slightly darker version for the neck.
            torso_color = agent.color
            
            # --- 3. Calculate Limb Positions for Both States ---
            # --- Walking Pose Calculation ---
            speed_ratio = np.linalg.norm(agent.velocity) / agent.base_max_speed if agent.base_max_speed > 0 else 0
            gait_cycle = math.sin(agent.gait_phase) * speed_ratio
            right_foot_fwd_walk = body_forward_vector * gait_cycle * constants.GAIT_AMPLITUDE_FORWARD
            left_foot_fwd_walk = body_forward_vector * -gait_cycle * constants.GAIT_AMPLITUDE_FORWARD
            right_hand_fwd_walk = body_forward_vector * -gait_cycle * constants.GAIT_AMPLITUDE_FORWARD
            left_hand_fwd_walk = body_forward_vector * gait_cycle * constants.GAIT_AMPLITUDE_FORWARD
            sideways_gait = constants.GAIT_AMPLITUDE_SIDEWAYS * speed_ratio
            
            right_foot_pos_walk = agent.position + (body_right_vector * sideways_gait) + right_foot_fwd_walk
            left_foot_pos_walk = agent.position - (body_right_vector * sideways_gait) + left_foot_fwd_walk
            right_hand_pos_walk = agent.position + (body_right_vector * (constants.SHOULDER_WIDTH / 2)) + right_hand_fwd_walk
            left_hand_pos_walk = agent.position - (body_right_vector * (constants.SHOULDER_WIDTH / 2)) + left_hand_fwd_walk

            # --- Sitting Pose Calculation ---
            feet_fwd_sit = body_forward_vector * (constants.TORSO_DEPTH + constants.FOOT_HEIGHT / 2)
            right_foot_pos_sit = agent.position + feet_fwd_sit + (body_right_vector * constants.FOOT_WIDTH)
            left_foot_pos_sit = agent.position + feet_fwd_sit - (body_right_vector * constants.FOOT_WIDTH)
            hands_side_sit = body_right_vector * (constants.SHOULDER_WIDTH / 2 + constants.HAND_RADIUS)
            hands_back_sit = -body_forward_vector * (constants.TORSO_DEPTH / 2)
            right_hand_pos_sit = agent.position + hands_side_sit + hands_back_sit
            left_hand_pos_sit = agent.position - hands_side_sit + hands_back_sit

            # --- 4. Determine Final Limb Positions based on State ---
            if agent.state in ["wandering", "seeking"]:
                right_foot_pos, left_foot_pos = right_foot_pos_walk, left_foot_pos_walk
                right_hand_pos, left_hand_pos = right_hand_pos_walk, left_hand_pos_walk
            elif agent.state == "resting":
                right_foot_pos, left_foot_pos = right_foot_pos_sit, left_foot_pos_sit
                right_hand_pos, left_hand_pos = right_hand_pos_sit, left_hand_pos_sit
            else: # Animating
                progress = (constants.SIT_STAND_ANIMATION_DURATION - agent.animation_timer) / constants.SIT_STAND_ANIMATION_DURATION
                
                if agent.state == "sitting_down":
                    start_foot_r, end_foot_r = right_foot_pos_walk, right_foot_pos_sit
                    start_foot_l, end_foot_l = left_foot_pos_walk, left_foot_pos_sit
                    start_hand_r, end_hand_r = right_hand_pos_walk, right_hand_pos_sit
                    start_hand_l, end_hand_l = left_hand_pos_walk, left_hand_pos_sit
                else: # standing_up
                    start_foot_r, end_foot_r = right_foot_pos_sit, right_foot_pos_walk
                    start_foot_l, end_foot_l = left_foot_pos_sit, left_foot_pos_walk
                    start_hand_r, end_hand_r = right_hand_pos_sit, right_hand_pos_walk
                    start_hand_l, end_hand_l = left_hand_pos_sit, left_hand_pos_walk
                
                # Linear Interpolation (lerp)
                right_foot_pos = start_foot_r + (end_foot_r - start_foot_r) * progress
                left_foot_pos = start_foot_l + (end_foot_l - start_foot_l) * progress
                right_hand_pos = start_hand_r + (end_hand_r - start_hand_r) * progress
                left_hand_pos = start_hand_l + (end_hand_l - start_hand_l) * progress

            # --- 5. Draw Components in Correct Layers (Bottom to Top) ---
            # Feet
            self._draw_rotated_rect(self.screen, (constants.FOOT_WIDTH, constants.FOOT_HEIGHT), left_foot_pos, -body_angle_deg - 90, constants.FOOT_COLOR)
            self._draw_rotated_rect(self.screen, (constants.FOOT_WIDTH, constants.FOOT_HEIGHT), right_foot_pos, -body_angle_deg - 90, constants.FOOT_COLOR)
            
            # Hands
            pygame.draw.circle(self.screen, constants.HAND_COLOR, left_hand_pos.astype(int), constants.HAND_RADIUS)
            pygame.draw.circle(self.screen, constants.HAND_COLOR, right_hand_pos.astype(int), constants.HAND_RADIUS)

            # Torso (shoulders) - Drawn AFTER limbs so it appears on top
            self._draw_rotated_rect(self.screen, (constants.SHOULDER_WIDTH, constants.TORSO_DEPTH), agent.position, -body_angle_deg - 90, torso_color)
            
            # Neck and Head
            # Reduce the offset to bring the head closer to the body's center.
            head_pos = agent.position + agent.heading_vector * (constants.TORSO_DEPTH * 0.6)
            neck_pos = agent.position + agent.heading_vector * (constants.TORSO_DEPTH * 0.3)
            self._draw_rotated_rect(self.screen, (constants.NECK_WIDTH, constants.NECK_HEIGHT), neck_pos, math.degrees(math.atan2(agent.heading_vector[1], agent.heading_vector[0])), constants.NECK_COLOR)
            pygame.draw.circle(self.screen, constants.HEAD_COLOR, head_pos.astype(int), constants.HEAD_RADIUS)
        
        # 4. Draw the current step counter
        step_text = self.font.render(f"Step: {step}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))

        # 5. Update the display
        pygame.display.flip()
        
        # 6. Tick the clock to control update speed
        self.clock.tick(30) # Increase frame rate slightly

    def handle_events(self) -> bool:
        """
        Processes the Pygame event queue. Checks for window close and Escape key.
        
        Returns:
            bool: False if a QUIT or K_ESCAPE event is detected, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def close(self):
        """Shuts down the Pygame module."""
        pygame.quit()