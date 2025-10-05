# Life_Engine\constants.py

"""
Application Constants for Life-Engine.
These values are static to the application and do not change between simulation runs.
"""

# Visualization settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)  # White

# Agent visual properties
AGENT_COLORS = [
    (200, 50, 50),   # Red
    (50, 200, 50),   # Green
    (50, 50, 200),   # Blue
    (200, 200, 50),  # Yellow
    (200, 50, 200),  # Magenta
]
SHOULDER_WIDTH = 14
TORSO_DEPTH = 8 # How "thick" the torso is from front to back

# Agent Head & Neck visual properties
HEAD_COLOR = (255, 200, 150) # Skin tone
HEAD_RADIUS = 5
NECK_COLOR = (225, 170, 120) # Darker skin tone
NECK_WIDTH = 4
NECK_HEIGHT = 3

# Agent Limb visual properties (viewed from top)
HAND_COLOR = (255, 200, 150) # Skin tone
HAND_RADIUS = 3
FOOT_COLOR = (80, 60, 40) # Brown/Shoe color
FOOT_WIDTH = 5
FOOT_HEIGHT = 6

# Gait cycle parameters
GAIT_AMPLITUDE_FORWARD = 8 # How far feet/hands move forward/back
GAIT_AMPLITUDE_SIDEWAYS = 6 # How far feet step out to the side

# Animation parameters
SIT_STAND_ANIMATION_DURATION = 15 # Duration in steps (frames) for the animation
PICKUP_ANIMATION_DURATION = 20
EAT_ANIMATION_DURATION = 90

# Food visual properties
APPLE_COLOR = (220, 40, 40) # Red
APPLE_RADIUS = 3

# Tree visual properties
APPLE_TREE_TRUNK_COLOR = (80, 50, 20) # Brown
APPLE_TREE_CANOPY_COLOR = (30, 140, 60) # Green

# Obstacle visual properties
OBSTACLE_COLOR = (100, 100, 100) # Dark Grey

# Agent State Visuals
DEAD_AGENT_COLOR = (50, 50, 50) # Dark Grey

# Agent Sensory Visualization
# RGBA format for transparency (A=alpha)
VISION_CONE_COLOR = (200, 200, 255, 100)
LARGE_VISION_CONE_COLOR = (255, 255, 200, 80) # Yellowish