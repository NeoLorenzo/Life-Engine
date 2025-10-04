# Life-Engine

Life-Engine is a top-down 2D simulation framework designed to model and visualize life-like agents with an emphasis on realistic, emergent behavior. The project is built with a strict focus on modularity, testability, and scientifically-grounded principles, allowing for the incremental addition of complex systems like senses, physics, and decision-making.

The current simulation features a single "human" agent that can navigate its environment by sight, exhibiting intelligent wandering and seeking behaviors.

## Core Philosophy

The development of Life-Engine adheres to a strict set of rules designed to ensure the project remains robust, scalable, and easy to understand.

*   **Configuration-Driven Design:** The simulation is separated into static **Application Constants** (`constants.py`) and scenario-specific **Simulation Parameters** (`config.json`). This allows for different experiments to be run without ever touching the core source code.
*   **Strict Modularity (SOLID):** The codebase is highly modular, following SOLID principles. Each component (Engine, Agent, Environment, Renderer) has a single responsibility and communicates through well-defined interfaces. Dependencies are injected, not created internally, making the system highly testable and flexible.
*   **Realism Through Abstraction:** The primary goal is realism in *behavior and outcomes*. Physical and biological processes are modeled with a basis in real-world mechanics (e.g., steering forces, inertia, decoupled head/body movement). When 1:1 realism is too complex, scientifically-grounded abstractions are used and documented.
*   **Focus on Emergent Behavior:** The simulation is not scripted. Complex behaviors (like searching for food) are intended to *emerge* from a set of simple, underlying rules (e.g., "scan the environment," "if you see food, move towards it").
*   **Robust Logging:** All runtime messages use Python's `logging` system. No `print()` statements are used in the core logic. Log levels (`INFO`, `DEBUG`) allow for controlling the verbosity of the output for different analysis needs.
*   **Deterministic & Seeded Randomness:** All randomness is controlled by a single master seed defined in the configuration. This ensures that any given simulation run is 100% reproducible, which is critical for debugging and analysis.

## Current Features

*   **Physics-Based Movement:** Agents don't just move; they are governed by forces. They have inertia, a maximum acceleration (`max_force`), and friction, resulting in smooth, curved, and believable paths.
*   **Decoupled Head-Body System:** An agent's "head" (senses and orientation) can move independently of its "body" (momentum and velocity). This allows for realistic scanning and targeting behaviors.
*   **Sensory System (Vision):** Agents have a configurable forward-facing vision cone with a limited range and field of view. They can only perceive objects within this cone.
*   **Behavioral States:** Agents have internal states (`wandering`, `seeking`) that change based on sensory input and determine their actions.
*   **Intelligent Wandering:** When no target is visible, an agent exhibits a sophisticated wandering behavior:
    1.  The head performs a smooth, rhythmic, side-to-side scan.
    2.  The body chooses a long-term target point from within the agent's field of view and moves towards it.
*   **Modular Pygame Renderer:** All visualization is handled by a dedicated, decoupled renderer. The simulation logic is completely independent of how it is displayed.
*   **Anatomical Top-Down Visualization:** Agents are rendered from a true top-down perspective, with a torso, head, hands, and feet. A contralateral gait cycle (left hand moves with right foot) provides a realistic walking animation where limb movement is tied to the agent's speed.

## Project Structure

```
Life_Engine/
├── venv/                 # Virtual environment
├── config.json           # Defines the parameters for a specific simulation run
├── constants.py          # Defines static application-wide constants
├── main.py               # Main entry point for the application
└── src/
    ├── __init__.py
    ├── agent.py          # Defines the Agent class, its physics, senses, and "brain"
    ├── engine.py         # Defines the Simulation class and contains the main loop
    ├── environment.py    # Defines the world, containing agents and food
    ├── renderer.py       # Handles all Pygame-based visualization
    └── utils/
        ├── __init__.py
        └── logger_setup.py # Configures the application's logging system
```

## Getting Started

### Prerequisites
*   Python 3.9+

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Life_Engine
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Create a `requirements.txt` file:**
    Create a file named `requirements.txt` in the root directory and add the following lines:
    ```
    numpy
    pygame
    numba
    ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation

To run the simulation, simply execute the `main.py` script from the root directory:
```bash
python main.py
```
The simulation will run indefinitely. To stop it, press the `Escape` key or close the Pygame window.

## Configuration (`config.json`)

All parameters for a specific scenario are defined in `config.json`.

*   `simulation`: Global simulation settings.
    *   `seed`: The master seed for all random number generation to ensure reproducibility.
*   `environment`: Defines the contents of the world.
    *   `agents`: A list of agent configurations.
        *   `id`: Unique identifier for the agent.
        *   `initial_position`: Starting `[x, y]` coordinates.
        *   `velocity`: Initial velocity vector `[vx, vy]`.
        *   `max_speed`: The maximum speed the agent can reach (pixels/step).
        *   `max_force`: The maximum steering force that can be applied in one step. Controls acceleration and turning radius.
        *   `head_turn_speed`: How quickly the head can turn to face a new target (0.0 to 1.0).
        *   `friction_strength`: Multiplier applied to velocity each step to simulate drag (e.g., 0.99).
        *   `vision_range`: The maximum distance the agent can see (in pixels).
        *   `field_of_view`: The total angle of the vision cone (in degrees).
        *   `wander_distance`: How far in front of itself the agent projects a point when choosing a new wander target.
        *   `wander_radius`: The radius of the circle from which a random wander target is chosen.
        *   `head_scan_angle`: The maximum angle (in degrees) the head will scan to the left or right of its base direction.
        *   `head_scan_speed`: How fast the head performs its side-to-side scan.
    *   `food`: A list of food item configurations.
        *   `id`: Unique identifier for the food.
        *   `position`: The `[x, y]` coordinates of the food.
*   `logging`: Configures the logging system.
    *   `level`: The minimum log level to display (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
    *   `format`: The format string for log messages.

## Future Development Roadmap

*   **Hearing and Smell:** Implement new sensory systems based on distance and propagation.
*   **Energy & Metabolism:** Give agents finite energy that is consumed by movement and replenished by eating food.
*   **Complex Environments:** Add static obstacles that agents must navigate around.
*   **Agent Interaction:** Introduce multiple agents that can perceive and react to each other.
*   **Saving & Loading State:** Implement functionality to save the state of a long-running simulation and resume it later.