# Life-Engine

Life-Engine is a top-down 2D simulation framework designed to model and visualize life-like agents with an emphasis on realistic, emergent behavior. The project is built with a strict focus on modularity, testability, and scientifically-grounded principles, allowing for the incremental addition of complex systems.

The current simulation features multiple agents governed by needs such as hunger, stamina, and vigilance. They navigate a dynamic environment with a day/night cycle, procedurally generated obstacles, and food sources (apple trees), exhibiting complex behaviors like foraging, resting, sleeping, and flocking.

## Core Philosophy

The development of Life-Engine adheres to a strict set of rules designed to ensure the project remains robust, scalable, and easy to understand.

*   **Configuration-Driven Design:** The simulation is separated into static **Application Constants** (`constants.py`) and scenario-specific **Simulation Parameters** (`config.json`). This allows for different experiments to be run without ever touching the core source code.
*   **Strict Modularity (SOLID):** The codebase is highly modular, following SOLID principles. Each component (Engine, Agent, Environment, Renderer) has a single responsibility and communicates through well-defined interfaces. Dependencies are injected, not created internally, making the system highly testable and flexible.
*   **Realism Through Abstraction:** The primary goal is realism in *behavior and outcomes*. Physical and biological processes are modeled with a basis in real-world mechanics. When 1:1 realism is too complex, scientifically-grounded abstractions are used and documented.
*   **Focus on Emergent Behavior:** The simulation is not scripted. Complex behaviors (like searching for food, resting when tired, or avoiding danger) are intended to *emerge* from a set of simple, underlying rules and internal needs.
*   **Robust Logging:** All runtime messages use Python's `logging` system. No `print()` statements are used in the core logic. Log levels (`INFO`, `DEBUG`) allow for controlling the verbosity of the output for different analysis needs.
*   **Deterministic & Seeded Randomness:** All randomness is controlled by a single master seed defined in the configuration. This ensures that any given simulation run is 100% reproducible, which is critical for debugging and analysis.

## Current Features

*   **Needs-Based State Machine:** Agents are driven by internal needs:
    *   **Hunger:** Decays over time; agents must find and eat apples to survive. Starvation leads to death.
    *   **Stamina:** Consumed by movement and dictates an agent's maximum speed. Agents will rest when stamina is low.
    *   **Vigilance:** Decreases while awake (faster at night) and is restored by sleeping. Low vigilance reduces an agent's stamina, forcing them to eventually sleep.

*   **Advanced AI & Navigation:**
    *   **Context Steering:** Agents use "interest" and "danger" maps to make navigation decisions, allowing them to fluidly pursue goals while avoiding obstacles, boundaries, and other agents.
    *   **Short-Term Memory:** Agents have a working memory that allows them to remember a target's last known position if it's temporarily obscured. This creates **task persistence** and more robust goal-seeking behavior.
    *   **Behavioral States:** A rich set of states (`wandering`, `seeking_tree`, `moving_to_apple`, `resting`, `sleeping`, `eating`) governs agent actions.
    *   **Flocking:** Agents of the same color exhibit a simple flocking behavior, creating emergent group movement.

*   **Dynamic Environment:**
    *   **Day/Night Cycle:** A smooth lighting transition affects agent vision, making it harder to see at night and increasing the rate of vigilance decay.
    *   **Procedural Generation:** Obstacles and apple trees are generated procedurally based on configuration settings.
    *   **Food Lifecycle:** Apple trees spawn apples over time, which can be picked up and consumed by agents.

*   **Physics and Senses:**
    *   **Physics-Based Movement:** Agents have inertia, a maximum acceleration (`max_force`), and friction, resulting in smooth, curved, and believable paths.
    *   **Decoupled Head-Body System:** An agent's "head" (senses) can orient independently of its "body" (momentum), allowing for realistic scanning behavior.
    *   **Dual-Cone Vision:** Agents have two vision cones: a standard one for nearby objects (apples) and a larger one for distant objects (trees), with clarity affected by the ambient light level.

*   **Detailed Visualization:**
    *   **Anatomical Rendering:** Agents are rendered from a top-down perspective with a torso, head, hands, and feet.
    *   **Stateful Animations:** A contralateral gait cycle provides a realistic walking animation tied to speed. Additional animations for sitting, standing up, lying down, sleeping, picking up apples, and eating provide clear visual feedback on agent state.
    *   **Configurable Debug Visuals:** The rendering of vision cones and agent short-term memory locations can be toggled on or off via the `config.json` file, allowing for easier debugging and analysis.
    *   **Modular Pygame Renderer:** All visualization is handled by a dedicated, decoupled renderer.

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
    *   `seed`: The master seed for all random number generation.
*   `environment`: Defines the world and its inhabitants.
    *   `day_night_cycle`: Controls the environment's lighting.
        *   `cycle_duration_steps`: Total steps for one full day-night cycle.
        *   `day_duration_percent`: The percentage of the cycle that is considered "day".
    *   `agent_quantity`: The number of agents to spawn.
    *   `agent_properties`: A dictionary of shared properties for all agents.
        *   **Movement & Physics:** `max_speed`, `max_force`, `friction_strength`, etc.
        *   **Senses:** `vision_range`, `field_of_view`, `large_vision_range`, `threat_detection_range`, etc.
        *   **AI & Behavior:** `wander_distance`, `flocking_strength`, `danger_avoidance_strength`, etc.
        *   **Stamina:** `max_stamina`, `stamina_consumption_rate`, `tired_threshold_percent`, etc.
        *   **Hunger:** `max_hunger`, `hunger_decay_rate`, `hungry_threshold_percent`, etc.
        *   **Vigilance System:** Contains sub-properties like `decay_rate_day`, `decay_rate_night`, `sleep_threshold_percent`, etc.
    *   `obstacles`: Defines procedural generation for static obstacles.
        *   `quantity`: Number of obstacles to create.
        *   `min_radius`, `max_radius`: The size range for generated obstacles.
    *   `apple_trees`: Defines procedural generation for apple trees.
        *   `quantity`: Number of trees to create.
        *   `max_apples`: Max apples a single tree can have at one time.
        *   `apple_spawn_rate`: The probability of spawning an apple per step if not at max.
*   `logging`: Configures the logging system.
    *   `level`: The minimum log level to display (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
    *   `format`: The format string for log messages.

## Future Development Roadmap

*   **Expand Agent Memory System:** Continue transitioning agents from purely reactive entities to more intelligent beings that learn from their experiences. The next layers of this multi-layered system include:
    *   **Long-Term Semantic Memory:** This system will allow agents to build a persistent **mental map** of their environment, representing their factual knowledge of the world ("What do I know?").
        *   **Functionality:** It will store the locations of static, high-value entities like apple trees and major, unmoving obstacles discovered during exploration.
        *   **Behavioral Impact:** This enables **deliberate, long-distance navigation**. A hungry agent far from any visible food source can query its memory for the nearest known tree and travel towards it with purpose, rather than relying on random wandering to find resources. This marks a shift from purely sensory-based action to memory-guided strategy.
    *   **Long-Term Episodic Memory:** This will be the agent's personal history, allowing it to recall specific, significant past events ("What happened to me, where, and when?").
        *   **Functionality:** It will store a log of key life events, such as `(ATE_APPLE, location, timestamp)` or `(LOW_STAMINA_EVENT, location, timestamp)`.
        *   **Behavioral Impact:** Agents will develop **learned preferences and aversions**. An agent might learn to associate a specific region with a high density of food and preferentially return there when hungry. This creates more efficient, individualized foraging patterns based on past successes.
    *   **Long-Term Implicit / Procedural Memory:** This system will represent skill acquisition, allowing agents to gradually refine their behaviors through practice and experience ("How can I do things better?").
        *   **Functionality:** Instead of storing explicit data, this system will slowly tune an agent's internal behavioral parameters (e.g., `danger_avoidance_strength`, `max_speed`, `head_scan_speed`) based on its life history.
        *   **Behavioral Impact:** This will lead to emergent agent **"personalities" or skills**. An agent that has had many collisions might become more "cautious" by increasing its avoidance strength, while an agent that has navigated efficiently for a long time might become more "confident" by slightly increasing its speed. This introduces behavioral diversity driven by individual experience.

*   **Advanced Social Behaviors:** Introduce more complex agent interactions, communication, and group dynamics beyond simple flocking.
*   **Hearing and Smell:** Implement new sensory systems based on distance and propagation.
*   **Genetics and Evolution:** Add a system for agents to reproduce, passing on traits that can be selected for over generations.
*   **Saving & Loading State:** Implement functionality to save the state of a long-running simulation and resume it later.
*   **UI/Dashboard:** Create a simple UI for tweaking simulation parameters in real-time or a dashboard for visualizing metrics.