# NeuralNav: TD3 Autonomous Car Navigation

NeuralNav is a high-performance autonomous navigation system built on the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** architecture. The agent learns to navigate a custom map, following a sequence of targets while avoiding obstacles and optimizing its speed.

## 🚀 Key Features

-   **Deep RL Architecture**: Implements the state-of-the-art **TD3** algorithm, featuring:
    -   **Twin Critics**: Dual Q-networks to mitigate overestimation bias.
    -   **Target Policy Smoothing**: Clipped noise added to target actions for robust learning.
    -   **Delayed Policy Updates**: Actor and target networks update less frequently for increased stability.
-   **Continuous Control**:
    -   **Direction**: Smooth, precise steering rather than discrete steps.
    -   **Speed**: Dynamic throttle control based on terrain and mission progress.
-   **Enhanced Perception**:
    -   **LIDAR-style Raycasting**: 7 continuous distance sensors providing granular obstacle awareness.
    -   **Trigonometric Heading**: State representation using `sin/cos` of target orientation to eliminate mathematical discontinuities.
-   **Advanced UI**:
    -   Built with **PyQt6**, featuring real-time visualization of sensors, targets, and neural network metrics.
    -   Integrated charts for Rewards, Loss, and Epsilon decay.

## 🛠️ Architecture Details

### State Space (10 Dimensions)
1.  **7x LIDAR Sensors**: Normalized distance to obstacles.
2.  **Target Sine**: `sin(target_angle - car_angle)`.
3.  **Target Cosine**: `cos(target_angle - car_angle)`.
4.  **Distance**: Normalized distance to the current target.

### Action Space (2 Dimensions)
1.  **Steering**: Range `[-45°, 45°]`.
2.  **Speed**: Range `[0.5, 3.0]` pixels per step.

### Reward System
-   **Heading Bonus**: Encourages the agent to face the target at all times.
-   **Progress Reward**: Scaled incentive for closing distance efficiently when properly aligned.
-   **Surface Awareness**: Speed multipliers for high-speed (Red) road surfaces.
-   **Crash Penalty**: Large negative reward for hitting boundaries or non-road terrain.

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- PyTorch
- NumPy
- PyQt6

### Running the Simulator
```bash
python citymap_assignment.py
```

### Usage
1.  **Setup**: Click on the map to place the **Car**.
2.  **Targets**: Click multiple times to place a sequence of **Targets**.
3.  **Finish Setup**: Right-click once all targets are placed.
4.  **Start/Pause**: Press **Space** or the "Start" button to begin training.

## 📊 Monitoring
Watch the real-time logs and charts to observe:
-   **Loss**: Should downward trend as the Critic stabilizes.
-   **Total Reward**: Increases as the car learns efficient paths.
-   **Epsilon**: Decays to shift the car from exploration to exploitation.

---
*Developed as part of the ERAv4 S17 Assignment.*
