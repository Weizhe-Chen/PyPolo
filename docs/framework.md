---
comments: true
---

# Framework

![framework](./assets/images/framework/framework.png){: style="height:100%;width:100%"}

## Class Diagram

``` mermaid
classDiagram
class Robot {
    +Array state
        +transition()
}
class Sensor {
    +sense()
}
class Task {
    +Robot robot
    +Sensor sensor
    +Array target_map
    +Array obstacle_map
    +float dt
    +step()
}
class Model {
    +add_data()
    +learn()
    +predict()
}
class Objective {
    +eval()
}
class Planner {
    +plan()
}
class Controller {
    +control()
}

Robot --* Task: Composition
Sensor --* Robot: Composition
Model --* Task: Composition

Model --> Objective: Prediction & Uncertainty
Objective --> Planner: Reward / Cost
Planner --> Controller: Waypoints
Controller --> Task: Action
```

## Models

- [x] Gaussian Process Regression (GPR)
- [ ] Occupanocy Grid Map (OGM)

## Objectives

- [x] Entropy

## Planners

- [x] Myopic Planner
- [ ] Monte Carlo Tree Search (MCTS)
- [ ] Rapidly-Exploring Random Trees (RRT)
- [ ] Bayesian Optimization (BO)
- [ ] Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

## Controllers

- [ ] Proportional Derivative (PD) Controller
- [ ] Model Predictive Controller (MPC)

## Sensors

- [x] Single-Beam Ranger
- [ ] Multi-Beam Ranger

## Robots

- [ ] Omni-Directional Robot
- [x] Differential-Drive Robot
- [ ] Ackermann-Steering Robot


## Tasks

![framework](./assets/images/framework/venn.png){: style="height:100%;width:100%"}

- [ ] Navigation
- [x] Active Mapping
    - [x] Environmental Mapping
    - [ ] Occupancy Mapping
- [ ] Active Localization
- [ ] Active Simultaneous Localization and Mapping (SLAM)
- [ ] Autonomous Inspection
- [ ] Search and Rescue
