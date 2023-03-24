
# Framework

![framework](./assets/framework.png){: style="height:100%;width:100%"}

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
class Policy {
    +act()
}

Robot --* Task: Composition
Sensor --* Robot: Composition
Model --* Task: Composition

Model --> Objective: Prediction & Uncertainty
Objective --> Planner: Reward / Cost
Objective --> Policy: Reward / Cost
Planner --> Controller: Waypoints
Controller --> Task: Action
Policy --> Task: Action
```

## Models

- [x] Gaussian Process Regression
- [ ] Occupanocy Grid Map

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

- [ ] Active Localization
- [ ] Active Mapping
    - [ ] Building an occupancy in an unknown environment
    - [x] Building a map of a target variable, e.g. underwater terrrain elevation, without considering obstacles
- [ ] Active Simultaneous Localization and Mapping (SLAM)
- [ ] Autonomous Inspection
- [ ] Search and Rescue
