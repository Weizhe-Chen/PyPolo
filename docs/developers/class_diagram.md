# Class Diagram

``` mermaid
classDiagram
direction LR
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

Robot --* Task
Sensor --* Robot
Model --* Task

Model --> Objective
Objective --> Planner
Planner --> Controller
Controller --> Task
```
