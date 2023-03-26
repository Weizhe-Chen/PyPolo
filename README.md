<img src="docs/assets/images/social/logo.svg" align="right" width="150" alt="header pic"/>

# What Is PyPolo? 🧐
PyPolo is a Python library for Robotic Information Gathering (RIG) -- a robotics research area that aims to answer:

> **How does a robot (team) collect informative data to efficiently build an accurate model of an unknown target function under the robot's embodiment constraints?**

<img src="docs/assets/images/framework/venn.png" width="50%"/>

Depending on the definition of the target function, a RIG problem boils down to Active Mapping, Active Localization, Active SLAM, or Active Dynamics Learning.


| Target Function | Problem |
| --------|-----------------------|
| Robot's Dynmaics | Active Dynamics Learning |
| Robot's Pose | Active Localization |
| Environment | Active Mapping|
| Environment & Pose | Active SLAM |

RIG has gained significant attention lately due to its broad range of applications, including environmental modeling and monitoring, 3D reconstruction and inspection, search and rescue, autonomous exploration and mapping, among others.
An illustration of RIG's potential use case is an Autonomous Surface Vehicle (ASV) that efficiently constructs a precise underwater terrain map by actively gathering sonar measurements and minimizing uncertainty.

<img src="docs/assets/images/framework/framework.png" width="50%"/>

## Why Would I Use PyPolo? 🤷

You might be interested in this library if you would like to

* **Learn** some robotics algorithms related to RIG painlessly;
* **Research** and quickly prototype RIG algorithms, e.g., new planner, controller, model;
* **Benchmark** your proposed algorithm with some popular baselines.

We keep these needs in mind and provide

* **Interactive tutorials** to guide you to the world of RIG;
* **Detailed documentation and readable code** to help you understand the basic idea of each algorithm;
* **Minimum dependency and cross-platform support** so that you can start your robotics journey painlessly on MacOS, Windows, Linux;
* **Modular implementation** to facilitate the development of new algorithms;
* **ROS integration guide** to demonstrate how to deploy the algorithms to simulated or real robots.
* **A list of awesome** books, literature, related researchers, and other resources.

RIG systems have highly coupled components.
The planning algorithm depends on the prediction of a probabilistic model, while the model in turn relies on the data collected by the planner and controller.
Due to its interdisciplinary nature, RIG can be daunting for beginners to learn, comparing to studying planning, learning, or control problems independently.
PyPolo is here to help.
We aim to provide an accessible starting point for those looking to learn RIG and jumpstart their research in this domain.
