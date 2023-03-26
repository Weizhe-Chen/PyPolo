<img src="docs/assets/images/social/logo.svg" align="right" width="150" alt="header pic"/>

# What Is PyPolo? 🧐
PyPolo is a Python library for Robotic Information Gathering (RIG) -- a robotics research area that aims to answer:

> **How does a robot (team) collect informative data to efficiently build an accurate model of an unknown target function under the robot's embodiment constraints?**

<img src="docs/assets/images/framework/venn.png" width="50%"/>

Depending on the definition of the target function, a RIG problem boils down to Active Mapping, Active Localization, Active SLAM, or Active Dynamics Learning.


| Target Function | Problem |
| --------|-----------------------|
| Environment | Active Mapping|
| Robot's Pose | Active Localization |
| Robot's Dynmaics | Active Dynamics Learning |
| Environment & Pose | Active SLAM |

RIG has recently received increasing attention due to its wide applicability.Applications include environmental modeling and monitoring Dunbabin and Marques (2012), 3D reconstruction and inspection Hollinger et al. (2013); Schmid et al. (2020), search and rescue Meera et al. (2019), exploration and mapping Jadidi et al. (2019), as well as active System Identification Buisson-Fenet et al. (2020).

Applications of RIG include but not limited to environmental mapping and monitoring 

For example, an Autonomous Surface Vehicle (ASV) actively collects sonar measurements by minimizing uncertainty to efficiently build an accurate underwater terrain map.


<img src="docs/assets/images/framework/framework.png" width="50%"/>



## Why would I use this library? 🤷

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

RIG systems have highly coupled components.
The planning algorithm depends on the prediction of a probabilistic model, while the model in turn relies on the data collected by the planner and controller.
Due to its interdisciplinary nature, RIG can be daunting for beginners to learn, comparing to studying planning, learning, or control problems independently.
PyPolo is here to help.
We aim to provide an accessible starting point for those looking to learn RIG and jumpstart their research in this domain.
