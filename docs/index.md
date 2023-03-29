![code quality](https://img.shields.io/codefactor/grade/github/Weizhe-Chen/PyPolo/main)
![build](https://github.com/Weizhe-Chen/PyPolo/actions/workflows/build.yml/badge.svg)
![coverage](https://img.shields.io/codecov/c/github/Weizhe-Chen/PyPolo)
![docs](https://img.shields.io/readthedocs/pypolo)
![license](https://img.shields.io/github/license/Weizhe-Chen/PyPolo)
![python](https://img.shields.io/pypi/pyversions/pypolo)

## What Is PyPolo? 🧐
PyPolo is a Python library for Robotic Information Gathering (RIG) -- a robotics research area that aims to answer:

> **How does a robot (team) collect informative data to efficiently build an accurate model of an unknown target function under the robot's embodiment constraints?**

<img src="assets/images/framework/venn.png" width="70%"/>

Depending on the definition of the target function, a RIG problem boils down to Active Mapping, Active Localization, Active SLAM, or Active Dynamics Learning.


| Target Function | Problem |
| --------|-----------------------|
| Robot's Dynmaics | Active Dynamics Learning |
| Robot's Pose | Active Localization |
| Environment | Active Mapping|
| Environment & Pose | Active SLAM |

RIG has gained significant attention lately due to its broad range of applications, including environmental modeling and monitoring, 3D reconstruction and inspection, search and rescue, autonomous exploration and mapping, among others.

An illustration of RIG's potential use case is an Autonomous Surface Vehicle (ASV) that efficiently constructs a precise underwater terrain map by actively gathering sonar measurements and minimizing uncertainty.

<img src="assets/images/framework/framework.png" width="70%"/>

## Why Would I Use PyPolo? 🤷

You might be interested in this library if you would like to

* **Learn** some robotics algorithms related to RIG painlessly;
* **Research** and quickly prototype RIG algorithms, e.g., new planner, controller, model;
* **Benchmark** your proposed algorithm with some popular baselines.

We keep these needs in mind and provide

* **Intuitive tutorials** to guide you to the world of RIG;
* **Detailed documentation and readable code** to help you understand the basic idea of each algorithm;
* **Minimum dependency and cross-platform support** so that you can start your robotics journey painlessly;
* **Modular implementation** to facilitate the development of new algorithms;
* **ROS integration guide** to demonstrate how to deploy the algorithms to simulated or real robots.
* **A list of awesome** books, literature, related researchers, and other resources.

Learning RIG can be daunting for beginners due to its highly integrated learning, planning, and control modules.
Our aim is to provide an accessible starting point for those looking to learn RIG and jumpstart their research in this exciting domain.
