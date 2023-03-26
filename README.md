<img src="docs/assets/images/social/logo.svg" align="right" width="200" alt="header pic"/>

# What is this? 🧐
PyPolo is a Python library for <b>Robotic Information Gathering (RIG)</b>

## What is Robotic Information Gathering? 🤖
RIG is a robotics research topic that aims to solve the following embodied active learning problem:

**How does a robot (team) collect *observations* to *efficiently* build an *accurate* model of a physical process under robot *embodiment constraints*?**

<img src="docs/assets/images/framework/framework.png" align="right" width="50%" alt="header pic"/>
For example, an Autonomous Surface Vehicle (ASV) actively collects sonar measurements by minimizing uncertainty to efficiently build an accurate underwater terrain map.
<br/>

<img src="docs/assets/images/framework/venn.png" width="400"/>

In addition to Active Mapping, RIG also includes Active Localization, Active SLAM, and Active Dynamics Learning.

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
