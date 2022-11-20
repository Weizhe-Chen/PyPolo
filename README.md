<img src="https://raw.githubusercontent.com/Weizhe-Chen/pypolo/main/logo.svg" align="right" width="200" alt="header pic"/>

# What is this? 🧐
PyPolo is a Python library for <b>Robotic Information Gathering</b>

<img src="./docs/images/caramel_gather_info.gif" width="200"/>

This is my fluffy friend -- Caramel -- learning about a drone she has never seen.
She **listens attentively, touches and sniffs the drone actively, and changes her angle of view** to gather more information about this unknown object.

Can robots also exhibit such *active information acquisition* behavior?

## What is Robotic Information Gathering? 🤖
Robotic Information Gathering is a robotics research topic that aims to solve the following embodied active learning problem:

**How does a robot (team) collect *observations* to *efficiently* build an *accurate* model of a physical process under robot *embodiment constraints*?**

For example, an Autonomous Surface Vehicle (ASV) actively collects sonar measurements/observations by minimizing uncertainty to efficiently build an accurate underwater terrain map.

<img src="./docs/images/rig_framework.png" width="400"/>

Other applications include but not limited to Autonomous Exploration, 3D Reconstruction or Inspection, Search and Rescue, Environmental Modeling and Monitoring, Active Dynamics Learning, and Active Localization.

## Why would I use this library? 🤷
In a Robotic Information Gathering system, planning relies on the learning component, while learning depends on the data collected by the planner and controller.
Compared to studying planning, learning, or control problems alone, the interdisciplinary nature of Robotic Information Gathering can make it relatively daunting for beginners to get started.
I hope PyPolo can lower the entry bar of this domain.

You might be interested in this library if
* you would like to learn more about Robotic Information Gathering;
* you need to prototype or verify your research idea, focusing only on learning or planning;
* you are looking for some baseline algorithms for benchmarking;
* you are learning some algorithms implemented in this library, e.g., Gaussian process regression.

PyPolo provides
* **interactive tutorials** to guide you to the world of Robotic Information Gathering,
* **detailed documentation and readable code** to help you understand the basic idea of each algorithm,
* **minimum dependency and cross-platform support** so that you can start your robotics journey without getting stuck in [Ubuntu](https://ubuntu.com/) and [Robot Operating System (ROS)](https://www.ros.org/),
* **modular implementation** to facilitate the development of new algorithms,
* **ROS integration guide** to demonstrate how to algorithms can be deployed to simulated/real robots.

# Installation 📥
1. Create a virtual environment
    ```bash
    conda create -n rig python=3.8
    conda activate rig
    ```
2. Clone this repository
    ```bash
    git clone https://github.com/Weizhe-Chen/PyPolo.git
    ```
3. Install PyPolo
    ```bash
    pip install -e .
    ```

# Get Started ⭐
