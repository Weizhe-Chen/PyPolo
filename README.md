<img src="https://raw.githubusercontent.com/Weizhe-Chen/pypolo/main/logo.svg" align="right" width="200" alt="header pic"/>

# What is this?
PyPolo is a Python library for **Robotic Information Gathering**.

![caramel](./docs/images/caramel_gather_info.gif)

This is my fluffy friend -- Caramel -- learning about a drone she has never seen.
She listens attentively, touches and sniffs the drone actively, and changes her angle of view to gather more information about this unknown object.
Can robots also exhibit such *active information acquisition* behavior?

## What is Robotic Information Gathering
Robotic Information Gathering is a research topic in Robotics that aims to answer the following question:

**How does a robot (team) collect *observations* to *efficiently* build an *accurate* model of a physical process under robot *embodiment constraints*?**

To endow robots with the ability to gather valuable information actively, we need to develop an *integrated planning and learning* system. The learning algorithm should provide the uncertainty of its prediction, and the planning algorithm should consider the uncertainty to collect informative observations or make informed decisions.

## Why would I use this library?
* **Interactive Tutorials** to guide you to the world of Robotic Information Gathering.
* **Detailed Documentation & Readable Code** to help you understand the basic idea of each algorithm.
* **Minimum Dependency & Cross-Platform Support** :-) start your robotics journey without getting stuck in [Ubuntu](https://ubuntu.com/) and [Robot Operating System (ROS)](https://www.ros.org/).
* **Modular Implementation** to facilitate the development of new algorithms.
* **ROS**. PyPolo does not depend on ROS, but we provide tutorials on deploying the algorithms to simulated robots in ROS.

# Installation
**Requirements**:

* Python >= 3.6 
* PyTorch >= 1.10.2 (NumPy included)
* Matplotlib

```bash
conda create -n pypolo python=3.8
conda activate pypolo
pip install -e .
```
