---
comments: true
---

# Installation

## Virtual Environment Manager

We use Python virtual environment manager such as `conda` or `mamba` for several reasaons:

1. **Package management**: `conda` or `mamba` are package managers that can help you install, update, and manage Python packages with ease.
2. **Environment management**: when working on multiple projects or collaborating with others, you may encounter situations where different projects require different versions of Python or its dependencies. `conda` or `mamba` can create isolated environments that allow you to switch between different versions of Python and packages without conflicting with each other.
3. **Cross-platform compatibility**:  `conda` and mamba are cross-platform tools that can help you manage Python packages and environments on Windows, MacOS, and Linux systems. This makes it easier to develop and deploy Python applications across different operating systems.

We recommend using [`miniforge` / `mambaforge`](https://github.com/conda-forge/miniforge) or [`miniconda`](https://docs.conda.io/en/latest/miniconda.html) instead of the `Anaconda` installer because they are more light-weight.
If you have not installed `conda` or `mamba` yet, we recommend installing `mamba` using the following installers.


=== "Linux"

    | OS      | Architecture          | Download  |
    | --------|-----------------------|-----------|
    | Linux   | x86_64 (amd64)        | [Mambaforge-Linux-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh) |
    | Linux   | aarch64 (arm64)       | [Mambaforge-Linux-aarch64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-aarch64.sh) |
    | Linux   | ppc64le (POWER8/9)    | [Mambaforge-Linux-ppc64le](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-ppc64le.sh) |

=== "MacOS X"

    | OS      | Architecture          | Download  |
    | --------|-----------------------|-----------|
    | OS X    | x86_64                | [Mambaforge-MacOSX-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-x86_64.sh) |
    | OS X    | arm64 (Apple Silicon) | [Mambaforge-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-arm64.sh) |

=== "Windows"

    | OS      | Architecture          | Download  |
    | --------|-----------------------|-----------|
    | Windows | x86_64                | [Mambaforge-Windows-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe) |


!!! note "Why choose `mamba` over `conda`?"

    Mamba is a high-speed package manager that is designed to be a drop-in replacement for conda.
    It can provide faster package installations and updates, especially for large packages or complex dependencies.
    Mamba also has a more efficient dependency solver that can handle complex dependency graphs with fewer conflicts than conda.

If you have already installed `conda`, you can also install `mamba` by

```bash
conda install mamba -c conda-forge
```

Now we can create a virtual environment for the PyPolo project:

=== "Mamba"

    ```
    mamba create -n pypolo python=3.8
    mamba activate pypolo
    ```

=== "Conda"

    ```
    conda create -n pypolo python=3.8
    conda activate pypolo
    ```

## Install PyPolo

```
git clone https://github.com/Weizhe-Chen/PyPolo.git
cd PyPolo
pip install -e .
```

If you would like to become a PyPolo developer and contribute to the project, please make sure to install the following development tools by running the command:

```
pip install -r requirements_dev.txt
```

This command will install all the necessary dependencies needed for development specified in the `requirements_dev.txt` file.
