#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup

def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        print(version_match.group(1))
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("pypolo", "__init__.py")


torch_min = "1.10"
install_requires = [
    ">=".join(["torch", torch_min]), 
    "matplotlib",
    "bezier",
    "scipy",
    "configargparse",
    "SciencePlots",
    "tqdm",
    "scikit-image",
]
try:
    import torch

    if torch.__version__ >= torch_min:
        install_requires = []
except ImportError:
    pass

setup(
    name="pypolo",
    version=version,
    author="Weizhe (Wesley) Chen",
    author_email="chenweiz@iu.edu",
    description="A library for informative planning and learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Weizhe-Chen/pypolo",
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require={
        "dev": ["yapf"],
        "docs": ["sphinx"],
        "examples": ["ipython", "jupyter"],
        "test": ["pytest"],
    },
    test_suite="test",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ],
)
