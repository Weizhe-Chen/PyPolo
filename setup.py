import os
import setuptools

root_dir = os.path.dirname(os.path.realpath(__file__))

# dependencies
INSTALL_REQUIRES = [
    "torch>=1.13.1",
    "tensorboard",
    "matplotlib",
    "tqdm",
    "pyvista",
]

setuptools.setup(
    name="pypolo",
    version="0.0.4",
    author="PyPolo Developers",
    author_email="chenweiz@iu.edu",
    description="PyPolo: A Python Library for Robotic Information Gathering",
    long_description=open(os.path.join(root_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    keywords=[
        "robotics", "machine", "learning", "information", "gathering",
        "planning", "acquisition", "exploration", "active"
    ],
    python_requires='>=3.6',
    install_requires=INSTALL_REQUIRES,
    url="https://github.com/Weizhe-Chen/PyPolo",
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    license='MIT',
    project_urls={
        "Documentation": "https://pypolo.readthedocs.io",
        "Repository": "https://github.com/Weizhe-Chen/PyPolo",
        "Bug Tracker": "https://github.com/Weizhe-Chen/PyPolo/issues",
        "Discussions": "https://github.com/Weizhe-Chen/PyPolo/discussions",
    },
)
