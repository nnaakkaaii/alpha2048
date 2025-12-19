"""Setup script for alpha2048 package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alpha2048",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Reinforcement Learning for 2048 game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nnaakkaaii/alpha2048",
    packages=find_packages(include=["pkg", "pkg.*", "training", "training.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Puzzle Games",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.8.0",
        "wandb>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "alpha2048-train=training.reinforcement_learning.train:main",
            "alpha2048-test=training.reinforcement_learning.test:main",
        ],
    },
)