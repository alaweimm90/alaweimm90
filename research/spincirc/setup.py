#!/usr/bin/env python3
"""
SpinCirc Setup Configuration
Professional setup script for SpinCirc Python components
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("python/requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
setup(
    name="spincirc",
    version="1.0.0",
    author="Meshal Alawein",
    author_email="meshal@berkeley.edu",
    description="Advanced Spin Transport Circuit Framework for Computational Spintronics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/alawein/SpinCirc",
    project_urls={
        "Documentation": "https://docs.spincirc.org",
        "Source": "https://github.com/alawein/SpinCirc",
        "Bug Tracker": "https://github.com/alawein/SpinCirc/issues",
        "Discussions": "https://github.com/alawein/SpinCirc/discussions",
        "Website": "https://malawein.com",
        "LinkedIn": "https://www.linkedin.com/in/meshal-alawein",
        "Research": "https://simcore.dev",
        "Scholar": "https://scholar.google.com/citations?user=IB_E6GQAAAAJ&hl=en",
    },
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.13.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "ml": [
            "tensorflow>=2.6.0",
            "torch>=1.9.0",
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spincirc-process=analysis.data_processor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords=[
        "spintronics",
        "computational physics",
        "spin transport",
        "magnetization dynamics",
        "device modeling",
        "LLG equation",
        "Berkeley",
        "quantum computing",
        "materials science",
    ],
    license="MIT",
    platforms=["any"],
    zip_safe=False,
)