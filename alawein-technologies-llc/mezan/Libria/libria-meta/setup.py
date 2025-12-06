"""
Librex.Meta - Tournament-based Meta-Learning for Solver Selection
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="libria-meta",
    version="0.1.0",
    author="Itqān Libria Suite",
    description="Tournament-based meta-learning for multi-agent solver selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "redis>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.6.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
)
