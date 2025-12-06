"""Setup configuration for TalAI Active Learning Experiment Designer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="talai-active-learning",
    version="1.0.0",
    author="AlaweinOS Research",
    author_email="research@alaweinos.com",
    description="Active learning and optimal experimental design system for TalAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlaweinOS/TalAI/tree/main/active-learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "advanced": [
            "GPy>=1.10.0",
            "botorch>=0.6.0",
            "ax-platform>=0.2.0",
            "optuna>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "talai-activelearn=active_learning_engine:main",
        ],
    },
)