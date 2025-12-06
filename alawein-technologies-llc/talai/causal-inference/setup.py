"""Setup configuration for TalAI Causal Inference Engine."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="talai-causal-inference",
    version="1.0.0",
    author="AlaweinOS Research",
    author_email="research@alaweinos.com",
    description="Advanced causal inference and discovery system for TalAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlaweinOS/TalAI/tree/main/causal-inference",
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
        "networkx>=2.6.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.12.0",
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
            "dowhy>=0.7",
            "causalml>=0.12",
            "pgmpy>=0.1.14",
        ],
    },
    entry_points={
        "console_scripts": [
            "talai-causal=causal_engine:main",
        ],
    },
)