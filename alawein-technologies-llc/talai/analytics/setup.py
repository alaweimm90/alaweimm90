"""Setup configuration for TalAI Analytics package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="talai-analytics",
    version="1.0.0",
    author="TalAI Team",
    author_email="talai@alawein.com",
    description="Advanced analytics and intelligence system for TalAI hypothesis validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlaweinOS/TalAI/analytics",
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
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "ruff>=0.0.261",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "talai-analytics=analytics.main:main",
        ],
    },
)