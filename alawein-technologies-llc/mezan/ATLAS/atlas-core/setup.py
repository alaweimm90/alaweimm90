from setuptools import setup, find_packages

setup(
    name="ORCHEX-core",
    version="0.1.0",
    description="ORCHEX Engine - Multi-Agent AI Research Orchestration System",
    author="ORCHEX Team",
    author_email="meshal@berkeley.edu",
    packages=find_packages(),
    install_requires=[
        "redis>=4.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pytest>=7.0.0",
        "anthropic>=0.18.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
