#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='atlas-sdk',
    version='1.0.0',
    author='ATLAS Platform',
    author_email='support@atlas-platform.com',
    description='Python SDK for ATLAS - Autonomous Technical Leadership & Adaptive System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/atlas-python-sdk',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: System :: Distributed Computing',
    ],
    keywords='atlas ai llm orchestration development automation',
    python_requires='>=3.8',
    install_requires=[
        'requests>=2.28.0',
        'pydantic>=1.10.0',
        'typing-extensions>=4.0.0',
        'asyncio-mqtt>=0.11.0;python_version>="3.7"',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'mypy>=1.0.0',
            'flake8>=5.0.0',
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'analysis': [
            'pandas>=1.5.0',
            'matplotlib>=3.6.0',
            'seaborn>=0.11.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'atlas-cli=atlas.cli:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/your-org/atlas-python-sdk/issues',
        'Source': 'https://github.com/your-org/atlas-python-sdk',
        'Documentation': 'https://atlas-sdk.readthedocs.io/',
    },
)