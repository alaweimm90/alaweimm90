#!/usr/bin/env python3
"""
SciComp Setup
Setup script for PyPI distribution of SciComp.
A cross-platform scientific computing suite for research and education.
"""
from setuptools import setup, find_packages
import os
import re
def read_file(filename):
    """Read file contents."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()
def get_version():
    """Extract version from __init__.py."""
    init_file = os.path.join('Python', '__init__.py')
    if os.path.exists(init_file):
        content = read_file(init_file)
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
        if match:
            return match.group(1)
    return '1.0.0'
def get_long_description():
    """Get long description from README."""
    try:
        return read_file('README.md')
    except FileNotFoundError:
        return "SciComp - Scientific Computing Excellence from UC Berkeley"
def get_requirements():
    """Get requirements from requirements.txt."""
    try:
        requirements = read_file('requirements.txt').splitlines()
        return [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
    except FileNotFoundError:
        return [
            'numpy>=1.20.0',
            'scipy>=1.7.0',
            'matplotlib>=3.3.0',
            'sympy>=1.8',
            'h5py>=3.0.0',
        ]
def get_dev_requirements():
    """Get development requirements."""
    try:
        requirements = read_file('requirements-dev.txt').splitlines()
        return [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
    except FileNotFoundError:
        return [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
        ]
# Package configuration
setup(
    name='berkeley-scicomp',
    version=get_version(),
    # Package information
    description='SciComp - Advanced Scientific Computing Platform',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    # Author and contact
    author='UC Berkeley SciComp Team',
    author_email='scicomp@berkeley.edu',
    maintainer='Meshal Alawein',
    maintainer_email='meshal@berkeley.edu',
    # URLs
    url='https://github.com/berkeley/scicomp',
    project_urls={
        'Documentation': 'https://berkeley-scicomp.readthedocs.io',
        'Bug Reports': 'https://github.com/berkeley/scicomp/issues',
        'Source': 'https://github.com/berkeley/scicomp',
        'Download': 'https://pypi.org/project/berkeley-scicomp/',
    },
    # Package discovery
    packages=find_packages(include=['Python', 'Python.*']),
    package_dir={'': '.'},
    # Include additional files
    include_package_data=True,
    package_data={
        'Python': [
            '*/data/*.dat',
            '*/data/*.json',
            '*/examples/*.py',
            '*/tests/*.py',
        ],
    },
    # Requirements
    python_requires='>=3.8',
    install_requires=get_requirements(),
    # Optional dependencies
    extras_require={
        'gpu': ['cupy-cuda12x>=12.0.0'],
        'ml': [
            'tensorflow>=2.8.0',
            'torch>=1.11.0',
            'scikit-learn>=1.0.0',
        ],
        'performance': [
            'numba>=0.56.0',
            'dask>=2022.0.0',
            'joblib>=1.1.0',
        ],
        'visualization': [
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
            'ipywidgets>=7.6.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'nbsphinx>=0.8.0',
            'jupyter>=1.0.0',
        ],
        'dev': get_dev_requirements(),
        'all': [
            'cupy-cuda12x>=12.0.0',
            'tensorflow>=2.8.0',
            'torch>=1.11.0',
            'scikit-learn>=1.0.0',
            'numba>=0.56.0',
            'dask>=2022.0.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
            'ipywidgets>=7.6.0',
        ],
    },
    # Entry points
    entry_points={
        'console_scripts': [
            'berkeley-scicomp=Python.utils.cli:main',
            'bsc=Python.utils.cli:main',
        ],
    },
    # Classification
    classifiers=[
        # Development Status
        'Development Status :: 5 - Production/Stable',
        # Intended Audience
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        # Topic
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # License
        'License :: OSI Approved :: MIT License',
        # Programming Language
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        # Operating System
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        # Natural Language
        'Natural Language :: English',
        # Environment
        'Environment :: Console',
        'Environment :: Web Environment',
        'Environment :: Other Environment',
    ],
    # Keywords
    keywords=[
        'berkeley', 'scicomp', 'quantum', 'physics', 'scientific-computing',
        'machine-learning', 'gpu-acceleration', 'thermal-transport',
        'quantum-optics', 'spintronics', 'materials-science', 'climate-modeling',
        'university-california-berkeley', 'research', 'education',
    ],
    # Platforms
    platforms=['any'],
    # Zip safe
    zip_safe=False,
)