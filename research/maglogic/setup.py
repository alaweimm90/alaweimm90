#!/usr/bin/env python3
"""
Setup script for MagLogic: Nanomagnetic Logic Simulation Suite

This package provides comprehensive tools for micromagnetic simulation
analysis, focusing on nanomagnetic logic devices and cellular automata.

Author: Meshal Alawein
Email: meshal@berkeley.edu
Institution: University of California, Berkeley
License: MIT
"""

import sys
import os
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    sys.exit('MagLogic requires Python 3.8 or higher')

# Read the README file for long description
here = Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_file = here / 'requirements.txt'
    if not requirements_file.exists():
        return []
    
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                # Remove comment from end of line if present
                if '#' in line:
                    line = line.split('#')[0].strip()
                requirements.append(line)
    return requirements

# Version information
__version__ = '1.0.0'

# Development dependencies
dev_requirements = [
    'pytest>=6.2.0',
    'pytest-cov>=2.12.0',
    'pytest-mock>=3.6.0',
    'pytest-xdist>=2.3.0',
    'black>=21.0.0',
    'flake8>=3.9.0',
    'mypy>=0.910',
    'pre-commit>=2.13.0',
    'bandit>=1.7.0',
    'safety>=1.10.0',
]

# Documentation dependencies
docs_requirements = [
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'myst-parser>=0.15.0',
    'sphinx-autodoc-typehints>=1.12.0',
    'sphinx-gallery>=0.10.0',
    'nbsphinx>=0.8.0',
]

# Machine learning dependencies
ml_requirements = [
    'scikit-learn>=1.0.0',
    'torch>=1.9.0',
    'tensorflow>=2.6.0',
    'xgboost>=1.5.0',
    'lightgbm>=3.2.0',
]

# GUI dependencies
gui_requirements = [
    'PyQt5>=5.15.0',
    'tkinter-page>=0.1.0',
    'pyqtgraph>=0.12.0',
]

# High-performance computing dependencies
hpc_requirements = [
    'mpi4py>=3.1.0',
    'h5py>=3.2.0',
    'zarr>=2.8.0',
    'dask[complete]>=2021.6.0',
]

setup(
    # Basic package information
    name='maglogic',
    version=__version__,
    description='Nanomagnetic Logic Simulation Suite for Computational Magnetism',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # Author and contact information
    author='Meshal Alawein',
    author_email='meshal@berkeley.edu',
    maintainer='Meshal Alawein',
    maintainer_email='meshal@berkeley.edu',
    
    # URLs and project information
    url='https://github.com/alaweimm90/MagLogic',
    download_url='https://github.com/alaweimm90/MagLogic/releases',
    project_urls={
        'Documentation': 'https://alaweimm90.github.io/MagLogic/',
        'Source': 'https://github.com/alaweimm90/MagLogic',
        'Website': 'https://malawein.com',
        'LinkedIn': 'https://www.linkedin.com/in/meshal-alawein',
        'Research': 'https://simcore.dev',
        'Bug Reports': 'https://github.com/alaweimm90/MagLogic/issues',
        'Feature Requests': 'https://github.com/alaweimm90/MagLogic/issues/new?template=feature_request.md',
        'Research Group': 'https://physics.berkeley.edu/',
        'Paper': 'https://doi.org/10.1109/LMAG.2019.2912398',
    },
    
    # Package configuration
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': dev_requirements,
        'docs': docs_requirements,
        'ml': ml_requirements,
        'gui': gui_requirements,
        'hpc': hpc_requirements,
        'all': dev_requirements + docs_requirements + ml_requirements + gui_requirements + hpc_requirements,
    },
    
    # Entry points for command-line tools
    # TODO: Implement command-line interface modules
    entry_points={
        'console_scripts': [
            # 'maglogic=maglogic.cli:main',
            # 'maglogic-convert=maglogic.scripts.convert_formats:main',
            # 'maglogic-analyze=maglogic.scripts.batch_analysis:main',
            # 'maglogic-gui=maglogic.gui.maglogic_dashboard:main',
            # 'maglogic-benchmark=maglogic.scripts.benchmark_performance:main',
        ],
    },
    
    # Package data - include data files in the package
    package_data={
        'maglogic': [
            'data/*.json',
            'data/*.yaml',
            'data/*.csv',
            'style/*.json',
            'style/*.mplstyle',
            'templates/*.template',
            'schemas/*.json',
        ],
    },
    
    # Include additional files specified in MANIFEST.in
    include_package_data=True,
    
    # Classification for PyPI
    classifiers=[
        # Development Status
        'Development Status :: 4 - Beta',
        
        # Intended Audience
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        
        # Topic
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
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
        'Environment :: X11 Applications :: Qt',
    ],
    
    # Keywords for discovery
    keywords=[
        'micromagnetism', 'nanomagnetic logic', 'computational physics',
        'OOMMF', 'MuMax3', 'cellular automata', 'spintronics',
        'magnetic simulation', 'scientific computing', 'materials science',
        'condensed matter physics', 'magnetism', 'spin dynamics',
        'logic gates', 'beyond CMOS', 'magnetic memory',
        'artificial intelligence', 'machine learning', 'data analysis'
    ],
    
    # Additional metadata
    license='MIT',
    platforms=['any'],
    zip_safe=False,
    
    # Minimum versions for critical dependencies
    python_requires='>=3.8',
    
    # Specify which Python implementations are supported
    # CPython is the main target, but PyPy compatibility is desired
    
    # Command line tools configuration
    scripts=[],  # We use entry_points instead
    
    # Test suite
    test_suite='tests',
    tests_require=[
        'pytest>=6.2.0',
        'pytest-cov>=2.12.0',
        'pytest-mock>=3.6.0',
    ],
    
    # Additional options
    options={
        'bdist_wheel': {
            'universal': False,  # Not universal because of potential C extensions
        },
        'egg_info': {
            'tag_build': '',
            'tag_date': False,
        },
    },
)