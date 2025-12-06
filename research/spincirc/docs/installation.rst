Installation Guide
==================

This comprehensive guide covers all installation methods for the SpinCirc framework, from quick Docker setup to full development environment configuration.

.. contents::
   :local:
   :depth: 2

Quick Start (Docker) - Recommended
-----------------------------------

The fastest way to get SpinCirc running is using Docker:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/alaweimm90/SpinCirc.git
   cd SpinCirc
   
   # Build Docker image
   docker build -t spincirc:latest .
   
   # Run interactive session
   docker run -it --rm spincirc:latest bash
   
   # Start Jupyter Lab
   docker run -it --rm -p 8888:8888 spincirc:latest jupyter

**Benefits of Docker Installation:**
   - ✅ Zero dependency management
   - ✅ Consistent environment across platforms
   - ✅ Includes all scientific libraries
   - ✅ Works on Windows, macOS, and Linux

System Requirements
-------------------

**Minimum Requirements**
   - 4 GB RAM
   - 2 GB free disk space
   - Modern CPU (2+ cores recommended)
   - Internet connection for installation

**Recommended Requirements**
   - 8+ GB RAM
   - 10+ GB free disk space
   - Multi-core CPU (4+ cores)
   - GPU (optional, for ML acceleration)

**Supported Operating Systems**
   - Ubuntu 20.04+ / Debian 10+
   - CentOS/RHEL 8+ / Fedora 34+
   - macOS 10.15+ (Intel/ARM)
   - Windows 10+ with WSL2

MATLAB Installation
-------------------

**Prerequisites**

SpinCirc requires MATLAB R2019b or later. Recommended version: **R2023a** or newer.

**Required Toolboxes:**
   - Signal Processing Toolbox
   - Statistics and Machine Learning Toolbox
   - Optimization Toolbox (recommended)
   - Parallel Computing Toolbox (recommended)
   - Curve Fitting Toolbox (optional)

**Installation Steps**

1. **Clone Repository**

   .. code-block:: bash

      git clone https://github.com/alaweimm90/SpinCirc.git
      cd SpinCirc

2. **Add to MATLAB Path**

   .. code-block:: matlab

      % Method 1: Automatic setup
      run('setup.m');
      
      % Method 2: Manual setup
      addpath(genpath('matlab'));
      savepath; % Save path permanently

3. **Verify Installation**

   .. code-block:: matlab

      % Run test suite
      results = runtests('matlab/tests', 'IncludeSubfolders', true);
      disp(results);
      
      % Quick functionality test
      solver = SpinTransportSolver();
      disp('✓ SpinCirc MATLAB installation successful');

4. **Apply Berkeley Styling** (Optional)

   .. code-block:: matlab

      berkeley(); % Apply UC Berkeley color scheme
      
      % Test plotting
      figure;
      plot(1:10, rand(10,1));
      title('Test Plot with Berkeley Theme');

**Alternative: GNU Octave**

For open-source compatibility, SpinCirc supports GNU Octave 6.0+:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt install octave octave-signal octave-statistics
   
   # macOS
   brew install octave
   
   # Start Octave and add path
   octave
   >> addpath(genpath('matlab'));
   >> pkg load signal statistics

.. note::
   Some advanced features may not be available in Octave. MATLAB is recommended for full functionality.

Python Installation
-------------------

**Option 1: Conda (Recommended)**

.. code-block:: bash

   # Create conda environment
   conda create -n spincirc python=3.10
   conda activate spincirc
   
   # Install scientific stack
   conda install numpy scipy matplotlib jupyter
   conda install pytorch torchvision torchaudio -c pytorch
   
   # Install additional dependencies
   pip install -r python/requirements.txt

**Option 2: pip with Virtual Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv spincirc-env
   source spincirc-env/bin/activate  # Linux/macOS
   # spincirc-env\Scripts\activate    # Windows
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r python/requirements.txt

**Python Dependencies**

Core scientific stack:

.. code-block:: text

   numpy>=1.20.0
   scipy>=1.7.0
   matplotlib>=3.5.0
   pandas>=1.3.0
   scikit-learn>=1.0.0
   jupyter>=1.0.0
   ipywidgets>=7.6.0

Machine learning stack:

.. code-block:: text

   torch>=1.12.0
   torchvision>=0.13.0
   optuna>=3.0.0
   bayesian-optimization>=1.4.0
   deap>=1.3.0

Visualization and UI:

.. code-block:: text

   plotly>=5.10.0
   seaborn>=0.11.0
   ipywidgets>=7.6.0
   tqdm>=4.60.0

**Verification**

.. code-block:: python

   # Test basic functionality
   import sys
   import os
   sys.path.append('python')
   
   # Test imports
   import ml_tools
   import visualization
   from ml_tools import ParameterExtractor
   from visualization import BerkeleyPlots
   
   print("✓ SpinCirc Python modules imported successfully")
   
   # Test GPU availability (optional)
   import torch
   if torch.cuda.is_available():
       print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
   else:
       print("ℹ CUDA not available, using CPU")

Docker Installation (Detailed)
-------------------------------

**System Requirements**
   - Docker Desktop 4.0+ (Windows/macOS) or Docker Engine 20.0+ (Linux)
   - 4GB RAM available to Docker
   - 10GB free disk space

**Installation Steps**

1. **Build Image**

   .. code-block:: bash

      # Basic build
      docker build -t spincirc:latest .
      
      # Build with custom tag
      docker build -t spincirc:1.0.0 .
      
      # Build specific target
      docker build --target python-env -t spincirc:python .

2. **Using Build Script**

   .. code-block:: bash

      # Make build script executable
      chmod +x build-docker.sh
      
      # Basic build
      ./build-docker.sh
      
      # Build and test
      ./build-docker.sh --test
      
      # Build development version
      ./build-docker.sh --dev

3. **Docker Compose Setup**

   .. code-block:: bash

      # Start all services
      docker-compose up -d
      
      # Start only Jupyter
      docker-compose up jupyter
      
      # Run interactive session
      docker-compose run --rm dev

**Available Docker Images**

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Image
     - Target
     - Contents
   * - ``spincirc:base``
     - ``base``
     - Ubuntu 22.04 with system dependencies
   * - ``spincirc:python``
     - ``python-env``
     - Python scientific stack
   * - ``spincirc:matlab``
     - ``matlab-env``
     - GNU Octave + MATLAB compatibility
   * - ``spincirc:latest``
     - ``final``
     - Complete SpinCirc environment

**Docker Usage Examples**

.. code-block:: bash

   # Run interactive Python
   docker run -it --rm spincirc:latest python
   
   # Run Jupyter with persistent notebooks
   docker run -d --name spincirc-jupyter \
     -p 8888:8888 \
     -v $(pwd)/notebooks:/opt/spincirc/notebooks \
     spincirc:latest jupyter
   
   # Run tests
   docker run --rm spincirc:latest tests
   
   # Run with data volumes
   docker run --rm \
     -v $(pwd)/data:/opt/spincirc/data \
     -v $(pwd)/results:/opt/spincirc/results \
     spincirc:latest python my_script.py

Development Installation
------------------------

For contributors or advanced users who want to modify SpinCirc:

**Prerequisites**
   - Git 2.20+
   - MATLAB R2021a+ or GNU Octave 6.0+
   - Python 3.10+
   - Docker (optional)

**Setup Development Environment**

.. code-block:: bash

   # Clone with full history
   git clone --recursive https://github.com/alaweimm90/SpinCirc.git
   cd SpinCirc
   
   # Create development branch
   git checkout -b feature/my-feature
   
   # Install in development mode
   pip install -e python/
   
   # Install pre-commit hooks
   pip install pre-commit
   pre-commit install

**Development Dependencies**

.. code-block:: bash

   # Install development tools
   pip install black flake8 pytest pytest-cov
   pip install sphinx sphinx-rtd-theme myst-parser
   
   # MATLAB development tools
   # Install MATLAB Coder, Test Framework, etc.

**Running Tests**

.. code-block:: bash

   # Python tests
   pytest python/tests/ -v --cov
   
   # MATLAB tests
   matlab -batch "addpath('matlab'); runtests('matlab/tests')"
   
   # Docker tests
   ./build-docker.sh --test

**Code Formatting**

.. code-block:: bash

   # Python formatting
   black python/
   flake8 python/
   
   # MATLAB formatting (using MATLAB Editor)
   # Follow guidelines in development/coding_standards.rst

Platform-Specific Instructions
------------------------------

**Windows 10/11**

1. **Enable WSL2** (Recommended)

   .. code-block:: powershell

      # Run in Administrator PowerShell
      wsl --install -d Ubuntu-22.04
      wsl --set-default-version 2

2. **Install in WSL2**

   .. code-block:: bash

      # Inside WSL2 Ubuntu
      sudo apt update
      sudo apt install python3-pip git
      git clone https://github.com/alaweimm90/SpinCirc.git
      cd SpinCirc
      pip3 install -r python/requirements.txt

3. **MATLAB Integration**

   .. code-block:: matlab

      % In Windows MATLAB
      addpath('\\wsl$\Ubuntu-22.04\home\username\SpinCirc\matlab');

**macOS**

.. code-block:: bash

   # Install Homebrew (if not installed)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install dependencies
   brew install python@3.10 git
   brew install --cask matlab  # If available
   
   # Install SpinCirc
   git clone https://github.com/alaweimm90/SpinCirc.git
   cd SpinCirc
   pip3 install -r python/requirements.txt

**Linux (Ubuntu/Debian)**

.. code-block:: bash

   # Update system
   sudo apt update && sudo apt upgrade
   
   # Install dependencies
   sudo apt install python3-pip python3-venv git
   sudo apt install octave octave-signal octave-statistics  # Optional
   
   # Install SpinCirc
   git clone https://github.com/alaweimm90/SpinCirc.git
   cd SpinCirc
   python3 -m venv venv
   source venv/bin/activate
   pip install -r python/requirements.txt

**Linux (CentOS/RHEL/Fedora)**

.. code-block:: bash

   # CentOS/RHEL 8+
   sudo dnf install python3-pip python3-venv git
   
   # Fedora
   sudo dnf install python3-pip python3-virtualenv git
   
   # Install SpinCirc (same as Ubuntu)

High-Performance Computing (HPC) Installation
----------------------------------------------

For cluster environments with limited internet access:

**Offline Installation**

1. **Download Dependencies**

   .. code-block:: bash

      # On machine with internet
      pip download -r python/requirements.txt -d packages/
      
      # Transfer packages/ directory to HPC system

2. **Install on HPC**

   .. code-block:: bash

      # On HPC system
      module load python/3.10  # Load Python module
      python -m venv spincirc-env
      source spincirc-env/bin/activate
      pip install --no-index --find-links packages/ -r python/requirements.txt

**Singularity/Apptainer Support**

.. code-block:: bash

   # Build Singularity image
   sudo singularity build spincirc.sif docker://spincirc:latest
   
   # Run with Singularity
   singularity exec spincirc.sif python my_script.py

**SLURM Integration**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=spincirc
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=8
   #SBATCH --time=24:00:00
   
   module load python/3.10
   source spincirc-env/bin/activate
   
   python large_simulation.py

Verification and Testing
------------------------

**Complete Installation Test**

.. code-block:: bash

   # Run comprehensive tests
   cd SpinCirc
   
   # Python tests
   python -c "
   import sys; sys.path.append('python')
   import ml_tools, visualization
   from ml_tools import ParameterExtractor
   print('✓ Python modules OK')
   "
   
   # MATLAB tests (if available)
   matlab -batch "
   addpath('matlab');
   runtests('matlab/tests');
   disp('✓ MATLAB tests OK');
   "
   
   # Docker tests
   docker run --rm spincirc:latest /opt/spincirc/healthcheck.sh

**Performance Benchmarks**

.. code-block:: python

   # Run performance benchmark
   from python.benchmarks import run_benchmark
   results = run_benchmark()
   print(f"Benchmark score: {results.score}")

Troubleshooting
---------------

**Common Issues**

1. **MATLAB Path Issues**

   .. code-block:: matlab

      % Check current path
      path
      
      % Reset and re-add
      restoredefaultpath;
      addpath(genpath('matlab'));
      savepath;

2. **Python Import Errors**

   .. code-block:: bash

      # Check Python path
      python -c "import sys; print('\n'.join(sys.path))"
      
      # Add SpinCirc to PYTHONPATH
      export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

3. **Docker Build Failures**

   .. code-block:: bash

      # Clean Docker cache
      docker system prune -a
      
      # Build with verbose output
      docker build --progress=plain -t spincirc:latest .

4. **Permission Issues (Linux)**

   .. code-block:: bash

      # Fix file permissions
      chmod -R 755 SpinCirc/
      
      # Fix Python package permissions
      pip install --user -r python/requirements.txt

**Performance Issues**

1. **Slow MATLAB Startup**

   .. code-block:: matlab

      % Disable unnecessary toolboxes
      matlab -nojvm -nodisplay -nosplash

2. **Python Memory Issues**

   .. code-block:: bash

      # Increase virtual memory (Linux)
      sudo sysctl vm.overcommit_memory=1
      
      # Use memory-efficient numpy
      export OMP_NUM_THREADS=1

**Getting Help**

If you encounter installation issues:

1. Check the **system requirements** and ensure compatibility
2. Review the **troubleshooting section** for common solutions
3. Search existing **GitHub issues**: https://github.com/alaweimm90/SpinCirc/issues
4. Create a new issue with detailed error messages and system information
5. Contact support: meshal@berkeley.edu

Next Steps
----------

After successful installation:

1. **Start with the basics**: :doc:`getting_started`
2. **Explore examples**: :doc:`examples/index`
3. **Learn the physics**: :doc:`physics_background`
4. **Try tutorials**: :doc:`tutorials/index`

.. note::
   **Installation complete!** 🎉 You're now ready to explore the world of computational spintronics with SpinCirc.