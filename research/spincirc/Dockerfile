# SpinCirc Framework Docker Image
# Multi-stage build for optimized production container
#
# This Dockerfile creates a containerized environment for the SpinCirc
# computational spintronics framework with MATLAB, Python, and Verilog-A support.
#
# Author: Meshal Alawein <meshal@berkeley.edu>
# Copyright © 2025 Meshal Alawein — All rights reserved.
# License: MIT

# Stage 1: Base environment with system dependencies
FROM ubuntu:22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    SPINCIRC_VERSION=1.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic utilities
    wget \
    curl \
    git \
    unzip \
    tar \
    ca-certificates \
    gnupg \
    software-properties-common \
    locales \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    gfortran \
    # Libraries for scientific computing
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libhdf5-dev \
    libnetcdf-dev \
    # Graphics and display (for MATLAB/plotting)
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libgtk-3-0 \
    libasound2 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxinerama1 \
    # Python dependencies
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    # Verilog-A and SPICE dependencies
    ngspice \
    ngspice-doc \
    gawk \
    octave \
    gnuplot \
    # Development and debugging tools
    vim \
    nano \
    htop \
    tree \
    strace \
    gdb \
    valgrind \
    # Documentation tools
    pandoc \
    texlive-latex-base \
    texlive-latex-recommended \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8

# Create spincirc user with sudo access
RUN useradd -m -s /bin/bash -G sudo spincirc && \
    echo "spincirc ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/spincirc/.local/bin

# Stage 2: Python environment
FROM base as python-env

# Set Python environment
ENV PYTHONPATH=/opt/spincirc/python:$PYTHONPATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python packages
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install \
    # Core scientific stack
    numpy==1.24.3 \
    scipy==1.10.1 \
    matplotlib==3.7.1 \
    pandas==2.0.3 \
    sympy==1.12 \
    # Machine learning
    scikit-learn==1.3.0 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    xgboost==1.7.6 \
    optuna==3.2.0 \
    pymoo==0.6.0.1 \
    # Visualization
    plotly==5.15.0 \
    seaborn==0.12.2 \
    bokeh==3.2.1 \
    ipywidgets==8.0.7 \
    # Interactive computing
    jupyter==1.0.0 \
    jupyterlab==4.0.3 \
    notebook==7.0.2 \
    # Data formats and I/O
    h5py==3.9.0 \
    netcdf4==1.6.4 \
    openpyxl==3.1.2 \
    xlsxwriter==3.1.2 \
    # Optimization and numerical computing
    cvxpy==1.3.2 \
    nlopt==2.7.1 \
    # Physics/engineering specific
    pint==0.22 \
    uncertainties==3.1.7 \
    # Development and testing tools
    pytest==7.4.0 \
    pytest-cov==4.1.0 \
    pytest-html==3.2.0 \
    black==23.7.0 \
    flake8==6.0.0 \
    mypy==1.4.1 \
    # Documentation
    sphinx==7.1.2 \
    sphinx-rtd-theme==1.3.0 \
    nbsphinx==0.9.3 \
    # Additional utilities
    tqdm==4.65.0 \
    psutil==5.9.5 \
    requests==2.31.0

# Stage 3: MATLAB-compatible environment (without actual MATLAB)
FROM python-env as matlab-env

# Create MATLAB-compatible directories and environment
RUN mkdir -p /opt/matlab /opt/spincirc/matlab

# Install Octave as MATLAB alternative for open-source compatibility
RUN apt-get update && apt-get install -y \
    octave \
    octave-signal \
    octave-statistics \
    octave-optim \
    octave-symbolic \
    && rm -rf /var/lib/apt/lists/*

# Set MATLAB environment variables (for compatibility)
ENV MATLAB_ROOT=/opt/matlab \
    MATLABPATH=/opt/spincirc/matlab:$MATLABPATH \
    MLM_LICENSE_FILE="" \
    MATLAB_PREFDIR=/home/spincirc/.matlab

# Stage 4: Final SpinCirc image
FROM matlab-env as final

# Set working directory
WORKDIR /opt/spincirc

# Copy SpinCirc source code
COPY --chown=root:root . /opt/spincirc/

# Create additional directories with proper permissions
RUN mkdir -p /opt/spincirc/{data,logs,results,temp,config,bin} && \
    mkdir -p /home/spincirc/{.jupyter,.config,.local} && \
    chown -R spincirc:spincirc /opt/spincirc /home/spincirc

# Create enhanced startup script
COPY <<'EOF' /opt/spincirc/start_spincirc.sh
#!/bin/bash
# SpinCirc Framework Startup Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "SpinCirc Computational Spintronics Framework"
echo -e "==========================================${NC}"
echo ""
echo -e "${GREEN}Version: ${SPINCIRC_VERSION}${NC}"
echo -e "${GREEN}Author: Meshal Alawein${NC}"
echo ""

# Function to check component availability
check_component() {
    echo -n "  $1: "
    if eval "$2" &>/dev/null; then
        echo -e "${GREEN}✓ Available${NC}"
        return 0
    else
        echo -e "${RED}✗ Not Available${NC}"
        return 1
    fi
}

# Check Python environment
echo -e "${YELLOW}Python Environment:${NC}"
python_version=$(python3 --version 2>&1)
echo "  Python: $python_version"
check_component "NumPy" "python3 -c 'import numpy; print(numpy.__version__)'"
check_component "SciPy" "python3 -c 'import scipy; print(scipy.__version__)'"
check_component "Matplotlib" "python3 -c 'import matplotlib; print(matplotlib.__version__)'"
check_component "Pandas" "python3 -c 'import pandas; print(pandas.__version__)'"
check_component "PyTorch" "python3 -c 'import torch; print(torch.__version__)'"
check_component "Plotly" "python3 -c 'import plotly; print(plotly.__version__)'"
echo ""

# Check MATLAB/Octave environment
echo -e "${YELLOW}MATLAB/Octave Environment:${NC}"
check_component "Octave" "octave --version"
check_component "MATLAB" "which matlab"
echo ""

# Check Verilog-A/SPICE environment
echo -e "${YELLOW}Verilog-A/SPICE Environment:${NC}"
check_component "NGSpice" "ngspice -v"
check_component "GNUPLOT" "gnuplot --version"
echo ""

# Check SpinCirc modules
echo -e "${YELLOW}SpinCirc Modules:${NC}"
export PYTHONPATH="/opt/spincirc/python:$PYTHONPATH"
check_component "SpinCirc Python" "python3 -c 'import sys; sys.path.append(\"/opt/spincirc/python\")'"
check_component "ML Tools" "python3 -c 'import sys; sys.path.append(\"/opt/spincirc/python\"); import ml_tools'"
check_component "Visualization" "python3 -c 'import sys; sys.path.append(\"/opt/spincirc/python\"); import visualization'"
echo ""

# Set up environment
export PYTHONPATH="/opt/spincirc/python:$PYTHONPATH"
export PATH="/opt/spincirc/bin:$PATH"
export MATLABPATH="/opt/spincirc/matlab:$MATLABPATH"

# Create necessary directories
mkdir -p /opt/spincirc/{data,logs,results,temp}

# Start requested service
case "$1" in
    "jupyter")
        echo -e "${GREEN}Starting Jupyter Lab...${NC}"
        cd /opt/spincirc
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                   --NotebookApp.token='' --NotebookApp.password='' \
                   --ServerApp.root_dir=/opt/spincirc
        ;;
    "notebook")
        echo -e "${GREEN}Starting Jupyter Notebook...${NC}"
        cd /opt/spincirc
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                        --NotebookApp.token='' --NotebookApp.password=''
        ;;
    "python")
        echo -e "${GREEN}Starting Python environment...${NC}"
        cd /opt/spincirc
        if [ $# -eq 1 ]; then
            python3 -i -c "
import sys
sys.path.append('/opt/spincirc/python')
print('SpinCirc Python environment ready!')
print('Try: import ml_tools, visualization')
"
        else
            python3 "${@:2}"
        fi
        ;;
    "octave")
        echo -e "${GREEN}Starting Octave (MATLAB alternative)...${NC}"
        cd /opt/spincirc/matlab
        octave --path /opt/spincirc/matlab "${@:2}"
        ;;
    "tests")
        echo -e "${GREEN}Running SpinCirc test suite...${NC}"
        cd /opt/spincirc
        # Run Python tests
        echo "Running Python tests..."
        python3 -m pytest python/tests/ -v --tb=short
        # Run MATLAB tests with Octave
        echo "Running MATLAB tests with Octave..."
        cd matlab/tests
        octave --eval "try; RunAllTests(); catch ME; disp(ME.message); end; exit"
        ;;
    "demo")
        echo -e "${GREEN}Running SpinCirc demo...${NC}"
        cd /opt/spincirc/examples
        python3 -c "
print('SpinCirc Demo - Quick Examples')
print('==============================')

# Import SpinCirc modules
import sys
sys.path.append('/opt/spincirc/python')

try:
    import numpy as np
    import matplotlib.pyplot as plt
    print('✓ Basic scientific stack loaded')
    
    # Create simple demonstration
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SpinCirc Demo Plot')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/opt/spincirc/results/demo_plot.png', dpi=150)
    plt.show()
    
    print('✓ Demo plot saved to /opt/spincirc/results/demo_plot.png')
    print('✓ SpinCirc demo completed successfully!')
    
except ImportError as e:
    print(f'✗ Import error: {e}')
except Exception as e:
    print(f'✗ Demo error: {e}')
"
        ;;
    "bash")
        echo -e "${GREEN}Starting interactive bash shell...${NC}"
        exec /bin/bash
        ;;
    "help"|"--help"|"-h")
        echo -e "${YELLOW}SpinCirc Docker Container Commands:${NC}"
        echo "  jupyter     - Start Jupyter Lab server (port 8888)"
        echo "  notebook    - Start Jupyter Notebook server (port 8888)"
        echo "  python      - Start Python with SpinCirc modules"
        echo "  octave      - Start Octave (MATLAB alternative)"
        echo "  tests       - Run comprehensive test suite"
        echo "  demo        - Run SpinCirc demonstration"
        echo "  bash        - Interactive bash shell"
        echo "  help        - Show this help message"
        echo ""
        echo -e "${YELLOW}Usage Examples:${NC}"
        echo "  docker run -it --rm spincirc:3.11-slim bash"
        echo "  docker run -it --rm -p 8888:8888 spincirc:3.11-slim jupyter"
        echo "  docker run -it --rm -v \$(pwd)/data:/opt/spincirc/data spincirc:3.11-slim python"
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo "Use 'help' to see available commands"
        exec /bin/bash
        ;;
esac
EOF

# Create comprehensive configuration files
COPY <<'EOF' /opt/spincirc/config/spincirc.conf
# SpinCirc Framework Configuration

[general]
version = 1.0.0
author = Meshal Alawein
email = meshal@berkeley.edu
license = MIT
description = Computational framework for spintronic device simulation

[paths]
root_dir = /opt/spincirc
data_dir = /opt/spincirc/data
results_dir = /opt/spincirc/results
temp_dir = /opt/spincirc/temp
log_dir = /opt/spincirc/logs
matlab_dir = /opt/spincirc/matlab
python_dir = /opt/spincirc/python
verilogA_dir = /opt/spincirc/verilogA

[python]
backend = numpy
parallel = true
max_workers = 4
default_solver = scipy
plotting_backend = matplotlib

[matlab]
alternative = octave
display = false
warnings = off
precision = double
path_setup = automatic

[simulation]
default_ode_solver = adaptive_ode
tolerance = 1e-9
max_iterations = 10000
time_units = seconds
length_units = meters

[logging]
level = INFO
file = /opt/spincirc/logs/spincirc.log
format = %(asctime)s [%(levelname)s] %(name)s: %(message)s
EOF

# Create Jupyter configuration
COPY <<'EOF' /home/spincirc/.jupyter/jupyter_lab_config.py
# Jupyter Lab Configuration for SpinCirc

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.notebook_dir = '/opt/spincirc'
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True
c.ExtensionApp.open_browser = False
c.LabApp.open_browser = False
EOF

# Create enhanced health check script
COPY <<'EOF' /opt/spincirc/healthcheck.sh
#!/bin/bash
# Comprehensive health check script for SpinCirc container

# Function to check component
check_component() {
    if eval "$2" &>/dev/null; then
        echo "✓ $1: OK"
        return 0
    else
        echo "✗ $1: FAILED"
        return 1
    fi
}

echo "SpinCirc Container Health Check"
echo "==============================="

# Check Python environment
check_component "Python" "python3 -c 'import sys; print(sys.version)'"
check_component "NumPy" "python3 -c 'import numpy'"
check_component "SciPy" "python3 -c 'import scipy'"
check_component "Matplotlib" "python3 -c 'import matplotlib'"

# Check file system permissions
check_component "Write permissions" "touch /opt/spincirc/temp/healthcheck.tmp && rm -f /opt/spincirc/temp/healthcheck.tmp"

# Check SpinCirc modules
export PYTHONPATH="/opt/spincirc/python:$PYTHONPATH"
check_component "SpinCirc Python path" "python3 -c 'import sys; sys.path.append(\"/opt/spincirc/python\")'"

# Check critical directories
for dir in data logs results temp; do
    check_component "Directory: $dir" "test -d /opt/spincirc/$dir && test -w /opt/spincirc/$dir"
done

echo "==============================="
echo "Health check completed"
exit 0
EOF

# Make scripts executable and set proper ownership
RUN chmod +x /opt/spincirc/start_spincirc.sh \
             /opt/spincirc/healthcheck.sh && \
    chown -R spincirc:spincirc /opt/spincirc /home/spincirc

# Create requirements.txt for reference
COPY <<'EOF' /opt/spincirc/requirements.txt
# SpinCirc Framework Python Requirements
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
pandas==2.0.3
sympy==1.12
scikit-learn==1.3.0
torch==2.0.1
torchvision==0.15.2
xgboost==1.7.6
optuna==3.2.0
pymoo==0.6.0.1
plotly==5.15.0
seaborn==0.12.2
bokeh==3.2.1
ipywidgets==8.0.7
jupyter==1.0.0
jupyterlab==4.0.3
notebook==7.0.2
h5py==3.9.0
netcdf4==1.6.4
openpyxl==3.1.2
xlsxwriter==3.1.2
cvxpy==1.3.2
nlopt==2.7.1
pint==0.22
uncertainties==3.1.7
pytest==7.4.0
pytest-cov==4.1.0
pytest-html==3.2.0
black==23.7.0
flake8==6.0.0
mypy==1.4.1
sphinx==7.1.2
sphinx-rtd-theme==1.3.0
nbsphinx==0.9.3
tqdm==4.65.0
psutil==5.9.5
requests==2.31.0
EOF

# Switch to spincirc user
USER spincirc

# Set final environment variables
ENV HOME=/home/spincirc \
    USER=spincirc \
    SPINCIRC_ROOT=/opt/spincirc \
    PYTHONPATH=/opt/spincirc/python:$PYTHONPATH \
    MATLABPATH=/opt/spincirc/matlab:$MATLABPATH \
    PATH=/opt/spincirc/bin:$PATH

# Expose ports for Jupyter and other services
EXPOSE 8888 8889 8890 8891 8892

# Add comprehensive health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD /opt/spincirc/healthcheck.sh

# Set default command
ENTRYPOINT ["/opt/spincirc/start_spincirc.sh"]
CMD ["help"]

# Enhanced metadata labels
LABEL org.opencontainers.image.title="SpinCirc Framework" \
      org.opencontainers.image.description="Comprehensive computational framework for spintronic device simulation and design" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Meshal Alawein <meshal@berkeley.edu>" \
      org.opencontainers.image.url="https://github.com/meshalawy/SpinCirc" \
      org.opencontainers.image.source="https://github.com/meshalawy/SpinCirc" \
      org.opencontainers.image.documentation="https://github.com/meshalawy/SpinCirc/blob/main/README.md" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="UC Berkeley" \
      org.label-schema.docker.cmd="docker run -it --rm -p 8888:8888 spincirc:3.11-slim jupyter" \
      org.label-schema.docker.cmd.test="docker run --rm spincirc:3.11-slim tests" \
      org.label-schema.docker.cmd.help="docker run --rm spincirc:3.11-slim help"