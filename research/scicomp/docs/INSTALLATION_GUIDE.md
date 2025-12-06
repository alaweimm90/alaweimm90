# SciComp Installation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](../LICENSE)

## Quick Start
```bash
pip install berkeley-scicomp
python -c "import Python.Quantum; print('SciComp ready')"
```

Docker option:
```bash  
docker pull berkeley/scicomp:latest
docker run -p 8888:8888 berkeley/scicomp:latest
```
## Requirements

- Python 3.8+
- Windows 10+/macOS 10.14+/Linux  
- 4 GB RAM, 2 GB storage
### Recommended Requirements
- **Python**: 3.11+ (optimal performance)
- **Operating System**: Latest versions
- **Memory**: 16 GB RAM (for large-scale simulations)
- **Storage**: 10 GB free space
- **GPU**: CUDA-compatible for acceleration
---
## 🛠️ Detailed Installation Methods
### Method 1: PyPI Package (Recommended)
#### Basic Installation
```bash
# Create virtual environment (recommended)
python -m venv berkeley-scicomp-env
source berkeley-scicomp-env/bin/activate  # Linux/macOS
# berkeley-scicomp-env\Scripts\activate   # Windows
# Install base package
pip install berkeley-scicomp
# Test installation
python -c "from Python.Quantum.core.quantum_states import BellStates; print('✅ Quantum module ready')"
```
#### Feature-Specific Installations
```bash
# GPU acceleration support
pip install berkeley-scicomp[gpu]
# Machine learning capabilities
pip install berkeley-scicomp[ml]
# Performance optimization
pip install berkeley-scicomp[performance]
# Enhanced visualization
pip install berkeley-scicomp[visualization]
# Documentation tools
pip install berkeley-scicomp[docs]
# Development tools
pip install berkeley-scicomp[dev]
# Everything included
pip install berkeley-scicomp[all]
```
### Method 2: Docker Installation
#### Quick Start Container
```bash
# Pull the latest image
docker pull berkeley/scicomp:latest
# Run with Jupyter Lab
docker run -p 8888:8888 berkeley/scicomp:latest
# Run with local data mounting
docker run -p 8888:8888 -v $(pwd)/data:/app/data berkeley/scicomp:latest
# Run interactive container
docker run -it berkeley/scicomp:latest /bin/bash
```
#### Custom Container Build
```dockerfile
FROM berkeley/scicomp:latest
# Add your custom dependencies
RUN pip install your-additional-packages
# Copy your code
COPY ./your-code /app/your-code
# Set working directory
WORKDIR /app/your-code
```
### Method 3: Development Installation
#### From Source
```bash
# Clone the repository
git clone https://github.com/berkeley/scicomp.git
cd scicomp
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
# Install in development mode
pip install -e .
# Install development dependencies
pip install -e .[dev]
# Verify installation
python -m pytest tests/ -v
```
#### Development Setup
```bash
# Install pre-commit hooks
pre-commit install
# Run code formatting
black Python/ examples/ tests/
isort Python/ examples/ tests/
# Run linting
flake8 Python/ examples/ tests/
# Run type checking
mypy Python/
# Run all tests
pytest tests/ -v --cov=Python
```
---
## 🔧 Platform-Specific Instructions
### Windows
#### Prerequisites
```powershell
# Install Python (if not already installed)
# Download from: https://python.org/downloads/
# Install Microsoft Visual C++ Build Tools (for some dependencies)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Verify Python installation
python --version
pip --version
```
#### Installation
```powershell
# Create virtual environment
python -m venv berkeley-scicomp-env
berkeley-scicomp-env\Scripts\activate
# Install Berkeley SciComp
pip install berkeley-scicomp[all]
# Test installation
python -c "import Python.Quantum; print('Windows installation successful!')"
```
#### Windows-Specific Features
```powershell
# Enable Windows Terminal for better output
# Install Windows Terminal from Microsoft Store
# For GPU acceleration on Windows
pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
```
### macOS
#### Prerequisites
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Install Python (if not using system Python)
brew install python@3.11
# Install required system dependencies
brew install gcc gfortran openblas lapack
```
#### Installation
```bash
# Create virtual environment
python3 -m venv berkeley-scicomp-env
source berkeley-scicomp-env/bin/activate
# Install Berkeley SciComp
pip install berkeley-scicomp[all]
# For macOS-specific optimizations
pip install accelerate  # Apple Silicon optimization
```
#### Apple Silicon (M1/M2) Specific
```bash
# For Apple Silicon Macs, use miniforge for better compatibility
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
# Create conda environment
conda create -n berkeley-scicomp python=3.11
conda activate berkeley-scicomp
# Install with conda-forge packages for better ARM64 support
conda install -c conda-forge numpy scipy matplotlib
pip install berkeley-scicomp
```
### Linux (Ubuntu/Debian)
#### Prerequisites
```bash
# Update package manager
sudo apt update && sudo apt upgrade -y
# Install Python development headers and build tools
sudo apt install python3-dev python3-pip python3-venv
sudo apt install build-essential gfortran
sudo apt install libopenblas-dev liblapack-dev
# For GPU support
sudo apt install nvidia-cuda-toolkit  # if using NVIDIA GPU
```
#### Installation
```bash
# Create virtual environment
python3 -m venv berkeley-scicomp-env
source berkeley-scicomp-env/bin/activate
# Upgrade pip
pip install --upgrade pip setuptools wheel
# Install Berkeley SciComp
pip install berkeley-scicomp[all]
# Verify installation
python -c "import Python; print('Linux installation successful!')"
```
### Linux (CentOS/RHEL/Fedora)
```bash
# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel python3-pip
sudo yum install openblas-devel lapack-devel
# Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel python3-pip
sudo dnf install openblas-devel lapack-devel
# Continue with standard installation
pip install berkeley-scicomp[all]
```
---
## ⚡ GPU Acceleration Setup
### NVIDIA GPU (CUDA)
#### Check CUDA Compatibility
```bash
# Check if NVIDIA GPU is available
nvidia-smi
# Check CUDA version
nvcc --version
# Check Python CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```
#### Install CUDA Support
```bash
# For CUDA 11.x
pip install cupy-cuda11x
# For CUDA 12.x
pip install cupy-cuda12x
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
# Verify GPU setup
python -c "from Python.gpu_acceleration.cuda_kernels import GPUAccelerator; gpu = GPUAccelerator(); print(f'GPU Available: {gpu.gpu_available}')"
```
### AMD GPU (ROCm)
```bash
# Install ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
# Note: Limited support compared to CUDA
```
### Apple Silicon GPU
```bash
# Install Metal Performance Shaders backend
pip install torch torchvision torchaudio
# TensorFlow Metal support
pip install tensorflow-metal
# Verify Metal GPU
python -c "import torch; print(torch.backends.mps.is_available())"
```
---
## 📦 Conda Installation Alternative
### Using Conda/Mamba
```bash
# Create conda environment
conda create -n berkeley-scicomp python=3.11
conda activate berkeley-scicomp
# Install scientific computing stack via conda-forge
conda install -c conda-forge numpy scipy matplotlib sympy h5py
# Install Berkeley SciComp via pip (no conda package yet)
pip install berkeley-scicomp
# For GPU support through conda
conda install -c conda-forge cupy  # CUDA support
conda install -c conda-forge pytorch  # PyTorch with GPU
```
### Environment File
Create `environment.yml`:
```yaml
name: berkeley-scicomp
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - numpy
  - scipy
  - matplotlib
  - sympy
  - h5py
  - jupyter
  - pytest
  - pip
  - pip:
    - berkeley-scicomp[all]
```
Install:
```bash
conda env create -f environment.yml
conda activate berkeley-scicomp
```
---
## 🧪 Verify Installation
### Basic Verification
```python
#!/usr/bin/env python3
"""Verification script for Berkeley SciComp installation."""
def test_installation():
    print("🔬 Berkeley SciComp Installation Verification")
    print("=" * 50)
    # Test core modules
    try:
        from Python.Quantum.core.quantum_states import BellStates
        print("✅ Quantum mechanics module")
    except ImportError as e:
        print(f"❌ Quantum module: {e}")
    try:
        from Python.Signal_Processing.core.fourier_transforms import FFT
        print("✅ Signal processing module")
    except ImportError as e:
        print(f"❌ Signal processing: {e}")
    try:
        from Python.Optimization.unconstrained import BFGS
        print("✅ Optimization module")
    except ImportError as e:
        print(f"❌ Optimization module: {e}")
    try:
        from Python.Thermal_Transport.core.heat_equation import HeatEquationSolver1D
        print("✅ Thermal transport module")
    except ImportError as e:
        print(f"❌ Thermal transport: {e}")
    # Test optional modules
    try:
        from Python.gpu_acceleration.cuda_kernels import GPUAccelerator
        gpu = GPUAccelerator()
        print(f"✅ GPU acceleration (Available: {gpu.gpu_available})")
    except ImportError:
        print("⚠️  GPU acceleration not available (optional)")
    try:
        from Python.ml_physics.physics_informed_nn import PINNConfig
        print("✅ ML physics module")
    except ImportError:
        print("⚠️  ML physics not available (optional)")
    print("\n🎉 Verification completed!")
    print("🐻 Go Bears! Berkeley SciComp is ready!")
if __name__ == "__main__":
    test_installation()
```
Save as `verify_installation.py` and run:
```bash
python verify_installation.py
```
### Performance Benchmark
```python
import numpy as np
import time
from Python.Quantum.core.quantum_states import BellStates, EntanglementMeasures
def benchmark():
    print("⚡ Performance Benchmark")
    print("=" * 30)
    # Quantum operations benchmark
    start = time.time()
    for _ in range(1000):
        bell_state = BellStates.phi_plus()
        concurrence = EntanglementMeasures.concurrence(bell_state)
    end = time.time()
    print(f"Quantum operations: {end - start:.4f} seconds (1000 iterations)")
    # Matrix operations benchmark
    start = time.time()
    A = np.random.random((1000, 1000))
    B = np.random.random((1000, 1000))
    C = A @ B
    end = time.time()
    print(f"Matrix multiplication (1000x1000): {end - start:.4f} seconds")
    print(f"Performance: {2 * 1000**3 / (end - start) / 1e9:.2f} GFLOPS")
if __name__ == "__main__":
    benchmark()
```
---
## 🔍 Troubleshooting
### Common Issues and Solutions
#### Import Errors
```python
# Problem: ModuleNotFoundError: No module named 'Python'
# Solution: Check if package is installed in correct environment
import sys
print(sys.path)  # Verify Python path
pip list | grep berkeley-scicomp  # Check if installed
# Reinstall if necessary
pip uninstall berkeley-scicomp
pip install berkeley-scicomp[all]
```
#### Installation Failures
```bash
# Problem: Building wheel for package failed
# Solution: Upgrade build tools
pip install --upgrade pip setuptools wheel
# Install with no cache to force rebuild
pip install --no-cache-dir berkeley-scicomp[all]
# On Windows, install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```
#### GPU Issues
```python
# Check CUDA installation
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
# Check CuPy installation
try:
    import cupy
    print(f"CuPy available: True")
    print(f"CuPy version: {cupy.__version__}")
except ImportError:
    print("CuPy not available")
```
#### Memory Issues
```python
# For large computations, monitor memory usage
import psutil
import gc
print(f"Available memory: {psutil.virtual_memory().available / 1e9:.2f} GB")
# Force garbage collection
gc.collect()
# Use memory-efficient operations
# Instead of: large_array = np.zeros((10000, 10000))
# Use chunked operations or memory mapping
```
### Platform-Specific Issues
#### Windows
- **Issue**: Microsoft Visual C++ 14.0 is required
- **Solution**: Install Microsoft Visual C++ Build Tools
#### macOS
- **Issue**: Compilation errors with Apple Silicon
- **Solution**: Use conda-forge packages or miniforge
#### Linux
- **Issue**: Permission denied errors
- **Solution**: Ensure proper permissions or use virtual environment
### Getting Help
If you encounter issues not covered here:
1. **Check Documentation**: https://berkeley-scicomp.readthedocs.io
2. **GitHub Issues**: https://github.com/berkeley/scicomp/issues
3. **Discussions**: https://github.com/berkeley/scicomp/discussions
4. **Email Support**: scicomp@berkeley.edu
---
## 🚀 Next Steps
After successful installation:
1. **Run Examples**:
   ```bash
   cd examples/beginner/
   python getting_started.py
   ```
2. **Explore Jupyter Notebooks**:
   ```bash
   jupyter lab notebooks/
   ```
3. **Read Documentation**:
   - [Quick Start Guide](QUICK_START.md)
   - [API Reference](API_REFERENCE.md)
   - [Tutorials](../examples/)
4. **Join the Community**:
   - GitHub Discussions
   - Berkeley SciComp Mailing List
   - Contribute to the project
---
## 📄 License
MIT License - see [LICENSE](../LICENSE) file for details.
---
## 🏛️ About UC Berkeley
The SciComp is developed at the **University of California, Berkeley**.
**Contact Information**:
- **Email**: scicomp@berkeley.edu
- **Principal Architect**: Meshal Alawein (meshal@berkeley.edu)
- **Institution**: University of California, Berkeley
- **GitHub**: https://github.com/berkeley/scicomp
---
**🐻💙💛 Go Bears! Fiat Lux - Let There Be Light 💙💛🐻**
*University of California, Berkeley - Scientific Computing Excellence Since 1868*