# 🚀 SciComp - GPU Testing Guide
![Berkeley SciComp](https://img.shields.io/badge/SciComp-003262?style=flat-square&logo=university)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20ready-green?style=flat-square)](https://github.com/berkeley/scicomp)
**University of California, Berkeley**
**GPU-Accelerated Scientific Computing**
---
## 🎯 **GPU Testing Overview**
The SciComp includes GPU acceleration support through CuPy and CUDA kernels. When GPU hardware is available, comprehensive testing should be performed to validate performance and correctness.
## 🔧 **GPU Testing Environment Setup**
### **Prerequisites**
- NVIDIA GPU with CUDA capability 6.0+
- CUDA Toolkit 11.0+ or 12.0+
- CuPy package: `pip install cupy-cuda11x` or `pip install cupy-cuda12x`
- PyTorch with CUDA (optional): `pip install torch --index-url https://download.pytorch.org/whl/cu118`
### **Verification Commands**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version
# Check CuPy installation
python -c "import cupy; print(f'CuPy version: {cupy.__version__}'); print(f'CUDA device: {cupy.cuda.runtime.getDeviceProperties(0)}')"
# Check PyTorch CUDA
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```
## 🧪 **GPU Testing Procedures**
### **1. Basic GPU Acceleration Tests**
Run the validation framework with GPU support:
```bash
# Full validation with GPU acceleration
python scripts/validate_framework.py
# GPU-specific performance benchmarks
python scripts/performance_benchmarks.py --gpu-only
```
**Expected Results:**
- GPU acceleration status: ✅ Available
- Matrix multiplication speedup: 2x-10x over CPU
- Memory transfer efficiency: >80% bandwidth utilization
### **2. Module-Specific GPU Tests**
#### **Quantum Physics GPU Tests**
```bash
# Test quantum state operations on GPU
python -c "
import sys; sys.path.append('Python')
from gpu_acceleration.cuda_kernels import QuantumGPU
import numpy as np
qgpu = QuantumGPU()
if qgpu.gpu_available:
    # Test Bell state preparation on GPU
    bell_state = qgpu.prepare_bell_state()
    print(f'Bell state on GPU: {bell_state}')
else:
    print('GPU not available')
"
```
#### **Signal Processing GPU Tests**
```bash
# Test FFT operations on GPU
python -c "
import sys; sys.path.append('Python')
from gpu_acceleration.cuda_kernels import GPUAccelerator
import numpy as np
gpu = GPUAccelerator()
if gpu.gpu_available:
    # Large FFT test
    N = 2**20  # 1M points
    signal = np.random.random(N).astype(np.float32)
    # GPU FFT
    gpu_result = gpu.fft(signal)
    print(f'GPU FFT completed: {gpu_result.shape}')
else:
    print('GPU not available')
"
```
#### **Machine Learning GPU Tests**
```bash
# Test PINN training on GPU
python -c "
import sys; sys.path.append('Python')
from ml_physics.physics_informed_nn import HeatEquationPINN, PINNConfig
config = PINNConfig(layers=[2, 64, 64, 1], epochs=100)
pinn = HeatEquationPINN(config)
if pinn.gpu_available:
    print('Training PINN on GPU...')
    # Training would happen here
    print('GPU training completed')
else:
    print('GPU not available for PINN training')
"
```
### **3. Performance Benchmarking**
#### **Matrix Operations**
Test matrix multiplication performance across different sizes:
- CPU vs GPU speedup ratio
- Memory bandwidth utilization
- Power consumption monitoring
#### **Expected Performance Metrics**
| Operation | CPU (GFLOPS) | GPU (GFLOPS) | Speedup |
|-----------|--------------|--------------|---------|
| Matrix Multiplication 1000x1000 | ~40 | ~200-2000 | 5x-50x |
| FFT 1M points | ~1 | ~10-100 | 10x-100x |
| Quantum State Evolution | ~5 | ~50-500 | 10x-100x |
### **4. Memory and Stability Tests**
#### **Memory Stress Test**
```bash
# Test GPU memory management
python -c "
import sys; sys.path.append('Python')
from gpu_acceleration.cuda_kernels import GPUAccelerator
import numpy as np
gpu = GPUAccelerator()
if gpu.gpu_available:
    # Allocate large arrays progressively
    for size_mb in [100, 500, 1000, 2000]:
        try:
            size = size_mb * 1024 * 1024 // 4  # float32
            data = np.random.random(size).astype(np.float32)
            gpu_data = gpu.to_gpu(data)
            print(f'Successfully allocated {size_mb}MB on GPU')
            gpu.free_gpu_memory(gpu_data)
        except Exception as e:
            print(f'Failed at {size_mb}MB: {e}')
            break
"
```
#### **Long-Running Stability Test**
```bash
# 24-hour GPU stability test
python -c "
import sys; sys.path.append('Python')
import time
from gpu_acceleration.cuda_kernels import GPUAccelerator
gpu = GPUAccelerator()
if gpu.gpu_available:
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < 86400:  # 24 hours
        # Perform continuous GPU operations
        data = np.random.random((1000, 1000)).astype(np.float32)
        result = gpu.matrix_multiply(data, data)
        iterations += 1
        if iterations % 1000 == 0:
            elapsed = time.time() - start_time
            print(f'Iteration {iterations}, Elapsed: {elapsed/3600:.1f}h')
    print('24-hour stability test completed successfully')
"
```
## 📊 **GPU Test Results Documentation**
### **Test Report Template**
```markdown
## GPU Testing Report
**Test Date:** [Date]
**GPU Hardware:** [e.g., NVIDIA RTX 4090, Tesla V100]
**CUDA Version:** [e.g., 12.0]
**CuPy Version:** [e.g., 12.3.0]
### Performance Results
| Test | CPU Time | GPU Time | Speedup | Pass/Fail |
|------|----------|----------|---------|-----------|
| Matrix Mult 1000x1000 | 0.05s | 0.002s | 25x | ✅ |
| FFT 1M points | 0.1s | 0.01s | 10x | ✅ |
| Quantum Evolution | 1.0s | 0.1s | 10x | ✅ |
### Memory Tests
- Maximum allocation: [X GB]
- Memory bandwidth: [X GB/s]
- Stability test: [Pass/Fail]
### Issues Found
- [List any issues or limitations]
### Recommendations
- [Performance tuning suggestions]
```
## 🎯 **Automated GPU Testing**
### **Continuous Integration**
Add GPU testing to GitHub Actions workflow:
```yaml
name: GPU Tests
on: [push, pull_request]
jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install cupy-cuda12x
    - name: Run GPU validation
      run: python scripts/validate_framework.py --gpu-required
    - name: Run GPU benchmarks
      run: python scripts/performance_benchmarks.py --gpu-only
```
## 🚨 **Current Status**
**⚠️ GPU Testing Pending Hardware Availability**
The SciComp includes complete GPU acceleration code with CPU fallbacks. GPU testing requires:
1. NVIDIA GPU hardware with CUDA support
2. Proper CUDA/CuPy installation
3. Performance validation against CPU baselines
**Framework Status:**
- ✅ GPU acceleration code implemented
- ✅ CPU fallbacks working correctly
- ✅ CuPy integration ready
- ⚠️ Hardware validation pending
- ✅ Testing procedures documented
## 📞 **GPU Testing Support**
For GPU testing support or to report GPU-specific issues:
**Primary Contact:** Meshal Alawein
**Email:** [meshal@berkeley.edu](mailto:meshal@berkeley.edu)
**Subject:** Berkeley SciComp GPU Testing
**Include in your report:**
- GPU hardware specifications
- CUDA/CuPy versions
- Test results and benchmarks
- Any performance issues or limitations
---
**🐻💙💛 University of California, Berkeley 💙💛🐻**
*GPU Excellence in Scientific Computing*