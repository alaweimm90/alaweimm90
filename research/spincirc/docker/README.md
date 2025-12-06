# SpinCirc Docker Documentation

This directory contains Docker-related files and documentation for the SpinCirc computational spintronics framework.

## Quick Start

### Build and Run
```bash
# Build the Docker image
docker build -t spincirc:latest .

# Run interactive shell
docker run -it --rm spincirc:latest bash

# Start Jupyter Lab
docker run -it --rm -p 8888:8888 spincirc:latest jupyter

# Run tests
docker run --rm spincirc:latest tests

# Run demo
docker run --rm -v $(pwd)/results:/opt/spincirc/results spincirc:latest demo
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# Start only Jupyter
docker-compose up jupyter

# Interactive development
docker-compose run --rm dev

# Run tests
docker-compose run --rm test-runner

# View logs
docker-compose logs -f spincirc
```

## Container Services

### Main SpinCirc Container
- **Image**: `spincirc:latest`
- **Base**: Ubuntu 22.04
- **Python**: 3.10+ with scientific stack
- **MATLAB Alternative**: GNU Octave
- **SPICE**: NGSpice for Verilog-A simulation

### Available Commands
| Command | Description | Port |
|---------|-------------|------|
| `jupyter` | Start Jupyter Lab server | 8888 |
| `notebook` | Start Jupyter Notebook | 8888 |
| `python` | Interactive Python with SpinCirc | - |
| `octave` | MATLAB-compatible environment | - |
| `tests` | Run comprehensive test suite | - |
| `demo` | Run demonstration examples | - |
| `bash` | Interactive shell | - |
| `help` | Show available commands | - |

## Environment Variables

### SpinCirc Configuration
```bash
SPINCIRC_VERSION=1.0.0          # Framework version
SPINCIRC_ROOT=/opt/spincirc     # Installation directory
PYTHONPATH=/opt/spincirc/python # Python module path
MATLABPATH=/opt/spincirc/matlab # MATLAB/Octave path
```

### Python Configuration
```bash
PYTHONDONTWRITEBYTECODE=1       # Don't create .pyc files
PYTHONUNBUFFERED=1              # Unbuffered stdout/stderr
PIP_NO_CACHE_DIR=1              # Don't cache pip downloads
```

### Jupyter Configuration
```bash
JUPYTER_ENABLE_LAB=yes          # Enable JupyterLab
JUPYTER_PORT=8888               # Jupyter server port
JUPYTER_IP=0.0.0.0              # Bind to all interfaces
```

## Volume Mounts

### Recommended Mounts
```bash
# Data directory (input files, datasets)
-v $(pwd)/data:/opt/spincirc/data

# Results directory (simulation outputs)
-v $(pwd)/results:/opt/spincirc/results

# Logs directory (application logs)
-v $(pwd)/logs:/opt/spincirc/logs

# Configuration files
-v $(pwd)/config:/opt/spincirc/config
```

### Development Mounts
```bash
# Source code (for development)
-v $(pwd)/matlab:/opt/spincirc/matlab
-v $(pwd)/python:/opt/spincirc/python
-v $(pwd)/verilogA:/opt/spincirc/verilogA
```

## Examples

### Basic Usage
```bash
# Run interactive Python session
docker run -it --rm spincirc:latest python

# Run specific Python script
docker run --rm -v $(pwd):/workspace spincirc:latest python /workspace/my_script.py

# Run MATLAB/Octave script
docker run --rm -v $(pwd):/workspace spincirc:latest octave /workspace/my_script.m
```

### Jupyter Development
```bash
# Start Jupyter with persistent storage
docker run -d --name spincirc-jupyter \
  -p 8888:8888 \
  -v $(pwd)/notebooks:/opt/spincirc/notebooks \
  -v $(pwd)/data:/opt/spincirc/data \
  -v $(pwd)/results:/opt/spincirc/results \
  spincirc:latest jupyter

# Access at http://localhost:8888
```

### Batch Processing
```bash
# Run simulation batch
docker run --rm \
  -v $(pwd)/input:/opt/spincirc/data \
  -v $(pwd)/output:/opt/spincirc/results \
  spincirc:latest python -c "
import sys
sys.path.append('/opt/spincirc/python')
# Your batch processing code here
"
```

## Advanced Configuration

### Custom Configuration
```bash
# Create custom config
mkdir -p ./config
cat > ./config/custom.conf << EOF
[simulation]
tolerance = 1e-12
max_iterations = 50000
EOF

# Run with custom config
docker run --rm \
  -v $(pwd)/config:/opt/spincirc/config \
  spincirc:latest python -c "
# Load custom configuration
import configparser
config = configparser.ConfigParser()
config.read('/opt/spincirc/config/custom.conf')
tolerance = float(config['simulation']['tolerance'])
print(f'Using tolerance: {tolerance}')
"
```

### Multi-Stage Development
```bash
# Development with live code reloading
docker run -it --rm \
  -v $(pwd)/python:/opt/spincirc/python:rw \
  -v $(pwd)/data:/opt/spincirc/data:rw \
  -v $(pwd)/results:/opt/spincirc/results:rw \
  -e PYTHONPATH=/opt/spincirc/python \
  -e SPINCIRC_DEV_MODE=1 \
  spincirc:latest python
```

## Troubleshooting

### Common Issues

#### Permission Issues
```bash
# Fix ownership of mounted volumes
docker run --rm -v $(pwd)/data:/data alpine chown -R 1000:1000 /data
```

#### Memory Issues
```bash
# Increase container memory limit
docker run --memory=4g --memory-swap=4g spincirc:latest
```

#### Port Conflicts
```bash
# Use different port for Jupyter
docker run -p 8889:8888 spincirc:latest jupyter
```

### Health Checks
```bash
# Check container health
docker inspect spincirc-jupyter --format='{{.State.Health.Status}}'

# View health check logs
docker inspect spincirc-jupyter --format='{{range .State.Health.Log}}{{.Output}}{{end}}'
```

### Performance Optimization
```bash
# Use tmpfs for temporary files
docker run --tmpfs /opt/spincirc/temp:rw,size=1G spincirc:latest

# Enable CPU optimizations
docker run --cpus=4 --memory=8g spincirc:latest

# Use specific CPU architectures
docker build --platform=linux/amd64 -t spincirc:amd64 .
```

## Production Deployment

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spincirc-jupyter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spincirc-jupyter
  template:
    metadata:
      labels:
        app: spincirc-jupyter
    spec:
      containers:
      - name: spincirc
        image: spincirc:latest
        command: ["jupyter"]
        ports:
        - containerPort: 8888
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml spincirc-stack
```

## Security Considerations

### Non-Root User
The container runs as a non-root user (`spincirc`) for security.

### Network Security
```bash
# Create isolated network
docker network create --driver bridge spincirc-network

# Run with custom network
docker run --network=spincirc-network spincirc:latest
```

### Secrets Management
```bash
# Use Docker secrets for sensitive data
echo "secret_value" | docker secret create spincirc_secret -

# Mount secret in container
docker run --secret spincirc_secret spincirc:latest
```

## Support

For issues related to Docker deployment:
1. Check container logs: `docker logs <container-name>`
2. Verify health checks: `docker inspect <container-name>`
3. Test with minimal configuration
4. Review Docker and Docker Compose versions
5. Check system resources (memory, disk space)

For SpinCirc-specific issues, refer to the main documentation.