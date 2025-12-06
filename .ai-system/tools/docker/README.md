# Docker Templates

Base Dockerfiles for projects. Extend instead of duplicate.

## Usage

### Python Project

```dockerfile
# Dockerfile
FROM ../../../tools/docker/Dockerfile.python
# Add project-specific commands
```

Or symlink:

```bash
ln -s ../../../tools/docker/Dockerfile.python Dockerfile
```

### Node Project

```dockerfile
# Dockerfile
FROM ../../../tools/docker/Dockerfile.node
# Add project-specific commands
```

## Templates

- `Dockerfile.python` - Python 3.11 + uvicorn
- `Dockerfile.node` - Node 20 + build step

## Customize

Override CMD or add layers:

```dockerfile
FROM ../../../tools/docker/Dockerfile.python
RUN pip install extra-package
CMD ["python", "custom_main.py"]
```
