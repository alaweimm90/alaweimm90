# Docker Container Build & Validation Superprompt

**Context**: Multi-organization repository with 5 priority projects requiring containerization. Some builds failed due to outdated package-lock.json files. Need comprehensive orchestration to build, test, and validate all containers.

---

## Task: Build & Validate All 5 Docker Containers

You are an expert DevOps engineer. Your task is to build, test, and validate 5 Docker containers for production deployment. Follow this checklist systematically.

### Projects to Containerize

1. **SimCore** - AlaweinOS (Vite + React + TypeScript)
   - Path: `.config/organizations/AlaweinOS/SimCore`
   - Port: 3000
   - Build command: `npm run build`

2. **repz** - alaweimm90-business (Vite + React + TypeScript)
   - Path: `.config/organizations/alaweimm90-business/repz`
   - Port: 8080
   - Build command: `npm run build:production`

3. **benchbarrier** - alaweimm90-business (Vite + React + TypeScript)
   - Path: `.config/organizations/alaweimm90-business/benchbarrier`
   - Port: 8081
   - Build command: `npm run build:production`
   - **Status**: ✅ Already built successfully

4. **mag-logic** - alaweimm90-science (Python Scientific Computing)
   - Path: `.config/organizations/alaweimm90-science/mag-logic`
   - Port: 8888
   - Build: Python package with requirements.txt

5. **Attributa** - AlaweinOS (Vite + React + TypeScript, NLP Platform)
   - Path: `.config/organizations/AlaweinOS/Attributa`
   - Port: 3000 (map to 3001 locally to avoid conflict)
   - Build command: `npm run build`

---

## Execution Checklist

### Phase 1: Pre-Build Validation

For EACH project, verify:

- [ ] **Dockerfile exists** and uses correct base image
- [ ] **.dockerignore exists** and excludes node_modules, dist, .git
- [ ] **package.json exists** (Node projects) or **requirements.txt** (Python)
- [ ] **Build script is defined** in package.json
- [ ] **Port is not already in use** locally

**Commands to run:**
```bash
# Check if Dockerfiles exist
ls .config/organizations/AlaweinOS/SimCore/Dockerfile
ls .config/organizations/alaweimm90-business/repz/Dockerfile
ls .config/organizations/alaweimm90-business/benchbarrier/Dockerfile
ls .config/organizations/alaweimm90-science/mag-logic/Dockerfile
ls .config/organizations/AlaweinOS/Attributa/Dockerfile

# Check ports are free
netstat -ano | findstr :3000
netstat -ano | findstr :8080
netstat -ano | findstr :8081
netstat -ano | findstr :8888
netstat -ano | findstr :3001
```

---

### Phase 2: Build All Containers

Build each container with proper error handling. **Skip benchbarrier** since it's already built.

#### 2.1: SimCore

```bash
cd .config/organizations/AlaweinOS/SimCore

# Build
docker build -t simcore:latest .

# If build fails:
# - Check error logs
# - Verify npm install vs npm ci issue
# - Check if dist folder is being generated
# - Verify vite.config.ts has correct build settings

# Expected output: "Successfully built" and "Successfully tagged simcore:latest"
```

#### 2.2: repz

```bash
cd ../../../alaweimm90-business/repz

# Build
docker build -t repz:latest .

# If build fails:
# - Check if build:production script exists in package.json
# - Verify all dependencies are installable
# - Check for TypeScript compilation errors

# Expected output: "Successfully built" and "Successfully tagged repz:latest"
```

#### 2.3: mag-logic (Python)

```bash
cd ../../../alaweimm90-science/mag-logic

# Build
docker build -t mag-logic:latest .

# If build fails:
# - Check if requirements.txt has compatible versions
# - Verify setup.py is correct
# - Check if pyproject.toml is valid
# - May need to add system dependencies to Dockerfile

# Expected output: "Successfully built" and "Successfully tagged mag-logic:latest"
```

#### 2.4: Attributa

```bash
cd ../../../AlaweinOS/Attributa

# Build
docker build -t attributa:latest .

# If build fails:
# - Same checks as SimCore
# - Verify all Radix UI components are installable
# - Check for bun.lockb vs package-lock.json conflicts

# Expected output: "Successfully built" and "Successfully tagged attributa:latest"
```

---

### Phase 3: Verify All Images Built

```bash
# List all images
docker images | grep -E "simcore|repz|benchbarrier|mag-logic|attributa"

# Should see 5 images:
# simcore          latest    <id>    <time>    <size>
# repz             latest    <id>    <time>    <size>
# benchbarrier     latest    <id>    <time>    <size>
# mag-logic        latest    <id>    <time>    <size>
# attributa        latest    <id>    <time>    <size>
```

**Validation:**
- [ ] All 5 images exist
- [ ] All images have "latest" tag
- [ ] Image sizes are reasonable (< 500MB for Node, < 1GB for Python)

---

### Phase 4: Run & Test Each Container

#### 4.1: Test SimCore

```bash
# Run
docker run -d -p 3000:3000 --name simcore simcore:latest

# Wait 5 seconds
sleep 5

# Test health
curl http://localhost:3000

# Check logs
docker logs simcore --tail 20

# Expected: HTTP 200 response, logs show "Accepting connections"

# If fails:
# - Check if serve is installed correctly
# - Verify dist folder was copied
# - Check health check command syntax
```

#### 4.2: Test repz

```bash
# Run
docker run -d -p 8080:8080 --name repz repz:latest

# Wait 5 seconds
sleep 5

# Test health
curl http://localhost:8080

# Check logs
docker logs repz --tail 20

# Expected: HTTP 200 response, logs show "Accepting connections"
```

#### 4.3: Test benchbarrier

```bash
# Run
docker run -d -p 8081:8081 --name benchbarrier benchbarrier:latest

# Wait 5 seconds
sleep 5

# Test health
curl http://localhost:8081

# Check logs
docker logs benchbarrier --tail 20

# Expected: HTTP 200 response
```

#### 4.4: Test mag-logic

```bash
# Run
docker run -d -p 8888:8888 --name mag-logic mag-logic:latest

# Wait 10 seconds (Python takes longer)
sleep 10

# Test health
docker exec mag-logic python -c "import sys; print('OK')"

# Check logs
docker logs mag-logic --tail 20

# Expected: "OK" printed, no import errors

# If fails:
# - Check if all Python packages installed
# - Verify setup.py ran successfully
# - Check for missing system dependencies
```

#### 4.5: Test Attributa

```bash
# Run (map to 3001 to avoid conflict with SimCore)
docker run -d -p 3001:3000 --name attributa attributa:latest

# Wait 5 seconds
sleep 5

# Test health
curl http://localhost:3001

# Check logs
docker logs attributa --tail 20

# Expected: HTTP 200 response
```

---

### Phase 5: Browser Testing

Open browser and manually verify each application loads:

- [ ] **SimCore**: http://localhost:3000 - Should show React app
- [ ] **repz**: http://localhost:8080 - Should show REPZ platform
- [ ] **benchbarrier**: http://localhost:8081 - Should show benchmarking UI
- [ ] **Attributa**: http://localhost:3001 - Should show NLP platform

**For mag-logic** (no UI):
```bash
# Test Python module
docker exec mag-logic python -c "import mag_logic; print('Module loaded')"
```

---

### Phase 6: Resource Monitoring

```bash
# Check all containers are running
docker ps

# Check resource usage
docker stats --no-stream

# Verify:
# - All 5 containers in "Up" state
# - CPU usage < 10% at idle
# - Memory usage reasonable (< 200MB for Node, < 500MB for Python)
```

---

### Phase 7: Cleanup (Optional)

If you need to stop and remove containers:

```bash
# Stop all
docker stop simcore repz benchbarrier mag-logic attributa

# Remove all
docker rm simcore repz benchbarrier mag-logic attributa

# Remove images (only if rebuilding)
docker rmi simcore:latest repz:latest benchbarrier:latest mag-logic:latest attributa:latest
```

---

## Known Issues & Fixes

### Issue 1: npm ci fails with lock file mismatch

**Error**: `npm ci can only install packages when your package.json and package-lock.json are in sync`

**Fix**: Changed all Dockerfiles to use `npm install` instead of `npm ci`

**Files fixed:**
- ✅ SimCore Dockerfile (line 14)
- ✅ repz Dockerfile (line 13)
- ✅ Attributa Dockerfile (line 14)

### Issue 2: Python dependencies fail

**Possible causes:**
- System dependencies missing (gcc, g++, make)
- Incompatible Python package versions
- setup.py errors

**Fix**: Check mag-logic Dockerfile has all system dependencies:
```dockerfile
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

### Issue 3: Port conflicts

**Error**: `Bind for 0.0.0.0:3000 failed: port is already allocated`

**Fix**: Either:
1. Stop the conflicting container
2. Map to different port: `-p 3001:3000`

---

## Success Criteria

### Build Success
- [ ] All 5 Docker images built successfully
- [ ] No errors in build logs
- [ ] Image sizes are reasonable

### Runtime Success
- [ ] All 5 containers start without errors
- [ ] Health checks pass for all containers
- [ ] Applications accessible via browser/curl
- [ ] No crashes in first 60 seconds

### Validation Success
- [ ] All ports responding correctly
- [ ] Logs show no errors
- [ ] Resource usage is normal
- [ ] Applications functional (can load UI/run Python)

---

## Output Required

After completing all phases, provide:

### 1. Build Summary Table

| Project | Build Status | Build Time | Image Size | Notes |
|---------|-------------|-----------|-----------|-------|
| SimCore | ✅ Success | 5m 30s | 250MB | - |
| repz | ✅ Success | 6m 15s | 280MB | - |
| benchbarrier | ✅ Success | 5m 50s | 240MB | Already built |
| mag-logic | ❌ Failed | 8m 23s | - | Python dependencies |
| Attributa | ✅ Success | 7m 10s | 290MB | - |

### 2. Runtime Summary

| Container | Status | Port | Health Check | Browser Test |
|-----------|--------|------|--------------|--------------|
| simcore | Running | 3000 | ✅ Pass | ✅ Loads |
| repz | Running | 8080 | ✅ Pass | ✅ Loads |
| benchbarrier | Running | 8081 | ✅ Pass | ✅ Loads |
| mag-logic | Running | 8888 | ✅ Pass | N/A (CLI) |
| attributa | Running | 3001 | ✅ Pass | ✅ Loads |

### 3. Issues Encountered

List any errors and how they were resolved:
- Issue: npm ci lock file mismatch
  - Fix: Changed to npm install
  - Status: ✅ Resolved

### 4. Next Steps

- [ ] Update `.metaHub/projects-registry.json` with `containerized: true` for all 5
- [ ] Recalculate health scores (each project +1 point)
- [ ] Create docker-compose.yml for full stack
- [ ] Commit Dockerfiles to git
- [ ] Push images to registry (optional)

---

## Final Validation Commands

Run these to verify everything:

```bash
# All images exist
docker images | grep -E "simcore|repz|benchbarrier|mag-logic|attributa"

# All containers running
docker ps | grep -E "simcore|repz|benchbarrier|mag-logic|attributa"

# All ports responding
curl -I http://localhost:3000  # SimCore
curl -I http://localhost:8080  # repz
curl -I http://localhost:8081  # benchbarrier
docker exec mag-logic python -c "print('OK')"  # mag-logic
curl -I http://localhost:3001  # Attributa

# All health checks passing
docker ps --filter "health=healthy"

# Resource usage reasonable
docker stats --no-stream
```

---

## Additional Context

- **Working Directory**: `c:\Users\mesha\Desktop\GitHub`
- **Docker Desktop**: Running on Windows
- **Total Projects**: 80 (these are top 5 priority)
- **Target Containerization Rate**: 55% → 61.3%
- **Phase**: Week 2 of 15-week Master Plan
- **LLM to Use**: This is a multi-step orchestration task - perfect for SOLO editor (Trae)

---

## Edge Cases to Handle

1. **Disk space**: Check `docker system df` before building
2. **Network issues**: Downloads may fail, add retry logic
3. **Build cache**: Use `--no-cache` if builds are inconsistent
4. **Concurrent builds**: May cause resource exhaustion on lower-end machines
5. **Python virtual environments**: Ensure container doesn't try to use host venv

---

## Expected Timeline

- **Pre-validation**: 2 minutes
- **Build all 5**: 15-25 minutes (parallel: 8-12 minutes)
- **Testing**: 5 minutes
- **Validation**: 3 minutes

**Total**: 25-35 minutes for complete orchestration

---

**Ready to execute? Start with Phase 1 and work through systematically.**
