# Containerization Guide - Phase 2

**Status**: Dockerfiles created for Top 5 Priority Projects
**Date**: 2025-11-24
**Phase**: 2 (Week 2-4 Containerization)

---

## Summary

All 5 priority projects have been containerized with production-ready Dockerfiles:

### Projects Containerized

1. **SimCore** (AlaweinOS) - Vite + React + TypeScript
2. **repz** (alaweimm90-business) - Vite + React + TypeScript
3. **benchbarrier** (alaweimm90-business) - Vite + React + TypeScript
4. **mag-logic** (alaweimm90-science) - Python Scientific Computing
5. **Attributa** (AlaweinOS) - Vite + React + TypeScript (NLP Platform)

---

## Testing Instructions

### Prerequisites

1. **Start Docker Desktop**
   ```powershell
   # Ensure Docker Desktop is running
   docker --version
   docker ps
   ```

2. **Navigate to GitHub root**
   ```powershell
   cd C:\Users\mesha\Desktop\GitHub
   ```

---

## Project 1: SimCore

**Location**: `.config/organizations/AlaweinOS/SimCore/`
**Port**: 3000
**Tech**: Vite + React + TypeScript + Supabase

### Build
```bash
cd .config/organizations/AlaweinOS/SimCore
docker build -t simcore:latest .
```

### Run
```bash
docker run -d -p 3000:3000 --name simcore simcore:latest
```

### Test
```bash
# Check if running
docker ps | grep simcore

# View logs
docker logs simcore

# Test in browser
# Open: http://localhost:3000

# Health check
curl http://localhost:3000
```

### Stop & Clean
```bash
docker stop simcore
docker rm simcore
```

---

## Project 2: repz

**Location**: `.config/organizations/alaweimm90-business/repz/`
**Port**: 8080
**Tech**: Vite + React + TypeScript (REPZ Platform)

### Build
```bash
cd .config/organizations/alaweimm90-business/repz
docker build -t repz:latest .
```

### Run
```bash
docker run -d -p 8080:8080 --name repz repz:latest
```

### Test
```bash
# Check if running
docker ps | grep repz

# View logs
docker logs repz

# Test in browser
# Open: http://localhost:8080

# Health check
curl http://localhost:8080
```

### Stop & Clean
```bash
docker stop repz
docker rm repz
```

---

## Project 3: benchbarrier

**Location**: `.config/organizations/alaweimm90-business/benchbarrier/`
**Port**: 8081
**Tech**: Vite + React + TypeScript

### Build
```bash
cd .config/organizations/alaweimm90-business/benchbarrier
docker build -t benchbarrier:latest .
```

### Run
```bash
docker run -d -p 8081:8081 --name benchbarrier benchbarrier:latest
```

### Test
```bash
# Check if running
docker ps | grep benchbarrier

# View logs
docker logs benchbarrier

# Test in browser
# Open: http://localhost:8081

# Health check
curl http://localhost:8081
```

### Stop & Clean
```bash
docker stop benchbarrier
docker rm benchbarrier
```

---

## Project 4: mag-logic

**Location**: `.config/organizations/alaweimm90-science/mag-logic/`
**Port**: 8888
**Tech**: Python 3.11 + Scientific Computing

### Build
```bash
cd .config/organizations/alaweimm90-science/mag-logic
docker build -t mag-logic:latest .
```

### Run
```bash
docker run -d -p 8888:8888 --name mag-logic mag-logic:latest
```

### Test
```bash
# Check if running
docker ps | grep mag-logic

# View logs
docker logs mag-logic

# Health check
docker exec mag-logic python -c "import sys; print('OK')"
```

### Stop & Clean
```bash
docker stop mag-logic
docker rm mag-logic
```

---

## Project 5: Attributa

**Location**: `.config/organizations/AlaweinOS/Attributa/`
**Port**: 3000
**Tech**: Vite + React + TypeScript (NLP Platform)

### Build
```bash
cd .config/organizations/AlaweinOS/Attributa
docker build -t attributa:latest .
```

### Run
```bash
docker run -d -p 3001:3000 --name attributa attributa:latest
```
*Note: Using port 3001 locally to avoid conflict with SimCore*

### Test
```bash
# Check if running
docker ps | grep attributa

# View logs
docker logs attributa

# Test in browser
# Open: http://localhost:3001

# Health check
curl http://localhost:3001
```

### Stop & Clean
```bash
docker stop attributa
docker rm attributa
```

---

## Test All Together (Dev Stack)

Create a `docker-compose.yml` file to run all services together:

```yaml
version: '3.8'

services:
  simcore:
    build: .config/organizations/AlaweinOS/SimCore
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    restart: unless-stopped

  repz:
    build: .config/organizations/alaweimm90-business/repz
    ports:
      - "8080:8080"
    environment:
      - NODE_ENV=production
    restart: unless-stopped

  benchbarrier:
    build: .config/organizations/alaweimm90-business/benchbarrier
    ports:
      - "8081:8081"
    environment:
      - NODE_ENV=production
    restart: unless-stopped

  mag-logic:
    build: .config/organizations/alaweimm90-science/mag-logic
    ports:
      - "8888:8888"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  attributa:
    build: .config/organizations/AlaweinOS/Attributa
    ports:
      - "3001:3000"
    environment:
      - NODE_ENV=production
    restart: unless-stopped
```

### Run Full Stack
```bash
docker-compose up -d
```

### Check All Services
```bash
docker-compose ps
docker-compose logs -f
```

### Stop Full Stack
```bash
docker-compose down
```

---

## Troubleshooting

### Build Fails

**Issue**: `npm ci` fails during build
**Fix**: Ensure `package-lock.json` exists and is up to date
```bash
npm install
```

**Issue**: Python dependencies fail to install
**Fix**: Check `requirements.txt` for any system dependencies needed

### Container Won't Start

**Issue**: Port already in use
**Fix**: Change the external port mapping
```bash
docker run -d -p 3002:3000 --name simcore simcore:latest
```

**Issue**: Container exits immediately
**Fix**: Check logs for errors
```bash
docker logs <container-name>
```

### Health Check Fails

**Issue**: Health check reports unhealthy
**Fix**: Check if the application is actually serving on the expected port
```bash
docker exec <container-name> wget -q -O- http://localhost:3000
```

---

## Next Steps

After testing all containers:

### 1. Update Registry
```powershell
powershell -ExecutionPolicy Bypass -File .metaHub\scripts\update-registry-containerization.ps1
```

### 2. Recalculate Health Scores
```powershell
powershell -ExecutionPolicy Bypass -File .metaHub\scripts\calculate-health-scores.ps1
```

### 3. Verify Metrics
- Expected containerization rate: 55% → 61.3%
- All 5 projects should show `containerized: true`
- Health scores should increase by +1 point each

### 4. Commit Dockerfiles
```bash
git add */Dockerfile */.dockerignore
git commit -m "feat(containers): add Dockerfiles for top 5 priority projects

- SimCore: Vite + React + TypeScript
- repz: REPZ Platform
- benchbarrier: Benchmarking tool
- mag-logic: Python scientific computing
- Attributa: NLP attribution platform

All containers use multi-stage builds for optimized production images.
Health checks included for monitoring.

Phase 2 containerization - Week 2 progress.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Build Time Estimates

Based on Tier 1 classification:

| Project | Estimated Build Time | Actual Time |
|---------|---------------------|-------------|
| SimCore | 5-10 minutes | TBD |
| repz | 5-10 minutes | TBD |
| benchbarrier | 5-10 minutes | TBD |
| mag-logic | 10-15 minutes (Python) | TBD |
| Attributa | 5-10 minutes | TBD |

**Total Estimated**: 30-55 minutes for all 5 builds

---

## Success Criteria

- [ ] All 5 projects build successfully
- [ ] All 5 projects run without errors
- [ ] All 5 projects accessible via their ports
- [ ] Health checks pass for all projects
- [ ] Projects-registry.json updated
- [ ] Health scores recalculated
- [ ] Dockerfiles committed to git

---

## Files Created

### Dockerfiles
1. [.config/organizations/AlaweinOS/SimCore/Dockerfile](.config/organizations/AlaweinOS/SimCore/Dockerfile)
2. [.config/organizations/alaweimm90-business/repz/Dockerfile](.config/organizations/alaweimm90-business/repz/Dockerfile)
3. [.config/organizations/alaweimm90-business/benchbarrier/Dockerfile](.config/organizations/alaweimm90-business/benchbarrier/Dockerfile)
4. [.config/organizations/alaweimm90-science/mag-logic/Dockerfile](.config/organizations/alaweimm90-science/mag-logic/Dockerfile)
5. [.config/organizations/AlaweinOS/Attributa/Dockerfile](.config/organizations/AlaweinOS/Attributa/Dockerfile)

### .dockerignore Files
1. [.config/organizations/AlaweinOS/SimCore/.dockerignore](.config/organizations/AlaweinOS/SimCore/.dockerignore)
2. [.config/organizations/alaweimm90-business/repz/.dockerignore](.config/organizations/alaweimm90-business/repz/.dockerignore)
3. [.config/organizations/alaweimm90-business/benchbarrier/.dockerignore](.config/organizations/alaweimm90-business/benchbarrier/.dockerignore)
4. [.config/organizations/alaweimm90-science/mag-logic/.dockerignore](.config/organizations/alaweimm90-science/mag-logic/.dockerignore)
5. [.config/organizations/AlaweinOS/Attributa/.dockerignore](.config/organizations/AlaweinOS/Attributa/.dockerignore)

---

## Phase 2 Progress

**Week 2 Status**: Dockerfiles Created ✅

**Next Week**: Test builds, deploy to dev environment, create docker-compose stack

**Target Completion**: Week 4
