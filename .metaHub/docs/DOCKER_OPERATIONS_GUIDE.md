# Docker Operations Guide

Complete guide to managing the multi-organization Docker stack with production-grade operational tools.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Health Monitoring](#health-monitoring)
3. [Backup & Restore](#backup--restore)
4. [Development Workflow](#development-workflow)
5. [Troubleshooting](#troubleshooting)
6. [Production Operations](#production-operations)
7. [Automation Scripts](#automation-scripts)

---

## Quick Reference

### Essential Commands

```bash
# Start services
make up

# Check health
make health
make monitor

# View logs
make logs

# Backup volumes
make backup

# Stop services
make down
```

### All Available Make Commands

```bash
make help                  # Show all available commands
make setup                 # Initial environment setup
make up                    # Start all services
make down                  # Stop all services
make restart               # Restart all services
make logs                  # Show logs (follow mode)
make ps                    # Show running containers
make test                  # Run all tests
make build                 # Rebuild all images
make clean                 # Remove everything (with confirmation)
make validate              # Validate docker-compose config
make health                # Check service health
make stats                 # Show resource usage
make prune                 # Clean unused resources
make monitor               # Health monitoring (single run)
make monitor-continuous    # Continuous health monitoring
make backup                # Backup all volumes
make backup-stop           # Backup with containers stopped
make restore               # Restore from backup (requires TIMESTAMP)
make list-backups          # List available backups
make update-registry       # Update containerization registry
make identify-tier2        # Find next projects to containerize
make status                # Show phase status
make deploy-staging        # Deploy to staging
make deploy-prod           # Deploy to production (with confirmation)
```

---

## Health Monitoring

### Single Health Check

Check current status of all services:

```bash
make monitor
```

**Output**:
```
╔══════════════════════════════════════════════════════════╗
║          DOCKER HEALTH MONITOR REPORT                   ║
╚══════════════════════════════════════════════════════════╝

Timestamp: 2025-11-24 12:00:00

Overall Status: ✅ ALL HEALTHY
Health Score: 5/5 (100%)

✅ Healthy Services (5):
   • simcore - healthy - Up 2 hours
   • repz - healthy - Up 2 hours
   • benchbarrier - healthy - Up 2 hours
   • mag-logic - running (no healthcheck) - Up 2 hours
   • attributa - healthy - Up 2 hours

📊 Resource Usage:
NAME          CPU %   MEM USAGE
simcore       0.5%    245MB / 512MB
repz          0.3%    198MB / 512MB
...
```

### Continuous Monitoring

Monitor health in real-time (30-second intervals):

```bash
make monitor-continuous
```

Press `Ctrl+C` to stop.

### Custom Monitoring

Use the PowerShell script directly for advanced options:

```bash
# Check every 60 seconds
pwsh .metaHub/scripts/docker-health-monitor.ps1 -ContinuousMode -IntervalSeconds 60

# With alerts on unhealthy containers
pwsh .metaHub/scripts/docker-health-monitor.ps1 -ContinuousMode -AlertOnUnhealthy

# Custom log location
pwsh .metaHub/scripts/docker-health-monitor.ps1 -LogPath "logs/health.log"
```

**Features**:
- ✅ Real-time health status
- 📊 Resource usage (CPU/Memory)
- 🚨 Alert on unhealthy containers
- 📝 Automatic log rotation
- 🔍 Recent error logs for failing services

---

## Backup & Restore

### Create Backup

Backup all Docker volumes (while services are running):

```bash
make backup
```

**Output**:
```
╔══════════════════════════════════════════════════════════╗
║          DOCKER VOLUME BACKUP                            ║
╚══════════════════════════════════════════════════════════╝

ℹ️ Backup started at: 2025-11-24 12:00:00
ℹ️ Backup location: .metaHub/backups/volumes/20251124-120000

📦 Backing up volume: simcore-data
✅ Archive created: 145.32 MB
✅ Compressed to: 52.18 MB (saved 64.1%)

📦 Backing up volume: repz-data
✅ Archive created: 98.45 MB
✅ Compressed to: 34.89 MB (saved 64.6%)

...

╔══════════════════════════════════════════════════════════╗
║          BACKUP SUMMARY                                  ║
╚══════════════════════════════════════════════════════════╝

✅ Backup completed at: 2025-11-24 12:05:30
ℹ️ Backup location: .metaHub/backups/volumes/20251124-120000
ℹ️ Total backup size: 187.43 MB

Volume Backups:
  ✅ simcore-data - 52.18 MB - simcore-data.tar.gz
  ✅ repz-data - 34.89 MB - repz-data.tar.gz
  ✅ benchbarrier-data - 28.76 MB - benchbarrier-data.tar.gz
  ✅ mag-logic-data - 45.23 MB - mag-logic-data.tar.gz
  ✅ attributa-data - 26.37 MB - attributa-data.tar.gz

📄 Manifest saved to: .metaHub/backups/volumes/20251124-120000/manifest.json

🧹 Cleaning up old backups...
✅ Kept 7 most recent backups

✅ Backup complete!
```

### Backup with Stopped Containers

For consistency, stop containers during backup:

```bash
make backup-stop
```

Containers will be automatically restarted after backup completes.

### List Backups

```bash
make list-backups
```

**Output**:
```
Available volume backups:
20251124-120000
20251123-180000
20251122-120000
20251121-090000
...
```

### Restore from Backup

Restore all volumes from a specific backup:

```bash
make restore TIMESTAMP=20251124-120000
```

**Interactive Confirmation**:
```
╔══════════════════════════════════════════════════════════╗
║          DOCKER VOLUME RESTORE                           ║
╚══════════════════════════════════════════════════════════╝

ℹ️ Backup date: 2025-11-24 12:00:00
ℹ️ Backup location: .metaHub/backups/volumes/20251124-120000

📦 Available volumes in backup:
  • simcore-data - 52.18 MB
  • repz-data - 34.89 MB
  • benchbarrier-data - 28.76 MB
  • mag-logic-data - 45.23 MB
  • attributa-data - 26.37 MB

⚠️  WARNING: This will OVERWRITE existing volume data!

Volumes to be restored:
  • simcore-data
  • repz-data
  • benchbarrier-data
  • mag-logic-data
  • attributa-data

Type 'yes' to continue or 'no' to cancel:
```

### Restore Specific Volumes

Use the PowerShell script directly:

```bash
# Restore only simcore-data and repz-data
pwsh .metaHub/scripts/restore-volumes.ps1 -BackupTimestamp "20251124-120000" -Volumes @("simcore-data", "repz-data")

# Force restore without confirmation
pwsh .metaHub/scripts/restore-volumes.ps1 -BackupTimestamp "20251124-120000" -Force

# Restore without stopping containers (risky!)
pwsh .metaHub/scripts/restore-volumes.ps1 -BackupTimestamp "20251124-120000" -StopContainers:$false
```

### Backup Best Practices

1. **Schedule Regular Backups**: Use cron/Task Scheduler
   ```bash
   # Daily at 2 AM
   0 2 * * * cd /path/to/repo && make backup
   ```

2. **Before Major Changes**: Always backup
   ```bash
   make backup
   # Make changes
   # If issues: make restore TIMESTAMP=...
   ```

3. **Test Restores**: Verify backups work
   ```bash
   make backup
   make restore TIMESTAMP=latest-timestamp
   make monitor  # Verify services healthy
   ```

4. **Offsite Backups**: Copy to external storage
   ```bash
   # After backup
   rsync -av .metaHub/backups/volumes/ /external/backup/location/
   ```

---

## Development Workflow

### Daily Development

```bash
# Start services in dev mode
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# In another terminal: make changes to code
# Changes auto-reload thanks to volume mounts

# View logs
make logs

# Run tests
make test

# Stop when done
make down
```

### Building After Code Changes

```bash
# Rebuild specific service
docker compose build simcore

# Rebuild all
make build

# Start with new builds
make up
```

### Debugging a Service

```bash
# View logs
docker compose logs -f simcore

# Execute shell in container
docker compose exec simcore sh

# Inside container:
ps aux              # Check processes
env                 # Check environment
curl localhost:3000 # Test endpoint
exit
```

### Testing Changes

```bash
# Run test suite
make test

# Test specific service
docker compose -f docker-compose.yml -f docker-compose.test.yml run simcore npm test
```

---

## Troubleshooting

### Service Won't Start

**Check logs**:
```bash
make logs
# or for specific service
docker compose logs simcore
```

**Common issues**:

1. **Port already in use**:
   ```bash
   # Find process using port
   netstat -ano | findstr :3000

   # Kill process
   taskkill /F /PID <PID>
   ```

2. **Volume permission issues**:
   ```bash
   # Remove and recreate volume
   docker compose down -v
   docker volume rm simcore-data
   make up
   ```

3. **Image build failed**:
   ```bash
   # Rebuild without cache
   docker compose build --no-cache simcore
   ```

### Health Check Failing

**Investigate health check**:
```bash
# View health check logs
docker inspect simcore | grep -A 20 Health

# Test health check command manually
docker compose exec simcore sh
# Inside container, run health check command
wget -O- http://localhost:3000
```

**Fix common health issues**:
- Service takes longer to start → Increase health check interval
- Health endpoint changed → Update Dockerfile HEALTHCHECK
- Network issue → Check service can reach itself

### Container Keeps Restarting

**Check resource limits**:
```bash
# View current usage
make stats

# If hitting limits, edit docker-compose.yml:
services:
  simcore:
    deploy:
      resources:
        limits:
          cpus: '2.0'    # Increase from 1.0
          memory: 1G     # Increase from 512M
```

### Out of Disk Space

**Clean up Docker resources**:
```bash
# Check disk usage
docker system df

# Remove unused resources
make prune

# More aggressive cleanup
docker system prune -a --volumes

# Nuclear option (removes everything!)
make clean
```

### Services Can't Communicate

**Check networks**:
```bash
# List networks
docker network ls

# Inspect network
docker network inspect frontend-network

# Verify services are on same network
docker compose ps
```

**Test connectivity**:
```bash
# From one container to another
docker compose exec simcore sh
ping repz
curl http://repz:8080/health
```

---

## Production Operations

### Pre-Deployment Checklist

```bash
# 1. Validate configuration
make validate

# 2. Run tests
make test

# 3. Build images
make build

# 4. Backup current state
make backup

# 5. Check health monitoring
make monitor

# 6. Review logs for errors
make logs | grep ERROR
```

### Deployment

**Staging**:
```bash
make deploy-staging
```

**Production**:
```bash
make deploy-prod
```

Both commands will:
1. Start services in detached mode
2. Verify health checks
3. Show status

### Zero-Downtime Deployment

```bash
# 1. Start new containers alongside old
docker compose up -d --scale simcore=2

# 2. Wait for new containers to be healthy
make monitor

# 3. Stop old containers
docker compose up -d --scale simcore=1

# 4. Verify
make health
```

### Rollback

If deployment fails:

```bash
# 1. Check backup timestamp
make list-backups

# 2. Stop services
make down

# 3. Restore from backup
make restore TIMESTAMP=20251124-120000

# 4. Restart services
make up

# 5. Verify health
make monitor
```

### Health Monitoring in Production

Set up continuous monitoring:

```bash
# Option 1: Screen/tmux session
screen -S docker-monitor
make monitor-continuous
# Detach with Ctrl+A, D

# Option 2: System service
# Create systemd service or Windows scheduled task
```

### Log Management

```bash
# Rotate logs
docker compose logs --tail=1000 > logs/archive-$(date +%Y%m%d).log

# Configure log limits in docker-compose.yml:
services:
  simcore:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Automation Scripts

### Health Monitor Script

**Location**: `.metaHub/scripts/docker-health-monitor.ps1`

**Features**:
- Real-time health status monitoring
- Resource usage tracking
- Alert on unhealthy containers
- Automatic logging
- Recent error log display

**Usage**:
```powershell
# Single check
.\docker-health-monitor.ps1

# Continuous monitoring
.\docker-health-monitor.ps1 -ContinuousMode -IntervalSeconds 30

# With alerts
.\docker-health-monitor.ps1 -ContinuousMode -AlertOnUnhealthy

# Custom log path
.\docker-health-monitor.ps1 -LogPath "custom/path/health.log"
```

### Backup Script

**Location**: `.metaHub/scripts/backup-volumes.ps1`

**Features**:
- Backs up all Docker volumes
- Compression (gzip)
- Automatic cleanup (keeps last 7)
- Progress reporting
- Manifest generation

**Usage**:
```powershell
# Basic backup (services running)
.\backup-volumes.ps1

# Stop containers during backup
.\backup-volumes.ps1 -StopContainers

# Custom backup location
.\backup-volumes.ps1 -BackupPath "D:\Backups"

# Backup specific volumes
.\backup-volumes.ps1 -Volumes @("simcore-data", "repz-data")

# Without compression
.\backup-volumes.ps1 -Compress:$false
```

### Restore Script

**Location**: `.metaHub/scripts/restore-volumes.ps1`

**Features**:
- Restore all or specific volumes
- Interactive confirmation
- Automatic container stop/start
- Validation checks
- Progress reporting

**Usage**:
```powershell
# Interactive restore (all volumes)
.\restore-volumes.ps1 -BackupTimestamp "20251124-120000"

# Force restore (no confirmation)
.\restore-volumes.ps1 -BackupTimestamp "20251124-120000" -Force

# Restore specific volumes
.\restore-volumes.ps1 -BackupTimestamp "20251124-120000" -Volumes @("simcore-data")

# Don't stop containers (risky!)
.\restore-volumes.ps1 -BackupTimestamp "20251124-120000" -StopContainers:$false
```

### Registry Update Script

**Location**: `.metaHub/scripts/update-containerization-registry.ps1`

**Usage**:
```bash
make update-registry
```

Updates project registry with containerization status and metrics.

### Tier 2 Identification Script

**Location**: `.metaHub/scripts/identify-tier2-projects.ps1`

**Usage**:
```bash
make identify-tier2
```

Identifies next 5 projects for containerization based on complexity and priority.

---

## File Organization

```
.
├── docker-compose.yml              # Production config
├── docker-compose.dev.yml          # Development overrides
├── docker-compose.test.yml         # Test configuration
├── .dockerignore                   # Docker build exclusions
├── .env.example                    # Environment template
├── Makefile                        # Simplified operations
│
├── .metaHub/
│   ├── scripts/
│   │   ├── docker-health-monitor.ps1    # Health monitoring
│   │   ├── backup-volumes.ps1           # Volume backup
│   │   ├── restore-volumes.ps1          # Volume restore
│   │   ├── update-containerization-registry.ps1
│   │   └── identify-tier2-projects.ps1
│   │
│   ├── backups/
│   │   └── volumes/
│   │       ├── 20251124-120000/         # Backup directory
│   │       │   ├── manifest.json
│   │       │   ├── simcore-data.tar.gz
│   │       │   └── ...
│   │       └── ...
│   │
│   ├── logs/
│   │   └── health-monitor.log
│   │
│   └── docs/
│       ├── DOCKER_QUICKSTART.md
│       ├── DOCKER_OPERATIONS_GUIDE.md   # This file
│       └── ...
│
└── .config/organizations/               # Project sources
    ├── AlaweinOS/
    │   ├── SimCore/Dockerfile
    │   └── Attributa/Dockerfile
    ├── alaweimm90-business/
    │   ├── repz/Dockerfile
    │   └── benchbarrier/Dockerfile
    └── alaweimm90-science/
        └── mag-logic/Dockerfile
```

---

## Quick Troubleshooting Matrix

| Issue | Command | Notes |
|-------|---------|-------|
| Service won't start | `make logs` | Check for errors |
| Health check fails | `docker inspect <service>` | View health details |
| Port already used | `netstat -ano \| findstr :3000` | Find conflicting process |
| Out of memory | `make stats` | Check resource usage |
| Disk space full | `docker system df && make prune` | Clean up |
| Network issues | `docker network inspect <network>` | Check connectivity |
| Build fails | `make build` | Rebuild without cache |
| Data lost | `make restore TIMESTAMP=...` | Restore from backup |
| Slow performance | `make stats` | Check CPU/Memory |
| Container restarting | `docker compose ps` | Check restart count |

---

## Next Steps

1. **Test the setup**:
   ```bash
   make setup
   make up
   make monitor
   ```

2. **Create first backup**:
   ```bash
   make backup
   ```

3. **Set up monitoring**:
   ```bash
   make monitor-continuous
   ```

4. **Review documentation**:
   - [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) - Getting started
   - [PHASE2_SUMMARY.md](.metaHub/PHASE2_SUMMARY.md) - Phase 2 overview
   - CI/CD templates in `.metaHub/templates/ci-cd/`

---

**Created**: 2025-11-24
**Phase**: 2 of 15 (Week 2)
**Status**: Production Ready
**Containerization Rate**: 61.3% (49/80 projects)
**Version**: 1.0.0
