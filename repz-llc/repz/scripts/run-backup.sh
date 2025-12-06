#!/bin/bash
# Comprehensive Backup Runner

set -e

BACKUP_TYPE="${1:-full}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🔄 Starting automated backup system..."
echo "Backup type: $BACKUP_TYPE"
echo "Timestamp: $TIMESTAMP"

# Function to log with timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Pre-backup checks
log "🔍 Running pre-backup checks..."

# Check disk space
AVAILABLE_SPACE=$(df . | awk 'NR==2 {print $4}')
REQUIRED_SPACE=1048576  # 1GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
  log "❌ Insufficient disk space for backup"
  exit 1
fi

log "✅ Pre-backup checks passed"

# Run backups based on type
case "$BACKUP_TYPE" in
  "full")
    log "💾 Running full backup..."
    ./scripts/backup-database.sh
    ./scripts/backup-files.sh
    ./scripts/backup-config.sh
    ;;
  "database")
    log "🗄️  Running database backup..."
    ./scripts/backup-database.sh
    ;;
  "files")
    log "📁 Running file backup..."
    ./scripts/backup-files.sh
    ;;
  "config")
    log "⚙️  Running configuration backup..."
    ./scripts/backup-config.sh
    ;;
  *)
    log "❌ Unknown backup type: $BACKUP_TYPE"
    exit 1
    ;;
esac

# Post-backup validation
log "✅ Backup completed, running validation..."
node scripts/monitor-backups.mjs

# Generate backup report
BACKUP_REPORT="backup-report-$TIMESTAMP.json"
cat > "$BACKUP_REPORT" << EOF
{
  "timestamp": "$TIMESTAMP",
  "backupType": "$BACKUP_TYPE",
  "status": "completed",
  "duration": $SECONDS,
  "diskUsage": {
    "available": "$AVAILABLE_SPACE",
    "backupSize": "$(du -sh backups/ | cut -f1)"
  }
}
EOF

log "📊 Backup report generated: $BACKUP_REPORT"
log "🎉 Automated backup system completed successfully"
