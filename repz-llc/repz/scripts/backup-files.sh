#!/bin/bash
# Enterprise File Backup Script

set -e

BACKUP_DIR="$(dirname "$0")/../backups"
FILE_BACKUP_DIR="$BACKUP_DIR/files"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RETENTION_DAYS=30

echo "📁 Starting file backup..."

# Critical directories to backup
DIRS_TO_BACKUP=(
  "src"
  "public"
  "scripts"
  "docs"
  ".github"
  "branding"
)

# Files to backup
FILES_TO_BACKUP=(
  "package.json"
  "package-lock.json"
  "tsconfig.json"
  "vite.config.ts"
  "tailwind.config.js"
  "CLAUDE.md"
  "README.md"
)

BACKUP_ARCHIVE="$FILE_BACKUP_DIR/files_$TIMESTAMP.tar.gz"

echo "📦 Creating file archive..."

# Create tar archive with selected directories and files
tar -czf "$BACKUP_ARCHIVE" \
  --exclude="node_modules" \
  --exclude="dist" \
  --exclude="coverage" \
  --exclude=".git" \
  --exclude="*.log" \
  --exclude="temp" \
  --exclude="tmp" \
  "${DIRS_TO_BACKUP[@]}" \
  "${FILES_TO_BACKUP[@]}" 2>/dev/null || true

if [ -f "$BACKUP_ARCHIVE" ]; then
  BACKUP_SIZE=$(du -sh "$BACKUP_ARCHIVE" | cut -f1)
  echo "✅ File backup completed: $BACKUP_SIZE"
else
  echo "❌ File backup failed"
  exit 1
fi

# Cleanup old backups
echo "🧹 Cleaning up old file backups..."
find "$FILE_BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "💾 File backup finished: $TIMESTAMP"
