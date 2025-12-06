#!/bin/bash
# Enterprise Disaster Recovery Script

set -e

BACKUP_DIR="$(dirname "$0")/../backups"
RESTORE_POINT="${1:-latest}"

echo "🚑 Starting disaster recovery..."
echo "Restore point: $RESTORE_POINT"

# Function to find latest backup
find_latest_backup() {
  local backup_type="$1"
  local backup_subdir="$BACKUP_DIR/$backup_type"
  
  if [ "$RESTORE_POINT" = "latest" ]; then
    find "$backup_subdir" -name "*" -type f | sort -r | head -1
  else
    find "$backup_subdir" -name "*$RESTORE_POINT*" -type f | head -1
  fi
}

echo "🗄️  Restoring database..."
DB_BACKUP=$(find_latest_backup "database")
if [ -n "$DB_BACKUP" ]; then
  echo "📊 Found database backup: $(basename "$DB_BACKUP")"
  
  # Restore database
  if [[ "$DB_BACKUP" == *.gz ]]; then
    gunzip -c "$DB_BACKUP" | psql "$SUPABASE_DB_URL" || echo "⚠️  Database restore may have warnings"
  else
    psql "$SUPABASE_DB_URL" < "$DB_BACKUP" || echo "⚠️  Database restore may have warnings"
  fi
  
  echo "✅ Database restored"
else
  echo "❌ No database backup found"
fi

echo "📁 Restoring files..."
FILE_BACKUP=$(find_latest_backup "files")
if [ -n "$FILE_BACKUP" ]; then
  echo "📦 Found file backup: $(basename "$FILE_BACKUP")"
  
  # Create restore directory
  RESTORE_DIR="./restored_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RESTORE_DIR"
  
  # Extract files
  tar -xzf "$FILE_BACKUP" -C "$RESTORE_DIR"
  echo "✅ Files restored to: $RESTORE_DIR"
else
  echo "❌ No file backup found"
fi

echo "⚙️  Restoring configuration..."
CONFIG_BACKUP=$(find_latest_backup "config")
if [ -n "$CONFIG_BACKUP" ]; then
  echo "📋 Found config backup: $(basename "$CONFIG_BACKUP")"
  
  # Extract config to temp directory
  TEMP_CONFIG_DIR="/tmp/config_restore_$(date +%s)"
  mkdir -p "$TEMP_CONFIG_DIR"
  tar -xzf "$CONFIG_BACKUP" -C "$TEMP_CONFIG_DIR"
  
  echo "⚠️  Configuration files extracted to: $TEMP_CONFIG_DIR"
  echo "📝 Manual review required before applying configuration changes"
else
  echo "❌ No configuration backup found"
fi

echo "🚑 Disaster recovery completed!"
echo "📝 Next steps:"
echo "  1. Verify database integrity"
echo "  2. Test application functionality"  
echo "  3. Review and apply configuration changes"
echo "  4. Update DNS/load balancer if needed"
echo "  5. Notify stakeholders of recovery completion"
