#!/bin/bash
# Enterprise Database Backup Script

set -e

BACKUP_DIR="$(dirname "$0")/../backups"
DB_BACKUP_DIR="$BACKUP_DIR/database"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RETENTION_DAYS=30

echo "🗄️  Starting database backup..."

# Supabase database backup
if [ ! -z "$SUPABASE_DB_URL" ]; then
  echo "📊 Backing up Supabase database..."
  
  # Create schema backup
  pg_dump "$SUPABASE_DB_URL" --schema-only > "$DB_BACKUP_DIR/schema_$TIMESTAMP.sql"
  
  # Create data backup
  pg_dump "$SUPABASE_DB_URL" --data-only > "$DB_BACKUP_DIR/data_$TIMESTAMP.sql"
  
  # Create full backup
  pg_dump "$SUPABASE_DB_URL" > "$DB_BACKUP_DIR/full_$TIMESTAMP.sql"
  
  # Compress backups
  gzip "$DB_BACKUP_DIR/schema_$TIMESTAMP.sql"
  gzip "$DB_BACKUP_DIR/data_$TIMESTAMP.sql"
  gzip "$DB_BACKUP_DIR/full_$TIMESTAMP.sql"
  
  echo "✅ Database backup completed"
else
  echo "⚠️  No database URL configured"
fi

# Cleanup old backups
echo "🧹 Cleaning up old backups..."
find "$DB_BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
echo "✅ Cleanup completed"

echo "💾 Database backup finished: $TIMESTAMP"
