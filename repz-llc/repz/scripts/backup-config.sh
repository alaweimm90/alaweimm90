#!/bin/bash
# Enterprise Configuration Backup Script

set -e

BACKUP_DIR="$(dirname "$0")/../backups"
CONFIG_BACKUP_DIR="$BACKUP_DIR/config"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "⚙️  Starting configuration backup..."

# Configuration files to backup
CONFIG_FILES=(
  ".env.example"
  ".gitignore"
  ".eslintrc.json"
  "lighthouserc.json"
  "playwright.config.ts"
  "performance-alert-config.json"
)

# Create config backup directory for this timestamp
TIMESTAMPED_DIR="$CONFIG_BACKUP_DIR/$TIMESTAMP"
mkdir -p "$TIMESTAMPED_DIR"

# Backup configuration files
for config_file in "${CONFIG_FILES[@]}"; do
  if [ -f "$config_file" ]; then
    cp "$config_file" "$TIMESTAMPED_DIR/"
    echo "📋 Backed up: $config_file"
  fi
done

# Backup all JSON configuration files
find . -maxdepth 1 -name "*.json" -not -path "./node_modules/*" -not -path "./dist/*" | while read json_file; do
  cp "$json_file" "$TIMESTAMPED_DIR/"
  echo "📋 Backed up: $json_file"
done

# Create archive
cd "$CONFIG_BACKUP_DIR"
tar -czf "config_$TIMESTAMP.tar.gz" "$TIMESTAMP"
rm -rf "$TIMESTAMP"

echo "✅ Configuration backup completed"
echo "💾 Configuration backup finished: $TIMESTAMP"
