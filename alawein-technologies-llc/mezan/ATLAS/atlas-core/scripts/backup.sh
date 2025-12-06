#!/bin/bash
# Backup script for MEZAN ATLAS

set -e

# Configuration
BACKUP_DIR=${BACKUP_DIR:-"/backup"}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}
S3_BUCKET=${S3_BUCKET:-""}
S3_PREFIX=${S3_PREFIX:-"atlas-backups"}
ENCRYPTION_KEY=${ENCRYPTION_KEY:-""}
COMPRESS=${COMPRESS:-true}
NOTIFY_SLACK=${SLACK_WEBHOOK:-""}
NOTIFY_EMAIL=${EMAIL_TO:-""}

# Timestamp for backup naming
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="atlas-backup-${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

send_notification() {
    local status="$1"
    local message="$2"
    local size="$3"

    # Slack notification
    if [ -n "$NOTIFY_SLACK" ]; then
        local color="good"
        [ "$status" = "failed" ] && color="danger"

        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\":\"ATLAS Backup ${status}\",
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"fields\": [
                        {\"title\": \"Status\", \"value\": \"${status}\", \"short\": true},
                        {\"title\": \"Size\", \"value\": \"${size}\", \"short\": true},
                        {\"title\": \"Message\", \"value\": \"${message}\"},
                        {\"title\": \"Time\", \"value\": \"$(date)\"}
                    ]
                }]
            }" \
            "$NOTIFY_SLACK" 2>/dev/null || true
    fi

    # Email notification
    if [ -n "$NOTIFY_EMAIL" ] && command -v mail &> /dev/null; then
        echo -e "Backup Status: ${status}\nSize: ${size}\nMessage: ${message}\nTime: $(date)" | \
            mail -s "ATLAS Backup ${status}" "$NOTIFY_EMAIL" 2>/dev/null || true
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        log_info "Created backup directory: $BACKUP_DIR"
    fi

    # Check available disk space
    local available_space=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local required_space=1048576  # 1GB in KB

    if [ "$available_space" -lt "$required_space" ]; then
        log_error "Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB"
        exit 1
    fi

    # Check Docker
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running"
        exit 1
    fi

    # Check AWS CLI if S3 backup is enabled
    if [ -n "$S3_BUCKET" ]; then
        if ! command -v aws &> /dev/null; then
            log_error "AWS CLI is not installed but S3 backup is configured"
            exit 1
        fi
    fi

    log_info "Prerequisites check passed"
}

backup_database() {
    log_info "Backing up PostgreSQL database..."

    local db_backup_file="${BACKUP_DIR}/${BACKUP_NAME}-postgres.sql"

    # Check if PostgreSQL container is running
    if docker-compose ps postgres | grep -q "Up"; then
        # Perform database dump
        docker-compose exec -T postgres pg_dumpall -U atlas > "$db_backup_file" 2>/dev/null

        if [ $? -eq 0 ]; then
            # Compress if enabled
            if [ "$COMPRESS" = true ]; then
                gzip "$db_backup_file"
                db_backup_file="${db_backup_file}.gz"
            fi

            # Encrypt if key is provided
            if [ -n "$ENCRYPTION_KEY" ]; then
                openssl enc -aes-256-cbc -salt -in "$db_backup_file" \
                    -out "${db_backup_file}.enc" -pass pass:"$ENCRYPTION_KEY"
                rm "$db_backup_file"
                db_backup_file="${db_backup_file}.enc"
            fi

            local size=$(du -h "$db_backup_file" | cut -f1)
            log_info "Database backup completed: ${db_backup_file} (${size})"
        else
            log_error "Database backup failed"
            return 1
        fi
    else
        log_warn "PostgreSQL container is not running, skipping database backup"
    fi
}

backup_redis() {
    log_info "Backing up Redis data..."

    local redis_backup_file="${BACKUP_DIR}/${BACKUP_NAME}-redis.rdb"

    # Check if Redis container is running
    if docker-compose ps redis | grep -q "Up"; then
        # Trigger Redis SAVE
        docker-compose exec -T redis redis-cli BGSAVE 2>/dev/null

        # Wait for save to complete
        sleep 5

        # Copy Redis dump file
        docker-compose exec -T redis cat /data/atlas.rdb > "$redis_backup_file" 2>/dev/null

        if [ $? -eq 0 ]; then
            # Compress if enabled
            if [ "$COMPRESS" = true ]; then
                gzip "$redis_backup_file"
                redis_backup_file="${redis_backup_file}.gz"
            fi

            # Encrypt if key is provided
            if [ -n "$ENCRYPTION_KEY" ]; then
                openssl enc -aes-256-cbc -salt -in "$redis_backup_file" \
                    -out "${redis_backup_file}.enc" -pass pass:"$ENCRYPTION_KEY"
                rm "$redis_backup_file"
                redis_backup_file="${redis_backup_file}.enc"
            fi

            local size=$(du -h "$redis_backup_file" | cut -f1)
            log_info "Redis backup completed: ${redis_backup_file} (${size})"
        else
            log_error "Redis backup failed"
            return 1
        fi
    else
        log_warn "Redis container is not running, skipping Redis backup"
    fi
}

backup_volumes() {
    log_info "Backing up Docker volumes..."

    local volumes_backup_file="${BACKUP_DIR}/${BACKUP_NAME}-volumes.tar"

    # List of volumes to backup
    local volumes=(
        "atlas-core_atlas-data"
        "atlas-core_atlas-logs"
        "atlas-core_atlas-cache"
    )

    # Create tar archive of volumes
    docker run --rm \
        $(printf -- "-v %s:/backup/%s:ro " "${volumes[@]}" "${volumes[@]}") \
        -v "$BACKUP_DIR:/output" \
        alpine tar -cf "/output/$(basename $volumes_backup_file)" -C /backup . 2>/dev/null

    if [ $? -eq 0 ]; then
        # Compress if enabled
        if [ "$COMPRESS" = true ]; then
            gzip "$volumes_backup_file"
            volumes_backup_file="${volumes_backup_file}.gz"
        fi

        # Encrypt if key is provided
        if [ -n "$ENCRYPTION_KEY" ]; then
            openssl enc -aes-256-cbc -salt -in "$volumes_backup_file" \
                -out "${volumes_backup_file}.enc" -pass pass:"$ENCRYPTION_KEY"
            rm "$volumes_backup_file"
            volumes_backup_file="${volumes_backup_file}.enc"
        fi

        local size=$(du -h "$volumes_backup_file" | cut -f1)
        log_info "Volumes backup completed: ${volumes_backup_file} (${size})"
    else
        log_error "Volumes backup failed"
        return 1
    fi
}

backup_config() {
    log_info "Backing up configuration files..."

    local config_backup_file="${BACKUP_DIR}/${BACKUP_NAME}-config.tar"

    # Create tar archive of configuration files
    tar -cf "$config_backup_file" \
        -C .. \
        docker-compose.yml \
        docker-compose.*.yml \
        .env* \
        config/ \
        scripts/ \
        2>/dev/null || true

    if [ -f "$config_backup_file" ]; then
        # Compress if enabled
        if [ "$COMPRESS" = true ]; then
            gzip "$config_backup_file"
            config_backup_file="${config_backup_file}.gz"
        fi

        # Encrypt if key is provided
        if [ -n "$ENCRYPTION_KEY" ]; then
            openssl enc -aes-256-cbc -salt -in "$config_backup_file" \
                -out "${config_backup_file}.enc" -pass pass:"$ENCRYPTION_KEY"
            rm "$config_backup_file"
            config_backup_file="${config_backup_file}.enc"
        fi

        local size=$(du -h "$config_backup_file" | cut -f1)
        log_info "Configuration backup completed: ${config_backup_file} (${size})"
    fi
}

create_manifest() {
    log_info "Creating backup manifest..."

    local manifest_file="${BACKUP_DIR}/${BACKUP_NAME}-manifest.json"

    cat > "$manifest_file" <<EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "version": "$(git describe --tags --always 2>/dev/null || echo 'unknown')",
    "hostname": "$(hostname)",
    "files": [
        $(ls -1 ${BACKUP_DIR}/${BACKUP_NAME}-* | xargs -I {} basename {} | awk '{printf "\"%s\",", $0}' | sed 's/,$//')
    ],
    "environment": "$ENVIRONMENT",
    "compressed": $COMPRESS,
    "encrypted": $([ -n "$ENCRYPTION_KEY" ] && echo "true" || echo "false"),
    "sizes": {
        $(ls -lh ${BACKUP_DIR}/${BACKUP_NAME}-* | awk '{printf "\"%s\": \"%s\",", $9, $5}' | sed 's/,$//')
    }
}
EOF

    log_info "Manifest created: ${manifest_file}"
}

upload_to_s3() {
    if [ -n "$S3_BUCKET" ]; then
        log_info "Uploading backup to S3..."

        local s3_path="s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_NAME}/"

        # Upload all backup files
        aws s3 cp "${BACKUP_DIR}/" "$s3_path" \
            --recursive \
            --exclude "*" \
            --include "${BACKUP_NAME}-*" \
            --storage-class STANDARD_IA

        if [ $? -eq 0 ]; then
            log_info "Backup uploaded to S3: ${s3_path}"

            # Create lifecycle rule for auto-deletion
            aws s3api put-bucket-lifecycle-configuration \
                --bucket "$S3_BUCKET" \
                --lifecycle-configuration "{
                    \"Rules\": [{
                        \"Id\": \"atlas-backup-retention\",
                        \"Status\": \"Enabled\",
                        \"Prefix\": \"${S3_PREFIX}/\",
                        \"Expiration\": {
                            \"Days\": ${BACKUP_RETENTION_DAYS}
                        }
                    }]
                }" 2>/dev/null || true
        else
            log_error "Failed to upload backup to S3"
            return 1
        fi
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups..."

    # Remove local backups older than retention period
    find "$BACKUP_DIR" -name "atlas-backup-*" -type f -mtime +${BACKUP_RETENTION_DAYS} -delete

    # Count remaining backups
    local backup_count=$(ls -1 ${BACKUP_DIR}/atlas-backup-* 2>/dev/null | wc -l)
    log_info "Retained ${backup_count} local backups"
}

verify_backup() {
    log_info "Verifying backup integrity..."

    local all_valid=true

    # Check each backup file
    for file in ${BACKUP_DIR}/${BACKUP_NAME}-*; do
        if [ -f "$file" ]; then
            # Check file size
            if [ ! -s "$file" ]; then
                log_error "Backup file is empty: $file"
                all_valid=false
            fi

            # If encrypted, try to decrypt header
            if [[ "$file" == *.enc ]]; then
                if ! openssl enc -aes-256-cbc -d -in "$file" -pass pass:"$ENCRYPTION_KEY" 2>/dev/null | head -n 1 >/dev/null; then
                    log_error "Failed to verify encrypted file: $file"
                    all_valid=false
                fi
            fi

            # If compressed, test integrity
            if [[ "$file" == *.gz ]]; then
                if ! gzip -t "$file" 2>/dev/null; then
                    log_error "Corrupted compressed file: $file"
                    all_valid=false
                fi
            fi
        fi
    done

    if [ "$all_valid" = true ]; then
        log_info "Backup verification passed"
        return 0
    else
        log_error "Backup verification failed"
        return 1
    fi
}

main() {
    log_info "=========================================="
    log_info "Starting MEZAN ATLAS Backup"
    log_info "=========================================="

    local start_time=$(date +%s)
    local backup_success=true
    local total_size="0"

    # Run backup steps
    check_prerequisites

    backup_database || backup_success=false
    backup_redis || backup_success=false
    backup_volumes || backup_success=false
    backup_config || backup_success=false

    if [ "$backup_success" = true ]; then
        create_manifest
        verify_backup || backup_success=false
    fi

    if [ "$backup_success" = true ]; then
        upload_to_s3 || log_warn "S3 upload failed, but local backup is complete"
        cleanup_old_backups

        # Calculate total backup size
        total_size=$(du -sh ${BACKUP_DIR}/${BACKUP_NAME}-* 2>/dev/null | awk '{sum+=$1} END {print sum}' | numfmt --to=iec)

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        log_info "=========================================="
        log_info "Backup completed successfully!"
        log_info "Name: ${BACKUP_NAME}"
        log_info "Size: ${total_size}"
        log_info "Duration: ${duration} seconds"
        log_info "Location: ${BACKUP_DIR}"
        log_info "=========================================="

        send_notification "success" "Backup completed: ${BACKUP_NAME}" "$total_size"
    else
        log_error "Backup failed!"
        send_notification "failed" "Backup failed: ${BACKUP_NAME}" "N/A"
        exit 1
    fi
}

# Handle script arguments
case "$1" in
    --restore)
        # Restore functionality would go here
        log_error "Restore functionality not yet implemented"
        exit 1
        ;;
    --list)
        # List available backups
        echo "Available backups:"
        ls -lht ${BACKUP_DIR}/atlas-backup-* 2>/dev/null || echo "No backups found"
        [ -n "$S3_BUCKET" ] && aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" || true
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  (none)      Run backup"
        echo "  --list      List available backups"
        echo "  --restore   Restore from backup (not implemented)"
        echo "  --help      Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  BACKUP_DIR              Local backup directory"
        echo "  BACKUP_RETENTION_DAYS   Days to keep backups"
        echo "  S3_BUCKET               S3 bucket for remote backups"
        echo "  ENCRYPTION_KEY          Encryption key for backups"
        exit 0
        ;;
    *)
        main
        ;;
esac