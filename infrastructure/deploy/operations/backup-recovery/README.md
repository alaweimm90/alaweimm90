# ATLAS Backup and Recovery Procedures

This document outlines the backup and recovery procedures for ATLAS production environments.

## Backup Strategy

### Backup Types

#### 1. Database Backups
- **Frequency**: Daily full backups + hourly incremental
- **Retention**: 30 days for daily, 7 days for hourly
- **Storage**: Encrypted cloud storage (S3/GCS/Azure Blob)
- **Encryption**: AES-256 encryption at rest and in transit

#### 2. Configuration Backups
- **Frequency**: After every configuration change
- **Scope**: Kubernetes manifests, Helm values, ConfigMaps, Secrets
- **Storage**: Git repository + cloud storage
- **Encryption**: Repository encryption + secret encryption

#### 3. Application Data Backups
- **Frequency**: Real-time replication + daily snapshots
- **Scope**: User data, task history, metrics data
- **Storage**: Multi-region replication
- **Encryption**: Database-level encryption

#### 4. Infrastructure Backups
- **Frequency**: Weekly infrastructure snapshots
- **Scope**: VM images, disk snapshots, network configuration
- **Storage**: Cloud provider snapshots
- **Encryption**: Cloud provider encryption

## Automated Backup Procedures

### Database Backup

#### PostgreSQL Backup
```bash
#!/bin/bash
# Daily PostgreSQL backup script

BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/atlas_${DATE}.sql.gz"

# Create backup
kubectl exec -it atlas-postgres-0 -n atlas-system -- bash -c \
  "pg_dump -U atlas -d atlas | gzip > /tmp/backup.sql.gz"

# Copy to persistent volume
kubectl cp atlas-system/atlas-postgres-0:/tmp/backup.sql.gz ${BACKUP_FILE}

# Upload to cloud storage
aws s3 cp ${BACKUP_FILE} s3://atlas-backups/postgres/${DATE}.sql.gz

# Cleanup old backups (keep last 30 days)
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +30 -delete
```

#### Redis Backup
```bash
#!/bin/bash
# Redis backup script

BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

# Trigger Redis save
kubectl exec -it atlas-redis-0 -n atlas-system -- redis-cli save

# Copy RDB file
kubectl cp atlas-system/atlas-redis-0:/data/dump.rdb ${BACKUP_DIR}/redis_${DATE}.rdb

# Upload to cloud storage
aws s3 cp ${BACKUP_DIR}/redis_${DATE}.rdb s3://atlas-backups/redis/${DATE}.rdb
```

### Configuration Backup

#### Kubernetes Configuration Backup
```bash
#!/bin/bash
# Kubernetes configuration backup

BACKUP_DIR="/backups/kubernetes"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup all ATLAS resources
kubectl get all,configmap,secret,ingress -n atlas-system -o yaml > ${BACKUP_DIR}/atlas_${DATE}.yaml

# Backup Helm releases
helm list -n atlas-system -o yaml > ${BACKUP_DIR}/helm_releases_${DATE}.yaml

# Commit to Git
cd /backups/git
git add .
git commit -m "Backup ${DATE}"
git push origin main

# Upload to cloud storage
aws s3 cp ${BACKUP_DIR}/atlas_${DATE}.yaml s3://atlas-backups/kubernetes/${DATE}.yaml
```

### Application Data Backup

#### Elasticsearch Backup
```bash
#!/bin/bash
# Elasticsearch snapshot backup

REPO_NAME="atlas_backup"
SNAPSHOT_NAME="snapshot_$(date +%Y%m%d_%H%M%S)"

# Register snapshot repository
curl -X PUT "atlas-elasticsearch:9200/_snapshot/${REPO_NAME}" -H 'Content-Type: application/json' -d'
{
  "type": "s3",
  "settings": {
    "bucket": "atlas-backups",
    "region": "us-east-1",
    "role_arn": "arn:aws:iam::123456789012:role/ElasticsearchBackupRole"
  }
}'

# Create snapshot
curl -X PUT "atlas-elasticsearch:9200/_snapshot/${REPO_NAME}/${SNAPSHOT_NAME}"

# Verify snapshot
curl -X GET "atlas-elasticsearch:9200/_snapshot/${REPO_NAME}/${SNAPSHOT_NAME}"
```

## Recovery Procedures

### Database Recovery

#### PostgreSQL Recovery
```bash
#!/bin/bash
# PostgreSQL recovery script

BACKUP_FILE="atlas_20231201_020000.sql.gz"
NAMESPACE="atlas-system"

# Scale down ATLAS services
kubectl scale deployment --all --replicas=0 -n ${NAMESPACE}

# Restore database
kubectl exec -it atlas-postgres-0 -n ${NAMESPACE} -- bash -c \
  "dropdb -U atlas atlas && createdb -U atlas atlas"

kubectl exec -it atlas-postgres-0 -n ${NAMESPACE} -- bash -c \
  "gunzip -c /backups/${BACKUP_FILE} | psql -U atlas -d atlas"

# Scale up services
kubectl scale deployment --all --replicas=1 -n ${NAMESPACE}
```

#### Point-in-Time Recovery
```bash
#!/bin/bash
# Point-in-time recovery

RECOVERY_TIME="2023-12-01 14:30:00 UTC"
BACKUP_FILE="atlas_20231201_020000.sql.gz"

# Restore base backup
# ... (same as above)

# Apply WAL logs up to recovery time
kubectl exec -it atlas-postgres-0 -n atlas-system -- bash -c \
  "psql -U atlas -d atlas -c \"SELECT pg_wal_replay_resume()\""

# Stop recovery at specific time
kubectl exec -it atlas-postgres-0 -n atlas-system -- psql -U atlas -d atlas -c \
  "SELECT pg_wal_replay_pause()"

# Verify data integrity
kubectl exec -it atlas-postgres-0 -n atlas-system -- psql -U atlas -d atlas -c \
  "SELECT COUNT(*) FROM tasks WHERE created_at <= '${RECOVERY_TIME}'"
```

### Application Recovery

#### Full System Recovery
```bash
#!/bin/bash
# Complete system recovery

BACKUP_DATE="20231201_020000"
NAMESPACE="atlas-system"

# 1. Restore infrastructure (if needed)
terraform apply -var-file=backup_${BACKUP_DATE}.tfvars

# 2. Restore Kubernetes configuration
kubectl apply -f /backups/kubernetes/atlas_${BACKUP_DATE}.yaml

# 3. Restore database
./restore_postgres.sh ${BACKUP_DATE}

# 4. Restore Redis
./restore_redis.sh ${BACKUP_DATE}

# 5. Restore Elasticsearch
./restore_elasticsearch.sh ${BACKUP_DATE}

# 6. Verify system health
kubectl get pods -n ${NAMESPACE}
curl -f https://api.atlas.your-domain.com/health
```

### Configuration Recovery

#### Helm Release Recovery
```bash
# List available releases
helm history atlas -n atlas-system

# Rollback to specific version
helm rollback atlas 5 -n atlas-system

# Force upgrade if needed
helm upgrade --install atlas ./atlas -f values.yaml --force
```

#### Manual Configuration Recovery
```bash
# Restore from Git backup
cd /backups/git
git checkout <commit-hash>
kubectl apply -f kubernetes/

# Or restore from cloud storage
aws s3 cp s3://atlas-backups/kubernetes/atlas_20231201.yaml .
kubectl apply -f atlas_20231201.yaml
```

## Disaster Recovery

### Multi-Region Failover

#### Primary Region Failure
```bash
#!/bin/bash
# Disaster recovery failover script

PRIMARY_REGION="us-east-1"
SECONDARY_REGION="us-west-2"

# 1. Promote secondary database
aws rds failover-db-cluster --db-cluster-identifier atlas-postgres --target-db-instance-identifier atlas-postgres-replica-${SECONDARY_REGION}

# 2. Update DNS to point to secondary region
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://dns-failover.json

# 3. Scale up secondary region services
kubectl config use-context ${SECONDARY_REGION}
kubectl scale deployment atlas-api-gateway --replicas=5 -n atlas-system

# 4. Verify failover
curl -f https://api.atlas.your-domain.com/health
```

#### Failback to Primary
```bash
#!/bin/bash
# Failback to primary region

# 1. Ensure primary region is healthy
kubectl config use-context ${PRIMARY_REGION}
kubectl get pods -n atlas-system

# 2. Sync data from secondary to primary
# (Using logical replication or backup/restore)

# 3. Update DNS back to primary
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://dns-failback.json

# 4. Scale down secondary region
kubectl config use-context ${SECONDARY_REGION}
kubectl scale deployment atlas-api-gateway --replicas=0 -n atlas-system
```

### Data Center Recovery

#### Complete Data Center Loss
```bash
#!/bin/bash
# Complete data center recovery

NEW_REGION="us-west-2"
BACKUP_DATE=$(date -d '1 hour ago' +%Y%m%d_%H%M%S)

# 1. Provision new infrastructure
terraform workspace select ${NEW_REGION}
terraform apply

# 2. Restore latest backups
./restore_postgres.sh ${BACKUP_DATE}
./restore_redis.sh ${BACKUP_DATE}
./restore_elasticsearch.sh ${BACKUP_DATE}

# 3. Deploy applications
kubectl apply -f kubernetes/

# 4. Update DNS
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://dns-update.json

# 5. Verify recovery
./run_health_checks.sh
```

## Testing and Validation

### Backup Verification
```bash
#!/bin/bash
# Backup verification script

BACKUP_FILE="atlas_20231201.sql.gz"

# Test backup integrity
gunzip -c ${BACKUP_FILE} | head -20

# Test restore to temporary database
createdb test_restore
gunzip -c ${BACKUP_FILE} | psql test_restore
psql test_restore -c "SELECT COUNT(*) FROM tasks"

# Cleanup
dropdb test_restore
```

### Recovery Testing

#### Regular Recovery Drills
- Monthly full system recovery test
- Quarterly disaster recovery simulation
- Annual multi-region failover test

#### Recovery Time Objectives (RTO)
- Database: 1 hour
- Application services: 30 minutes
- Full system: 4 hours

#### Recovery Point Objectives (RPO)
- Database: 1 hour
- Configuration: 1 hour
- Application data: 15 minutes

## Monitoring and Alerting

### Backup Monitoring
```yaml
# Prometheus alerting rules for backups
groups:
  - name: backup_alerts
    rules:
      - alert: BackupFailed
        expr: atlas_backup_status{status="failed"} == 1
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "ATLAS backup failed"
          description: "Backup job {{ $labels.job }} failed"

      - alert: BackupStale
        expr: time() - atlas_backup_last_success > 86400
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "ATLAS backup is stale"
          description: "Last successful backup was more than 24 hours ago"
```

### Recovery Monitoring
- Monitor RTO/RPO compliance
- Track recovery success rates
- Alert on recovery failures

## Compliance and Auditing

### Backup Audit Logs
- Maintain detailed logs of all backup operations
- Track backup success/failure rates
- Audit backup data integrity

### Compliance Requirements
- SOX: Financial data backup retention (7 years)
- GDPR: User data backup and recovery capabilities
- HIPAA: Protected health information backup security

### Encryption Standards
- AES-256 for data at rest
- TLS 1.3 for data in transit
- Key rotation every 90 days

## Cost Optimization

### Backup Storage Costs
- Use compression to reduce storage costs
- Implement tiered storage (hot/cold/archive)
- Clean up old backups automatically

### Cross-Region Replication
- Balance RPO requirements with costs
- Use cheaper storage classes for older backups
- Implement intelligent retention policies

## Support and Escalation

### Backup Issues
1. Check backup job logs
2. Verify storage connectivity
3. Contact cloud provider support
4. Escalate to backup vendor if applicable

### Recovery Issues
1. Follow recovery runbooks
2. Contact database administrators
3. Engage infrastructure team
4. Escalate to executive leadership for major incidents

### Emergency Contacts
- Database Administrator: +1-555-0101
- Infrastructure Lead: +1-555-0102
- Security Officer: +1-555-0103
- Executive On-Call: +1-555-0104