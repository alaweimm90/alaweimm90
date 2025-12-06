# Backup System Setup Guide

Complete setup guide for configuring the Live It Iconic backup and disaster recovery system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (5 minutes)](#quick-setup-5-minutes)
3. [Full Setup with Cloud Storage (15 minutes)](#full-setup-with-cloud-storage-15-minutes)
4. [Verification & Testing](#verification--testing)
5. [Production Deployment](#production-deployment)

---

## Prerequisites

### Required

- Node.js 18+ installed
- npm or yarn package manager
- Access to Supabase project (service key)
- Live It Iconic codebase cloned

### Optional (for S3 backups)

- AWS account with S3 access
- AWS IAM credentials with S3 permissions
- S3 bucket created or ready to be created

---

## Quick Setup (5 minutes)

### 1. Create Backup Directory

```bash
mkdir -p /backups
chmod 700 /backups

# On macOS with Homebrew
sudo mkdir -p /backups
sudo chown $USER:staff /backups
```

### 2. Configure Environment

```bash
# Copy configuration template
cp scripts/backup/.env.example scripts/backup/.env

# Edit with your Supabase credentials
nano scripts/backup/.env
```

### Minimal Configuration

```bash
# scripts/backup/.env

BACKUP_DIR=/backups
SUPABASE_URL=https://[your-project].supabase.co
SUPABASE_SERVICE_KEY=[your-service-key]
ENABLE_S3_UPLOAD=false  # Skip S3 for now
```

### 3. Test Backup

```bash
# Create first backup
npm run backup:create

# Expected output:
# Created backup directory: /backups/2024-01-15T10-30-45-123Z
# ✓ Backed up 5234 records from products
# ✓ Backup completed successfully
```

### 4. Verify Backup

```bash
# List backups
npm run backup:list

# Validate backup
npm run backup:validate /backups/2024-01-15T10-30-45-123Z
```

**Success!** You now have working backups. Continue below for S3 and production setup.

---

## Full Setup with Cloud Storage (15 minutes)

### Step 1: Create AWS S3 Bucket

```bash
# Using AWS CLI
aws s3api create-bucket \
  --bucket live-it-iconic-backups \
  --region us-east-1 \
  --create-bucket-configuration LocationConstraint=us-east-1

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
  --bucket live-it-iconic-backups \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket live-it-iconic-backups \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Block public access
aws s3api put-public-access-block \
  --bucket live-it-iconic-backups \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

### Step 2: Create IAM User for Backups

```bash
# Create dedicated user for backups
aws iam create-user --user-name live-it-iconic-backup

# Create access keys
aws iam create-access-key --user-name live-it-iconic-backup
# Note: Save the AccessKeyId and SecretAccessKey
```

### Step 3: Attach S3 Permissions

Create file `backup-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketVersioning",
        "s3:ListBucketVersions"
      ],
      "Resource": [
        "arn:aws:s3:::live-it-iconic-backups",
        "arn:aws:s3:::live-it-iconic-backups/*"
      ]
    }
  ]
}
```

```bash
# Attach policy
aws iam put-user-policy \
  --user-name live-it-iconic-backup \
  --policy-name backup-policy \
  --policy-document file://backup-policy.json
```

### Step 4: Update Environment

```bash
# scripts/backup/.env

ENABLE_S3_UPLOAD=true
S3_BACKUP_BUCKET=live-it-iconic-backups
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=[your-access-key]
AWS_SECRET_ACCESS_KEY=[your-secret-key]
S3_RETENTION_DAYS=90
```

### Step 5: Install AWS SDK

```bash
npm install @aws-sdk/client-s3
```

### Step 6: Test S3 Upload

```bash
# Create test backup with S3 upload
npm run backup:create

# Verify in S3
aws s3 ls s3://live-it-iconic-backups/ --recursive
```

---

## Verification & Testing

### Health Check

```bash
# Verify backup system health
npm run backup:stats

# Expected output:
# {
#   "state": {
#     "lastBackupTime": "2024-01-15T10:30:00Z",
#     "lastBackupStatus": "success",
#     "backupCount": 1,
#     "errorCount": 0
#   }
# }
```

### Test Restore

```bash
# In a test database only!
# Create backup
npm run backup:create

# Modify some test data
# (optional - to see if restore brings back original)

# List backups
npm run backup:list

# Test restore (requires confirmation)
npm run backup:restore /backups/[backup-timestamp]
```

### Database Integrity Check

```bash
# Verify record counts
npm run db:verify-counts

# Expected output:
# Table          Expected    Actual      Status
# products       5234        5234        ✓
# orders         12543       12543       ✓
# users          3421        3421        ✓
```

---

## Production Deployment

### Step 1: Update .env.example

```bash
# Add backup configuration to main .env.example
cat scripts/backup/.env.example >> .env.example
```

### Step 2: Configure Notifications

```bash
# Update scripts/backup/.env

ENABLE_NOTIFICATIONS=true
BACKUP_NOTIFICATION_EMAIL=ops@liveitic.com
BACKUP_WEBHOOK_URL=https://hooks.slack.com/services/xxx

# For AWS SNS notifications
BACKUP_ALERT_TOPIC_ARN=arn:aws:sns:us-east-1:123456789:backups
```

### Step 3: Schedule Automated Backups

#### Option A: Using Cron (Linux/macOS)

```bash
# Edit crontab
crontab -e

# Add entry for 2 AM UTC daily:
# 0 2 * * * cd /path/to/live-it-iconic && npm run backup:start >> /var/log/backups.log 2>&1
```

#### Option B: Using AWS Lambda

```bash
# Create Lambda function
aws lambda create-function \
  --function-name live-it-iconic-backup \
  --runtime nodejs18.x \
  --handler index.handler \
  --zip-file fileb://function.zip

# Create CloudWatch rule
aws events put-rule \
  --name backup-schedule \
  --schedule-expression "cron(0 2 * * ? *)"

# Link rule to Lambda
aws events put-targets \
  --rule backup-schedule \
  --targets "Id"="1","Arn"="arn:aws:lambda:..."
```

#### Option C: Using Kubernetes CronJob

```yaml
# kubernetes/cronjob-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-job
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: live-it-iconic:latest
            command:
            - npm
            - run
            - backup:start
            env:
            - name: SUPABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secrets
                  key: url
            - name: SUPABASE_SERVICE_KEY
              valueFrom:
                secretKeyRef:
                  name: db-secrets
                  key: key
          restartPolicy: OnFailure
```

### Step 4: Monitor Backups

```bash
# View admin dashboard
# Navigate to: https://your-domain.com/admin/backups

# Monitor via CLI
npm run backup:stats

# Check logs
tail -f /backups/logs/$(date +%Y-%m-%d).log
```

### Step 5: Set Up Alerts

#### CloudWatch Alarms

```bash
# Alert if backup fails
aws cloudwatch put-metric-alarm \
  --alarm-name BackupFailure \
  --alarm-description "Alert if daily backup fails" \
  --metric-name BackupStatus \
  --namespace LiveItIconic \
  --statistic Average \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 0 \
  --comparison-operator LessThanOrEqualToThreshold \
  --alarm-actions arn:aws:sns:us-east-1:123456789:on-call
```

#### Slack Integration

```bash
# In scripts/backup/automated-backup.ts, notifications are already configured
# Just add your webhook URL to .env:
BACKUP_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Step 6: Documentation

```bash
# Create team wiki page
# Include:
# - How to create manual backup
# - How to restore from backup
# - When to contact DBA
# - Emergency contacts

# Example location: /wiki/BACKUPS.md
```

### Step 7: Schedule DR Drills

```bash
# Monthly disaster recovery test
# Create calendar event:
# - Frequency: 1st Friday of each month, 2 PM UTC
# - Duration: 2 hours
# - Participants: DevOps, DBA, Engineering Lead
# - Checklist: See /docs/DISASTER_RECOVERY.md
```

---

## Troubleshooting Setup

### Backup Directory Permission Error

```bash
# Error: "Permission denied: /backups"
sudo chown $USER:$(id -gn) /backups
chmod 700 /backups
```

### Supabase Connection Error

```bash
# Error: "Failed to connect to Supabase"

# Verify credentials
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY

# Test connection
npx tsx -e "
import { createClient } from '@supabase/supabase-js'
const client = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY)
const { error } = await client.from('products').select('count(*)')
console.log(error || 'Connected!')
"
```

### AWS S3 Permission Error

```bash
# Error: "AccessDenied: User is not authorized to perform: s3:PutObject"

# Verify IAM policy
aws iam get-user-policy --user-name live-it-iconic-backup --policy-name backup-policy

# Verify bucket exists and is accessible
aws s3 ls s3://live-it-iconic-backups/
```

### Disk Space Issues

```bash
# Error: "No space left on device"

# Check available space
df -h /backups

# Clean old backups
npm run backup:cleanup --retention-days=7

# Or manually delete
rm -rf /backups/2024-01-01T*
```

---

## Next Steps

1. ✅ **Setup Complete** - Backups are now running
2. **Verify Admin Dashboard** - Visit `/admin/backups`
3. **Test Restore** - Practice restoring in staging
4. **Schedule Monthly Drills** - Test full recovery procedures
5. **Monitor Backups** - Check daily backup completion

---

## Support

For issues or questions:

1. Check `/scripts/backup/README.md` for operational guide
2. Review `/docs/DISASTER_RECOVERY.md` for recovery procedures
3. Contact: devops@liveitic.com

---

**Setup Guide Version:** 1.0
**Last Updated:** 2024-01-15
