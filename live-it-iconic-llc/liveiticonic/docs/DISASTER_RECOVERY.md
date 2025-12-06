# Disaster Recovery Plan - Live It Iconic

## Executive Summary

This document outlines the comprehensive disaster recovery and business continuity strategy for Live It Iconic, ensuring minimal data loss and rapid service restoration in the event of system failures, data corruption, or disaster scenarios.

---

## Recovery Time and Point Objectives

### Recovery Time Objectives (RTO)

| Service Level | RTO | SLA |
|---|---|---|
| **Critical Services** | 1 hour | 99.9% |
| **Important Services** | 4 hours | 99% |
| **Non-Critical Services** | 24 hours | 95% |
| **Full System Recovery** | 24 hours | - |

### Recovery Point Objectives (RPO)

| Data Category | RPO | Backup Frequency |
|---|---|---|
| **Database (Products, Orders, Users)** | 24 hours | Daily at 2:00 AM UTC |
| **User Files & Media** | 1 hour | Continuous sync |
| **Application Configuration** | 0 hours | Version controlled (Git) |
| **Encryption Keys** | 0 hours | Secure vault (AWS Secrets Manager) |

---

## Backup Strategy

### 1. Automated Daily Backups

**Schedule:** 2:00 AM UTC every day

**Tables Backed Up:**
- `products` - Product catalog and descriptions
- `orders` - Order history and details
- `users` - User accounts and profiles
- `customers` - Customer information
- `inventory` - Stock levels and tracking
- `order_items` - Order line items
- `product_variants` - Product variations
- `reviews` - Customer reviews
- `wishlists` - User wishlists

**Storage:**
- **Local Storage:** `/backups` directory on server
- **Cloud Storage:** AWS S3 bucket (`live-it-iconic-backups`)
- **Retention:** 30 days local, 90 days S3

### 2. Backup Verification

**Weekly Testing:**
- Restore one random backup to staging environment
- Verify all tables restored correctly
- Check record counts match metadata
- Validate data integrity

**Monthly Testing:**
- Full disaster recovery drill
- Simulate complete system failure
- Test restore procedures end-to-end
- Document time and any issues

### 3. Backup Locations

| Location | Purpose | Retention | Encryption |
|---|---|---|---|
| Local Server | Rapid recovery | 30 days | AES-256 |
| AWS S3 | Long-term archive | 90 days | AES-256 |
| DR Site | Failover | Current | AES-256 |

---

## Backup File Structure

```
/backups/
├── 2024-01-15T02-00-00-000Z/
│   ├── metadata.json
│   ├── products.json
│   ├── orders.json
│   ├── users.json
│   ├── inventory.json
│   └── ... (other tables)
└── logs/
    └── 2024-01-15.log
```

### Metadata Format

```json
{
  "timestamp": "2024-01-15T02:00:00.000Z",
  "tables": ["products", "orders", "users", ...],
  "recordCounts": {
    "products": 5234,
    "orders": 12543,
    "users": 3421,
    ...
  },
  "totalRecords": 50000,
  "success": true
}
```

---

## Disaster Recovery Procedures

### Scenario 1: Database Corruption or Data Loss

**Estimated Recovery Time:** 30 minutes to 1 hour

**Steps:**

1. **Identify the Issue**
   ```bash
   # Check database health
   npm run db:health

   # Review error logs
   tail -f /var/log/supabase/error.log
   ```

2. **Determine Restore Point**
   ```bash
   # List available backups
   npm run backup:list

   # Validate most recent backup
   npm run backup:validate [backup-timestamp]
   ```

3. **Stop Application Services**
   ```bash
   # Stop the application to prevent further writes
   pm2 stop live-it-iconic
   # or
   docker-compose down
   ```

4. **Download Latest Backup**
   ```bash
   # From local storage
   cp -r /backups/[latest-backup] /restore/

   # Or from S3 if local backup unavailable
   npm run backup:download-s3 [backup-key]
   ```

5. **Validate Backup Integrity**
   ```bash
   # Verify backup files and metadata
   npm run backup:validate /restore/[backup-timestamp]
   ```

6. **Perform Restore**
   ```bash
   # Execute restore with confirmation
   SKIP_RESTORE_CONFIRMATION=false tsx scripts/backup/database-restore.ts /restore/[backup-timestamp]

   # Monitor progress
   tail -f restore.log
   ```

7. **Verify Restored Data**
   ```bash
   # Check record counts match metadata
   npm run db:verify-counts

   # Run data integrity checks
   npm run db:integrity-check
   ```

8. **Restart Services**
   ```bash
   # Restart application
   pm2 start live-it-iconic
   # or
   docker-compose up
   ```

9. **Monitor for Issues**
   ```bash
   # Check application logs
   npm run logs:app

   # Monitor database performance
   npm run db:monitor
   ```

10. **Post-Recovery Steps**
    - Notify stakeholders
    - Document issue root cause
    - Create incident report
    - Schedule post-mortem meeting

---

### Scenario 2: Complete System Failure

**Estimated Recovery Time:** 2-4 hours

**Prerequisites:**
- AWS account access
- Domain registrar access
- SSL certificate (already in ACM)
- Latest backup available in S3

**Steps:**

1. **Provision New Infrastructure**
   ```bash
   # Deploy to new EC2 instance using CloudFormation
   aws cloudformation create-stack \
     --stack-name live-it-iconic-dr \
     --template-body file://infrastructure/cf-template.yaml

   # Or use Terraform
   terraform apply -var="environment=dr"
   ```

2. **Deploy Application Code**
   ```bash
   # Clone latest code
   git clone https://github.com/live-it-iconic/platform.git
   cd platform

   # Install dependencies
   npm ci

   # Build application
   npm run build
   ```

3. **Download Latest Database Backup**
   ```bash
   # From S3
   aws s3 cp s3://live-it-iconic-backups/latest/ ./backup --recursive

   # Or use script
   npm run backup:download-s3 latest
   ```

4. **Initialize Supabase**
   ```bash
   # Set up new Supabase project
   supabase init

   # Apply migrations
   supabase migration up
   ```

5. **Restore Database from Backup**
   ```bash
   # Validate backup
   npm run backup:validate ./backup

   # Execute restore
   tsx scripts/backup/database-restore.ts ./backup

   # Verify restore completed
   npm run db:verify-counts
   ```

6. **Configure Environment Variables**
   ```bash
   # Create .env file with new resources
   cp .env.example .env

   # Update with new:
   # - Supabase URL and keys
   # - AWS credentials
   # - Third-party API keys
   # - Email service credentials
   ```

7. **Run Database Integrity Checks**
   ```bash
   # Verify referential integrity
   npm run db:check-constraints

   # Check for orphaned records
   npm run db:check-orphans

   # Verify counts match backup metadata
   npm run db:verify-counts
   ```

8. **Update DNS Records**
   ```bash
   # Update CNAME to point to new load balancer
   # In Route 53 or DNS provider:
   # live-it-iconic.com -> [new-load-balancer-url]

   # Verify DNS propagation (may take up to 48 hours)
   dig live-it-iconic.com
   ```

9. **Run Smoke Tests**
   ```bash
   # Execute critical path tests
   npm run test:smoke

   # Test API endpoints
   npm run test:api

   # Test UI functionality
   npm run test:e2e
   ```

10. **Verify Services**
    ```bash
    # Check application health
    curl -s https://live-it-iconic.com/health | jq .

    # Verify database connectivity
    npm run db:health

    # Test payment processing (staging)
    npm run test:payments
    ```

11. **Post-Disaster Steps**
    - Verify all critical functions working
    - Test email notifications
    - Confirm backup resumption
    - Monitor system for 24 hours
    - Document lessons learned

---

### Scenario 3: Partial Service Failure

**Estimated Recovery Time:** 15-30 minutes

**Example: API Gateway Failure**

```bash
# 1. Identify affected service
npm run monitor:services

# 2. Check service logs
docker logs [service-container]

# 3. Restart service
pm2 restart live-it-iconic-api

# 4. Verify service recovered
npm run health-check:api

# 5. Monitor for stabilization
npm run monitor:api --duration=5m
```

---

## Backup Operations

### Creating an On-Demand Backup

```bash
# Create immediate backup
npm run backup:create

# Or using script directly
tsx scripts/backup/database-backup.ts

# Expected output:
# Created backup directory: /backups/2024-01-15T10-30-45-123Z
# ✓ Backed up 5234 records from products
# ✓ Backed up 12543 records from orders
# ✓ Backup completed successfully
```

### Listing Available Backups

```bash
# Via API
curl -X GET https://live-it-iconic.com/api/admin/backups \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Via CLI
npm run backup:list

# Expected output:
# Backup ID                        Size      Records    Date
# 2024-01-15T02-00-00-000Z         124.5MB   50,234     Jan 15, 2024
# 2024-01-14T02-00-00-000Z         123.2MB   49,987     Jan 14, 2024
```

### Validating a Backup

```bash
# Validate backup integrity
npm run backup:validate /backups/2024-01-15T02-00-00-000Z

# Expected output:
# ✓ Backup validation passed
# ✓ metadata.json: Valid
# ✓ products.json: 5234 records
# ✓ orders.json: 12543 records
# ✓ All tables accounted for
```

### Restoring from Backup

```bash
# Restore from specific backup
SKIP_RESTORE_CONFIRMATION=true \
tsx scripts/backup/database-restore.ts /backups/2024-01-15T02-00-00-000Z

# Expected output:
# Restoring backup from 2024-01-15T02-00-00-000Z
# ✓ Restored 5234 records to products
# ✓ Restored 12543 records to orders
# ✓ Restore completed
```

---

## Testing and Drills

### Monthly Disaster Recovery Drill

**Schedule:** First Friday of each month, 2:00 PM UTC

**Participants:**
- DevOps Team
- Database Administrator
- System Administrator
- Engineering Lead

**Checklist:**

- [ ] Launch new test environment
- [ ] Download latest backup from S3
- [ ] Perform complete restore
- [ ] Run full test suite
- [ ] Verify data integrity
- [ ] Test critical user journeys
- [ ] Document time taken
- [ ] Identify improvement areas
- [ ] Update runbooks if needed
- [ ] Report results to management

**Success Criteria:**
- All tests pass ✓
- RTO met (< 1 hour) ✓
- RPO acceptable (< 24 hours) ✓
- No data loss ✓
- All systems operational ✓

---

## Security Considerations

### Backup Encryption

All backups are encrypted using AES-256:

```bash
# Local backups
- Stored in /backups directory
- Filesystem encryption: LUKS
- File permissions: 600 (owner only)

# S3 backups
- Server-side encryption: AES-256
- Versioning: Enabled
- Access control: IAM policies
- Logging: CloudTrail enabled
```

### Access Control

**Who can access backups:**
- Database Administrator (DBAs)
- DevOps Engineers
- Engineering Lead
- Emergency Support Team

**Access method:**
- SSH to backup server (DBAs/DevOps)
- AWS IAM for S3 access
- All access logged and audited

**Create admin IAM policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::live-it-iconic-backups",
        "arn:aws:s3:::live-it-iconic-backups/*"
      ],
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": ["YOUR_OFFICE_IP"]
        }
      }
    }
  ]
}
```

---

## Monitoring and Alerting

### Backup Monitoring

**Dashboard URL:** `https://live-it-iconic.com/admin/backups`

**Metrics to Monitor:**
- Last successful backup time
- Backup duration
- Backup size trending
- S3 upload status
- Database record counts

### Alerts

**Critical (Page on-call):**
- Backup failed for > 24 hours
- S3 upload failed
- Backup size increased > 50%

**Warning (Slack notification):**
- Backup took > 1 hour
- Backup size increased > 25%
- Disk space < 20% remaining

**Setup alerts:**
```bash
# CloudWatch alarm for backup failure
aws cloudwatch put-metric-alarm \
  --alarm-name BackupFailure \
  --alarm-description "Alert if backup fails" \
  --metric-name BackupStatus \
  --namespace LiveItIconic \
  --statistic Average \
  --period 3600 \
  --evaluation-periods 1 \
  --threshold 0 \
  --comparison-operator LessThanOrEqualToThreshold \
  --alarm-actions arn:aws:sns:us-east-1:123456789:on-call
```

---

## Communication Plan

### During Disaster

**Immediate (0-30 min):**
1. Activate incident response team
2. Identify issue severity
3. Notify on-call engineers
4. Begin diagnostics

**Notification (30-60 min):**
1. Update status page
2. Send customer email notification
3. Inform customer success team
4. Begin recovery procedures

**Ongoing (hourly):**
1. Update status page every 30 minutes
2. Provide progress updates to stakeholders
3. Monitor system health metrics
4. Adjust ETA as needed

### Post-Recovery

**Recovery Complete (0-1 hour):**
1. Verify all systems operational
2. Run smoke tests
3. Brief stakeholders on completion
4. Begin detailed investigation

**Post-Incident (24-48 hours):**
1. Complete incident report
2. Hold post-mortem meeting
3. Identify improvement opportunities
4. Update runbooks and procedures
5. Communicate lessons learned

---

## Runbook Checklist

### Quick Recovery Reference

**Database Corruption:**
```
[ ] Identify issue and confirm with logs
[ ] Notify on-call team
[ ] Stop application
[ ] List and validate recent backup
[ ] Restore from backup
[ ] Verify restored data
[ ] Run integrity checks
[ ] Restart services
[ ] Monitor system
```

**Complete Failure:**
```
[ ] Activate DR team
[ ] Provision new infrastructure
[ ] Deploy application code
[ ] Download backup from S3
[ ] Configure environment variables
[ ] Restore database
[ ] Run full test suite
[ ] Update DNS/routing
[ ] Verify critical paths
[ ] Monitor for 24 hours
```

---

## Contact Information

**On-Call Engineer:** (Check Slack #on-call channel)

**Database Administrator:** [Contact Info]

**AWS Account Owner:** [Contact Info]

**Incident Commander:** [Contact Info]

**Customer Support Lead:** [Contact Info]

---

## References

- [AWS RDS Backup Guide](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.html)
- [Supabase Backup Documentation](https://supabase.com/docs/guides/backups)
- [AWS Disaster Recovery Strategy](https://aws.amazon.com/blogs/disaster-recovery/)
- Live It Iconic Backup Scripts: `scripts/backup/`

---

## Document History

| Date | Version | Changes |
|---|---|---|
| 2024-01-15 | 1.0 | Initial DR plan |
| | 1.1 | Added Scenario 3 |

---

**Last Updated:** 2024-01-15
**Next Review:** 2024-04-15
