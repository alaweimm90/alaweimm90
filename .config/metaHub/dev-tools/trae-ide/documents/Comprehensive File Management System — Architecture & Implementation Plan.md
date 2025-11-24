## Objectives

Implement an automated, auditable file management system covering retention policies, scanning, cleanup, safety/approvals, backups, monitoring and reporting. Integrates with existing CI, runs cross‑platform, and maintains rollback via PRs and archives.

## Architecture

- Components:
  - Scanner: recursive inventory builder (hashes, metadata, owner, access stats)
  - Policy Engine: retention rules + exceptions
  - Cleanup Orchestrator: daily/weekly/monthly tasks (delete/rotate/archive/wipe)
  - Safety/Approvals: active file exclusion, lock detection, reference preservation, approval thresholds, quarantine
  - Backup & Verification: snapshot to backup location + integrity checks
  - Audit Logger: append‑only JSONL logs with ISO timestamps
  - Reporting: dashboards (space reclaimed, counts, health), alerts
- Storage:
  - SQLite inventory database: `data/file_inventory.sqlite` (tables: `files`, `access_stats`, `operations`, `references`)
  - Config: `${config_format}` at `${config_path}` (YAML/JSON schema)
  - Archives: `archives/{backups,logs,quarantine}`; long‑term storage `${archive_location}`

## Configuration

- Retention:
  - `document_retention`, `log_retention`, `temp_retention`, `report_retention` (days)
  - Exceptions: tags `Permanent`, `Do Not Delete` in metadata or override list
- Scanning thresholds:
  - `age_threshold` (days), duplicate detection (MD5), large thresholds (`max_log_size` MB, `max_doc_size` MB)
- Schedules:
  - Daily: temp deletion after `temp_grace_period` hours; log rotation above `rotation_threshold` MB
  - Weekly: compress with `${compression_format}`, move to `${archive_location}`, encrypt `${encryption_standard}`
  - Monthly: secure wipe `${wipe_method}` passes; orphaned file scan
- Safety:
  - Active window `active_threshold` days; `lock_detection_method`; preserve `${reference_sources}`
  - Backup confirmation `${backup_location}` + `${verification_method}`
  - Approvals: `${approval_level}` required for files > `${approval_threshold}` MB; quarantine rules
- Implementation:
  - Language: `${programming_language}` with `${framework}`
  - OS: `${supported_os_list}`; tests `${test_coverage}`%; docs `${documentation_format}`
  - Test env: `${test_environment}`

## CI/Workflows (GitHub Actions)

- `file-inventory.yml` (daily): run scanner, update SQLite, export `inventory.json`
- `file-cleanup-daily.yml`: temp deletion, log rotation, audit logging, open PR with changes
- `file-archive-weekly.yml`: compress, move, encrypt, audit, PR
- `file-deep-clean-monthly.yml`: secure wipe (simulate in test env), orphaned scan, audit, PR
- `file-approvals.yml`: block PRs that remove files > threshold without `${approval_level}` label; quarantine job
- `file-backup-verify.yml`: backup to `${backup_location}`, verify `${verification_method}`, record audit; gate deletions
- `file-policy-compliance.yml`: validate config schema, retention mapping, exceptions; fail on policy violations
- `file-dashboard.yml`: aggregate reclaimed space, preserved/removed counts, system metrics; push historical snapshots via auto‑PR
- `file-alerts.yml`: open issues for failed cleanup, storage threshold breaches, policy violations

## Scripts/Services

- Scanner (cross‑platform):
  - Walk directories, detect temp patterns (`*.tmp`, `~*`, `.bak`), compute MD5, classify type (doc/log/temp/report)
  - Populate SQLite: path, size, mtime/atime/ctime, type, owner (stat or mapping), hash, access count (updated on scans)
- Policy Engine:
  - Evaluate retention per category; apply exceptions; compute candidate sets for each schedule
- Cleanup:
  - Daily: delete expired temps after `temp_grace_period`; rotate logs (`rotation_threshold`) preserving tail; audit each action
  - Weekly: compress docs (`${compression_format}`), move to `${archive_location}`, encrypt sensitive (AES‑256‑GCM or `${encryption_standard}`)
  - Monthly: secure delete (`${wipe_method}` passes; simulated in `${test_environment}`), orphaned scan (files not referenced by inventory/repo)
- Safety & Approvals:
  - Exclude files accessed within `active_threshold` days
  - Lock detection via `${lock_detection_method}` (exclusive open test / OS locks)
  - Preserve references discovered via `${reference_sources}` (repo index, config allowlists)
  - Approvals: PR label `${approval_level}` required; quarantine large deletions to `archives/quarantine` pending approval
- Backups:
  - Ensure backup in `${backup_location}` before deletion; verify integrity via `${verification_method}` (checksum) and record in audit
- Audit:
  - JSONL entries: timestamp, action, file metadata (pre‑deletion), actor (`${auth_method}`), reason (auto/manual); retention `${audit_retention}` years

## Quality Gates

- Pre‑commit: file naming conventions, doc validation, link checks
- CI: schema validation, inventory integrity, policy mapping, approvals, backup verification, audit completeness

## Monitoring & Reporting

- Dashboards: compliance %, reclaimed space by type, preserved/removed counts, duplicates/temps/outdated, storage usage
- Alerts: failed jobs, threshold breaches, violations (issues with labels)

## Tests

- Unit tests (≥ `${test_coverage}`%): policy engine, scanner classification, lock detection, backup validation, audit logging
- Integration (in `${test_environment}`): daily/weekly/monthly pipelines with mocks (encryption/wipe simulated), quarantine/approvals flow
- Regression: inventory DB migrations, rollback via PRs, orphan detection

## Deliverables

- Config schema and default config at `${config_path}`
- Scripts/services with CLI entry points
- SQLite schema migration files
- CI workflows and dashboards
- Documentation in `${documentation_format}` with SOPs/runbooks

## Verification & Rollout

- Dry‑run mode first (no destructive ops) in `${test_environment}`
- Enable gates sequentially: inventory → cleanup (temp/log) → archive → deep clean
- Require approvals for large deletions; enforce backup verification gate

Approve to proceed; I will scaffold config/schema, scripts, SQLite setup, CI workflows and tests, integrate approval/backup gates, and deliver dashboards and alerts with full audit trails.
