# 💾 PHASE 9: BACKUP & DISASTER RECOVERY - CASCADE HANDOFF

## Mission
Implement automated backup and disaster recovery procedures. AI-accelerated: 15-25 minutes.

## Context
- Multiple production databases (Supabase for REPZ, LiveItIconic)
- Critical config files need backup
- No current automated backup system

---

## Tasks (Execute in Order)

### 1. Create Backup Configuration (5 min)
Create `.config/backup/config.yaml`:
```yaml
version: "1.0"
backup:
  enabled: true
  
  schedules:
    daily:
      time: "02:00"
      retention_days: 7
    weekly:
      day: "sunday"
      time: "03:00"
      retention_days: 30
    monthly:
      day: 1
      time: "04:00"
      retention_days: 365
  
  targets:
    configs:
      paths:
        - ".config/"
        - ".metaHub/policies/"
        - "package.json"
        - "tsconfig.json"
      destination: ".backups/configs/"
    
    databases:
      # Handled by Supabase automatic backups
      supabase_projects:
        - "repz-prod"
        - "liveiticonic-prod"
      note: "Supabase handles DB backups automatically"
    
    secrets:
      # Never backup actual secrets - just track their existence
      audit_only: true
      
  exclusions:
    - "node_modules/"
    - ".git/"
    - "dist/"
    - ".next/"
    - "__pycache__/"
    - "*.log"
```

### 2. Create Backup Script (10 min)
Create `tools/backup/backup.ts`:
```typescript
#!/usr/bin/env tsx
/**
 * Config backup utility
 * Usage: npm run backup [configs|full]
 */
import { cpSync, mkdirSync, existsSync, writeFileSync } from 'fs';
import { join } from 'path';

const BACKUP_DIR = '.backups';
const CONFIGS_TO_BACKUP = [
  '.config',
  '.metaHub/policies',
  'package.json',
  'tsconfig.json',
  'turbo.json',
  '.pre-commit-config.yaml'
];

function getTimestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

function backupConfigs() {
  const timestamp = getTimestamp();
  const dest = join(BACKUP_DIR, `configs-${timestamp}`);
  
  mkdirSync(dest, { recursive: true });
  
  console.log(`💾 Backing up configs to ${dest}\n`);
  
  for (const path of CONFIGS_TO_BACKUP) {
    try {
      if (existsSync(path)) {
        const destPath = join(dest, path);
        mkdirSync(join(dest, path.split('/').slice(0, -1).join('/')), { recursive: true });
        cpSync(path, destPath, { recursive: true });
        console.log(`  ✅ ${path}`);
      }
    } catch (err) {
      console.log(`  ❌ ${path}: ${err}`);
    }
  }
  
  // Write manifest
  const manifest = {
    timestamp,
    files: CONFIGS_TO_BACKUP,
    created_by: 'backup.ts'
  };
  writeFileSync(join(dest, 'manifest.json'), JSON.stringify(manifest, null, 2));
  
  console.log(`\n✅ Backup complete: ${dest}`);
}

function listBackups() {
  if (!existsSync(BACKUP_DIR)) {
    console.log('No backups found');
    return;
  }
  
  const { readdirSync } = require('fs');
  const backups = readdirSync(BACKUP_DIR);
  console.log('📦 Available Backups:\n');
  backups.forEach((b: string) => console.log(`  - ${b}`));
}

const [,, cmd] = process.argv;
switch (cmd) {
  case 'configs':
  case 'full':
    backupConfigs();
    break;
  case 'list':
    listBackups();
    break;
  default:
    console.log('Usage: npm run backup <configs|list>');
}
```

### 3. Create DR Runbook (5 min)
Create `docs/DR-RUNBOOK.md`:
```markdown
# Disaster Recovery Runbook

## Quick Recovery Steps

### 1. Repository Loss
```bash
git clone https://github.com/alawein/meta-governance.git
npm install
```

### 2. Config Restoration
```bash
npm run backup list          # Find latest backup
cp -r .backups/configs-XXXX/.config .config
```

### 3. Database Recovery (Supabase)
- Login to Supabase Dashboard
- Navigate to Project > Database > Backups
- Select point-in-time recovery

### 4. Secrets Recovery
- Retrieve from password manager
- Update `.env` files
- Rotate if compromised

## Contact
- Primary: [your-email]
- Supabase Support: support@supabase.io
```

### 4. Add npm Scripts (2 min)
Add to `package.json`:
```json
"backup": "tsx tools/backup/backup.ts",
"backup:configs": "tsx tools/backup/backup.ts configs",
"backup:list": "tsx tools/backup/backup.ts list"
```

### 5. Update .gitignore (2 min)
Add to `.gitignore`:
```
.backups/
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `.config/backup/config.yaml` | Create |
| `tools/backup/backup.ts` | Create |
| `docs/DR-RUNBOOK.md` | Create |
| `package.json` | Add scripts |
| `.gitignore` | Add .backups/ |

---

## Success Criteria

- [ ] `npm run backup configs` creates timestamped backup
- [ ] `npm run backup list` shows available backups
- [ ] DR runbook documents recovery procedures
- [ ] .backups/ excluded from git

---

## Commit
`feat(backup): Complete Phase 9 backup & disaster recovery`

