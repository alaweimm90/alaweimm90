# Disaster Recovery Runbook

## Quick Recovery Steps

### 1. Repository Loss
```bash
git clone https://github.com/alaweimm90/alaweimm90.git
cd alaweimm90
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
- Primary: meshal@alaweintechnologies.com
- Supabase Support: support@supabase.io

## Backup Testing
Test monthly:
```bash
npm run backup configs    # Create backup
# Test restoration process
npm run health:services   # Verify systems
```
