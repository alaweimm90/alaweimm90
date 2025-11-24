# Claude Code Configuration

**Directory:** `.claude/`  
**Purpose:** Claude Code IDE integration and automation settings  
**Last Updated:** 2025-11-22

---

## 📁 Files in This Directory

### `settings.local.json`
**Primary configuration file for Claude Code permissions and YOLO mode.**

- **Purpose:** Define auto-approved commands for bash and PowerShell
- **Status:** ✅ Active
- **Commands:** 205 allowed, 19 blocked
- **Mode:** YOLO (auto-approve enabled)

### `YOLO_MODE_CONFIGURATION.md`
**Comprehensive documentation for YOLO mode setup and usage.**

- **Purpose:** Detailed guide on configuration, patterns, and security
- **Audience:** Developers and DevOps engineers
- **Length:** ~300 lines
- **Includes:** Full command reference, safety guidelines, troubleshooting

### `YOLO_MODE_QUICK_REFERENCE.md`
**Quick reference card for common commands and patterns.**

- **Purpose:** Fast lookup for daily development tasks
- **Audience:** All developers
- **Length:** ~250 lines
- **Includes:** Common commands, quick actions, verification steps

---

## 🎯 What is YOLO Mode?

YOLO (You Only Live Once) mode enables **automatic approval** of pre-defined safe commands, eliminating repetitive permission prompts during development.

### Benefits
- ✅ Faster development workflow
- ✅ No interruptions for common commands
- ✅ Maintains safety with explicit deny list
- ✅ Comprehensive logging for audit

### Safety Features
- 🛡️ Explicit deny list for dangerous commands
- 🛡️ System path protection
- 🛡️ Audit logging enabled
- 🛡️ Easy rollback to safe mode

---

## 🚀 Quick Start

### 1. Verify Configuration
```bash
cat .claude/settings.local.json
```

### 2. Test Auto-Approval
```bash
# These should run without prompts
git status
npm install
ls -la
```

### 3. View Statistics
```bash
# PowerShell
$config = Get-Content .claude\settings.local.json | ConvertFrom-Json
Write-Host "Allowed: $($config.permissions.allow.Count)"
Write-Host "Blocked: $($config.permissions.deny.Count)"
```

---

## 📊 Current Configuration

```
✅ Allowed Commands: 205
🚫 Blocked Commands: 19
❓ Ask Commands: 0
📝 Mode: YOLO (Auto-Approve)
```

### Command Categories

| Category | Count | Examples |
|----------|-------|----------|
| **File Operations** | 40+ | ls, cat, mkdir, cp, mv, grep, sed |
| **Git Commands** | 20+ | status, add, commit, push, pull, branch |
| **Package Managers** | 30+ | npm, pnpm, yarn, pip, poetry, cargo |
| **Development Tools** | 50+ | node, python, docker, kubectl, make |
| **System Info** | 20+ | which, env, Get-Process, Get-Service |
| **Text Processing** | 25+ | grep, sed, awk, jq, Select-String |
| **Archives** | 10+ | tar, gzip, zip, Compress-Archive |
| **Network** | 10+ | curl, wget, Invoke-WebRequest |

---

## 🔧 Common Commands (Auto-Approved)

### Git Workflow
```bash
git status
git add .
git commit -m "message"
git push origin main
git pull
git checkout -b feature
```

### Node.js Development
```bash
npm install
npm run dev
npm test
npm run build
pnpm install
yarn install
npx turbo build
```

### Python Development
```bash
pip install -r requirements.txt
pytest
poetry install
python script.py
```

### Docker & Kubernetes
```bash
docker build -t image:tag .
docker-compose up -d
kubectl get pods
kubectl apply -f config.yaml
```

---

## 🚫 Blocked Commands

These commands are **explicitly denied** for safety:

```bash
# System Destruction
rm -rf /
rm -rf /*
dd if=/dev/zero of=/dev/sda

# System Control
shutdown
reboot
halt
systemctl poweroff

# Disk Operations
mkfs
fdisk
Format-Volume
Clear-Disk

# PowerShell System
Stop-Computer
Restart-Computer
Remove-Item -Path C:\Windows
```

---

## 📝 Configuration Syntax

### Pattern Format
```json
{
  "permissions": {
    "allow": [
      "Bash(command:*)",           // Any arguments
      "PowerShell(Cmdlet:*)",      // Any arguments
      "Bash(exact command)"        // Exact match only
    ],
    "deny": [
      "Bash(dangerous:*)"
    ],
    "ask": []
  }
}
```

### Examples
```json
"Bash(git:*)"                    → All git commands
"PowerShell(Get-*:*)"            → All Get- cmdlets
"Bash(npm install)"              → Only npm install (no args)
"Bash(git status:*)"             → git status with any args
```

---

## 🔄 Managing YOLO Mode

### Disable YOLO Mode
```bash
# Backup current config
cp .claude/settings.local.json .claude/settings.local.json.backup

# Switch to ask mode (prompt for everything)
cat > .claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": [],
    "deny": [],
    "ask": ["*"]
  }
}
EOF
```

### Re-enable YOLO Mode
```bash
# Restore from backup
cp .claude/settings.local.json.backup .claude/settings.local.json
```

### Add New Commands
Edit `.claude/settings.local.json` and add to the `allow` array:
```json
"Bash(your-command:*)",
"PowerShell(Your-Cmdlet:*)"
```

---

## 🛡️ Security Best Practices

1. **Review Regularly:** Audit the allow list monthly
2. **Principle of Least Privilege:** Only allow necessary commands
3. **Test Safely:** Test new patterns in development first
4. **Keep Backups:** Maintain `.backup` versions
5. **Monitor Logs:** Check `.automation/logs/` for anomalies
6. **Version Control:** Commit config changes with clear messages

---

## 📚 Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| `YOLO_MODE_CONFIGURATION.md` | Complete guide | All developers |
| `YOLO_MODE_QUICK_REFERENCE.md` | Quick lookup | Daily use |
| `README.md` (this file) | Overview | New users |

---

## 🔗 Related Resources

### Internal Documentation
- [Automation Framework](../.automation/README.md)
- [Security Guidelines](../SECURITY.md)
- [Workflow Automation](../WORKFLOW_AUTOMATION_QUICK_START.md)
- [Governance Config](../.governance/governance-config.json)

### External Resources
- [Claude Code Documentation](https://docs.anthropic.com/claude/docs)
- [Git Documentation](https://git-scm.com/doc)
- [PowerShell Documentation](https://docs.microsoft.com/powershell)

---

## ✅ Verification Checklist

Before using YOLO mode, verify:

- [ ] Configuration file exists at `.claude/settings.local.json`
- [ ] JSON syntax is valid (test with `jq`)
- [ ] Common commands work without prompts
- [ ] Dangerous commands are blocked
- [ ] Backup configuration exists
- [ ] Team members are aware of YOLO mode
- [ ] Audit logging is enabled

---

## 🆘 Troubleshooting

### Command Still Prompts for Approval
1. Check if pattern exists in allow list
2. Verify JSON syntax: `cat .claude/settings.local.json | jq .`
3. Restart Claude Code
4. Check for typos in command pattern
5. Ensure no conflicting deny patterns

### JSON Syntax Error
```bash
# Validate JSON
cat .claude/settings.local.json | jq . > /dev/null

# If error, restore backup
cp .claude/settings.local.json.backup .claude/settings.local.json
```

### Commands Failing Unexpectedly
1. Check deny list for conflicts
2. Review `.automation/logs/automation.log`
3. Test with ask mode first
4. Verify command syntax

---

## 📞 Support

For issues or questions:

1. **Check Logs:** `.automation/logs/automation.log`
2. **Review Docs:** Read `YOLO_MODE_CONFIGURATION.md`
3. **Test Syntax:** Validate JSON with `jq`
4. **Restore Backup:** Use `.backup` files if needed

---

## 📈 Statistics

**Last Updated:** 2025-11-22

```
Total Patterns: 224
├── Allowed: 205 (91.5%)
├── Blocked: 19 (8.5%)
└── Ask: 0 (0%)

Coverage:
├── Bash Commands: ~140
├── PowerShell Commands: ~65
└── Safety Blocks: 19
```

---

**Note:** This configuration is optimized for development environments. Use with appropriate caution in production settings.

