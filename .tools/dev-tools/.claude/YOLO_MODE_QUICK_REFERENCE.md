# YOLO Mode - Quick Reference Card

**Status:** ✅ Active | **Commands:** 205 Allowed, 19 Blocked | **Updated:** 2025-11-22

---

## 🚀 Quick Stats

```
✅ Allowed Commands: 205
🚫 Blocked Commands: 19
❓ Ask Commands: 0
📁 Config: .claude/settings.local.json
```

---

## 📋 Most Common Commands (Auto-Approved)

### Git Workflow
```bash
git status
git add .
git commit -m "message"
git push
git pull
git branch
git checkout -b feature
git log --oneline
git diff
```

### Node.js Development
```bash
npm install
npm run dev
npm test
npm run build
pnpm install
pnpm dev
yarn install
npx turbo build
```

### File Operations
```bash
# Bash
ls -la
cat file.txt
mkdir -p dir/subdir
cp -r src dest
mv old new
find . -name "*.js"
grep -r "pattern" .

# PowerShell
Get-ChildItem -Recurse
Get-Content file.txt
New-Item -ItemType Directory -Path dir
Copy-Item -Recurse src dest
Move-Item old new
Select-String -Pattern "text" -Path *.js
```

### Python Development
```bash
python -m venv venv
pip install -r requirements.txt
pytest
poetry install
poetry run python script.py
```

### Docker & Kubernetes
```bash
docker build -t image:tag .
docker-compose up -d
docker ps
kubectl get pods
kubectl apply -f deployment.yaml
```

### Build Tools
```bash
make build
turbo run build
cargo build --release
go build
dotnet build
```

---

## 🚫 Blocked Commands (Will Fail)

```bash
# NEVER ALLOWED
rm -rf /
rm -rf /*
dd if=/dev/zero of=/dev/sda
shutdown now
reboot
Format-Volume C:
Stop-Computer
```

---

## 🔧 Common Patterns

### Pattern Syntax
```
Bash(command:*)          # Any arguments
PowerShell(Cmdlet:*)     # Any arguments
Bash(exact command)      # Exact match only
```

### Examples
```json
"Bash(git:*)"                    → All git commands
"PowerShell(Get-*:*)"            → All Get- cmdlets
"Bash(npm install)"              → Only npm install
"Bash(git status:*)"             → git status with any args
```

---

## 📊 Command Categories

### File System (40+ commands)
- **Bash:** ls, cat, mkdir, cp, mv, rm, find, grep, sed, awk
- **PowerShell:** Get-ChildItem, Get-Content, New-Item, Copy-Item, Move-Item

### Version Control (20+ commands)
- git status, add, commit, push, pull, branch, checkout, merge, rebase, stash

### Package Managers (30+ commands)
- npm, pnpm, yarn, pip, poetry, cargo, go, dotnet, mvn, gradle

### Development Tools (50+ commands)
- node, python, docker, kubectl, make, turbo, eslint, prettier, tsc, jest

### System Info (20+ commands)
- which, whereis, env, Get-Process, Get-Service, Get-Command

### Text Processing (25+ commands)
- grep, sed, awk, jq, yq, Select-String, Where-Object, Sort-Object

### Archives (10+ commands)
- tar, gzip, zip, unzip, Compress-Archive, Expand-Archive

### Network (10+ commands)
- curl, wget, Invoke-WebRequest, Invoke-RestMethod

---

## 🎯 Usage Tips

### 1. Test Before Committing
```bash
# These run without prompts
npm test
npm run lint
git status
```

### 2. Chain Commands Safely
```bash
# All auto-approved
npm install && npm test && npm run build
```

### 3. Use Wildcards
```bash
# Works with any arguments
git log --graph --oneline --all
npm run build -- --watch
```

### 4. PowerShell Pipelines
```powershell
# All cmdlets auto-approved
Get-ChildItem -Recurse | Where-Object {$_.Extension -eq ".js"} | Measure-Object
```

---

## 🔄 Quick Actions

### Enable YOLO Mode
Already enabled! Config at `.claude/settings.local.json`

### Disable YOLO Mode
```bash
# Backup first
cp .claude/settings.local.json .claude/settings.local.json.backup

# Switch to ask mode
echo '{"permissions":{"allow":[],"deny":[],"ask":["*"]}}' > .claude/settings.local.json
```

### Restore YOLO Mode
```bash
cp .claude/settings.local.json.backup .claude/settings.local.json
```

### Verify Configuration
```bash
cat .claude/settings.local.json | jq '.permissions.allow | length'
```

---

## 📝 Adding New Commands

Edit `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(your-new-command:*)",
      "PowerShell(Your-New-Cmdlet:*)"
    ]
  }
}
```

---

## 🛡️ Safety Features

1. **Explicit Deny List:** Dangerous commands blocked
2. **No Wildcards in Deny:** Specific dangerous patterns only
3. **Path Restrictions:** System paths protected
4. **Audit Logs:** All commands logged in `.automation/logs/`
5. **Backup Config:** Keep `.backup` version

---

## 📚 Full Documentation

- **Detailed Guide:** `.claude/YOLO_MODE_CONFIGURATION.md`
- **Automation Framework:** `.automation/README.md`
- **Security Policy:** `SECURITY.md`
- **Governance:** `.governance/governance-config.json`

---

## ✅ Verification

Run these to verify YOLO mode is working:

```bash
# Should execute without prompts
ls -la
git status
npm --version
node --version
docker --version

# PowerShell
Get-ChildItem
Get-Process
Get-Command git
```

---

## 🆘 Troubleshooting

### Command Still Prompts
1. Check if pattern exists in allow list
2. Verify JSON syntax is valid
3. Restart Claude Code
4. Check for typos in command pattern

### JSON Syntax Error
```bash
# Validate JSON
cat .claude/settings.local.json | jq .
```

### Restore from Backup
```bash
cp .claude/settings.local.json.backup .claude/settings.local.json
```

---

## 📞 Quick Help

```bash
# View current config
cat .claude/settings.local.json

# Count allowed commands
cat .claude/settings.local.json | jq '.permissions.allow | length'

# Search for specific command
cat .claude/settings.local.json | jq '.permissions.allow[] | select(contains("git"))'

# Validate JSON
cat .claude/settings.local.json | jq . > /dev/null && echo "Valid" || echo "Invalid"
```

---

**Remember:** YOLO mode is for development. Use caution in production environments!

