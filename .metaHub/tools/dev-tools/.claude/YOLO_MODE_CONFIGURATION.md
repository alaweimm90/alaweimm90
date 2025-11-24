# Claude Code - YOLO Mode Configuration

**Last Updated:** 2025-11-22  
**Status:** ✅ Active  
**Configuration File:** `.claude/settings.local.json`

---

## 📋 Overview

This document describes the YOLO (You Only Live Once) mode configuration for Claude Code, which enables auto-approval of common bash and PowerShell commands to streamline development workflows.

## ⚙️ Configuration Location

```
.claude/settings.local.json
```

## 🎯 What's Enabled

### Bash Commands (200+ patterns)

#### File Operations
- `ls`, `pwd`, `cat`, `echo`, `mkdir`, `touch`
- `cp`, `mv`, `rm` (with safety restrictions)
- `find`, `grep`, `sed`, `awk`
- `wc`, `head`, `tail`, `sort`, `uniq`, `diff`

#### System Information
- `which`, `whereis`, `file`, `stat`
- `du`, `df`, `tree`
- `env`, `export`, `alias`, `history`

#### Archive & Compression
- `tar`, `gzip`, `gunzip`, `zip`, `unzip`

#### Network
- `curl`, `wget`

#### Git (Full Suite)
- `git status`, `git add`, `git commit`, `git push`, `git pull`
- `git branch`, `git checkout`, `git log`, `git diff`
- `git merge`, `git clone`, `git fetch`, `git remote`
- `git stash`, `git tag`, `git reset`, `git rebase`

#### Package Managers
- **Node.js:** `npm`, `pnpm`, `yarn`, `npx`
- **Python:** `pip`, `pip3`, `poetry`, `pytest`
- **Rust:** `cargo`, `rustc`
- **Go:** `go`
- **Java:** `mvn`, `gradle`
- **C#/.NET:** `dotnet`

#### Development Tools
- `node`, `python`, `python3`
- `docker`, `docker-compose`, `kubectl`
- `make`, `bash`, `sh`, `zsh`
- `turbo`, `eslint`, `prettier`, `tsc`
- `jest`, `vitest`

#### Text Editors
- `vim`, `nano`, `code`, `less`, `more`

#### JSON/YAML Processing
- `jq`, `yq`

#### Utilities
- `xargs`, `tee`, `clear`, `exit`

### PowerShell Commands (100+ patterns)

#### File System Operations
- `Get-ChildItem`, `Get-Content`, `Set-Content`, `Add-Content`, `Clear-Content`
- `New-Item`, `Remove-Item`, `Copy-Item`, `Move-Item`, `Rename-Item`
- `Test-Path`, `Get-Item`, `Get-ItemProperty`, `Set-ItemProperty`

#### Navigation
- `Get-Location`, `Set-Location`, `Push-Location`, `Pop-Location`
- `Join-Path`, `Split-Path`, `Resolve-Path`

#### Text Processing
- `Select-String`, `Where-Object`, `ForEach-Object`
- `Sort-Object`, `Group-Object`, `Measure-Object`, `Select-Object`

#### Output & Formatting
- `Write-Host`, `Write-Output`, `Write-Error`, `Write-Warning`, `Write-Verbose`
- `Out-File`, `Out-String`, `Out-Null`
- `Format-Table`, `Format-List`

#### Process Management
- `Get-Process`, `Stop-Process`, `Start-Process`
- `Get-Service`, `Start-Service`, `Stop-Service`, `Restart-Service`

#### System Information
- `Get-Command`, `Get-Help`, `Get-Member`
- `Get-Variable`, `Set-Variable`, `Clear-Variable`
- `Get-Alias`, `Set-Alias`, `New-Alias`
- `Get-Date`, `Get-Random`

#### Network & Web
- `Invoke-WebRequest`, `Invoke-RestMethod`

#### Data Conversion
- `ConvertTo-Json`, `ConvertFrom-Json`
- `ConvertTo-Csv`, `ConvertFrom-Csv`
- `Export-Csv`, `Import-Csv`

#### Comparison & Utilities
- `Compare-Object`, `Get-FileHash`
- `Compress-Archive`, `Expand-Archive`

#### Development Tools (PowerShell)
- `git`, `npm`, `pnpm`, `yarn`, `node`, `npx`
- `python`, `pip`, `docker`, `docker-compose`, `kubectl`
- `dotnet`, `code`, `turbo`, `eslint`, `prettier`
- `tsc`, `jest`, `cargo`, `go`

---

## 🚫 Safety Restrictions (Deny List)

The following dangerous commands are **explicitly blocked**:

### Bash Blocked Commands
```bash
rm -rf /          # Root deletion
rm -rf /*         # Root wildcard deletion
dd                # Disk destroyer
mkfs              # Format filesystem
fdisk             # Partition editor
shutdown          # System shutdown
reboot            # System reboot
halt              # System halt
init              # Init system control
systemctl poweroff
systemctl reboot
```

### PowerShell Blocked Commands
```powershell
Remove-Item -Path C:\              # System drive deletion
Remove-Item -Path C:\Windows       # Windows folder deletion
Remove-Item -Path C:\Program Files # Program Files deletion
Format-Volume                      # Format disk
Clear-Disk                         # Clear disk
Stop-Computer                      # Shutdown
Restart-Computer                   # Restart
Remove-Computer                    # Remove from domain
```

---

## 🔧 How to Use

### 1. Verify Configuration

```bash
cat .claude/settings.local.json
```

### 2. Test a Command

Try running any allowed command - it should execute without prompting:

```bash
# Bash
ls -la
git status
npm install

# PowerShell
Get-ChildItem
Get-Process
npm test
```

### 3. Modify Configuration

Edit `.claude/settings.local.json` to add/remove commands:

```json
{
  "permissions": {
    "allow": [
      "Bash(your-command:*)",
      "PowerShell(Your-Command:*)"
    ],
    "deny": [
      "Bash(dangerous-command:*)"
    ],
    "ask": []
  }
}
```

---

## 📝 Pattern Syntax

### Wildcards
- `*` - Matches any arguments
- `Bash(git:*)` - Allows all git commands
- `PowerShell(Get-*:*)` - Allows all Get- cmdlets

### Specific Commands
- `Bash(npm install)` - Only allows exact command
- `PowerShell(Get-ChildItem:*)` - Allows with any arguments

### Path-Specific
- `Bash(git -C "C:\\Users\\mesha\\Desktop\\GitHub" status)` - Specific path

---

## 🛡️ Security Best Practices

1. **Review Regularly:** Audit the allow list monthly
2. **Principle of Least Privilege:** Only allow what you need
3. **Test in Safe Environment:** Test new patterns in non-production
4. **Backup Configuration:** Keep a backup of working config
5. **Monitor Logs:** Check `.automation/logs/` for unusual activity

---

## 🔄 Reverting to Safe Mode

If you need to disable YOLO mode:

```bash
# Backup current config
cp .claude/settings.local.json .claude/settings.local.json.backup

# Revert to ask mode
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

---

## 📊 Statistics

- **Total Bash Patterns:** ~200
- **Total PowerShell Patterns:** ~100
- **Blocked Patterns:** 18
- **Coverage:** Common development workflows

---

## 🔗 Related Documentation

- [Automation Framework](./.automation/README.md)
- [Security Guidelines](../SECURITY.md)
- [Development Workflow](../WORKFLOW_AUTOMATION_QUICK_START.md)
- [Governance Config](../.governance/governance-config.json)

---

## 📞 Support

If you encounter issues:

1. Check logs: `.automation/logs/automation.log`
2. Verify syntax in `settings.local.json`
3. Test with `ask` mode first
4. Review deny list for conflicts

---

## ✅ Verification Checklist

- [ ] Configuration file exists at `.claude/settings.local.json`
- [ ] JSON syntax is valid
- [ ] Common commands work without prompts
- [ ] Dangerous commands are blocked
- [ ] Backup configuration exists
- [ ] Team members are aware of YOLO mode

---

**Note:** This configuration is designed for development environments. Use with caution in production settings.

