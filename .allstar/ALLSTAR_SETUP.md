# OpenSSF Allstar Setup Guide

Allstar is a GitHub App that continuously monitors and enforces security policies across your repositories.

## What is Allstar?

Allstar is part of the OpenSSF (Open Source Security Foundation) and provides:

- **Continuous security monitoring** - Always-on policy enforcement
- **Auto-remediation** - Can automatically fix certain security issues
- **Issue creation** - Creates GitHub issues for policy violations
- **Multi-policy support** - Branch protection, binary artifacts, dangerous workflows, etc.

## Installation

### Option 1: Install OpenSSF Allstar GitHub App (Recommended)

1. **Visit the Allstar GitHub App:**
   - Go to: https://github.com/apps/allstar-app
   - Click "Install" or "Configure"

2. **Select repository:**
   - Choose "Only select repositories"
   - Select: `alaweimm90/alaweimm90`
   - Grant required permissions

3. **Verify installation:**
   - Allstar will automatically read `.allstar/` configuration
   - Check for created issues at: https://github.com/alaweimm90/alaweimm90/issues?q=label:allstar

### Option 2: Self-Host Allstar

If you prefer to self-host:

1. **Clone Allstar:**

   ```bash
   git clone https://github.com/ossf/allstar.git
   cd allstar
   ```

2. **Create GitHub App:**
   - Go to: https://github.com/settings/apps/new
   - Set webhook URL to your server
   - Grant permissions:
     - Repository permissions:
       - Administration: Read & write
       - Checks: Read & write
       - Contents: Read & write
       - Issues: Read & write
       - Pull requests: Read & write
       - Metadata: Read

3. **Configure Allstar:**

   ```yaml
   # config/config.yaml
   github:
     app_id: YOUR_APP_ID
     installation_id: YOUR_INSTALLATION_ID
     private_key_path: /path/to/private-key.pem

   operator:
     action: issue # or: log, fix
     org_config: .allstar/allstar.yaml
   ```

4. **Deploy:**

   ```bash
   # Using Docker
   docker build -t allstar .
   docker run -p 8080:8080 -v /path/to/config:/config allstar

   # Or use provided Kubernetes manifests
   kubectl apply -f deploy/
   ```

## Configuration

Our Allstar configuration is located in `.allstar/` directory:

### Main Configuration: `.allstar/allstar.yaml`

```yaml
optConfig:
  optInRepos:
    - alaweimm90 # Enabled for this repo

action: issue # Create GitHub issues for violations

policies:
  branch_protection: enabled
  binary_artifacts: enabled
  outside: enabled
  security: enabled
  dangerous_workflow: enabled
```

### Policy-Specific Configs

- **`.allstar/branch_protection.yaml`** - Branch protection enforcement
  - Requires pull requests with 1 approval
  - Requires code owner reviews
  - Requires status checks (Super-Linter, Scorecard, OPA)
  - Blocks force pushes and deletions

## Active Policies

### 1. Branch Protection Policy

**What it checks:**

- ✅ Branch protection enabled for `master`/`main`
- ✅ Require pull request before merging
- ✅ Require 1+ approvals
- ✅ Dismiss stale reviews on push
- ✅ Require code owner reviews
- ✅ Require status checks to pass
- ✅ Block force pushes
- ✅ Block branch deletions

**Action:** Creates issue if misconfigured

### 2. Binary Artifacts Policy

**What it checks:**

- ❌ No binary files committed (executables, compiled code)
- ❌ No JAR, WAR, EAR files
- ❌ No compiled binaries (.exe, .dll, .so)

**Action:** Creates issue listing binary files

**Exceptions:** Intentional binaries in specific directories

### 3. Outside Collaborators Policy

**What it checks:**

- ✅ No unauthorized outside collaborators
- ✅ Only approved GitHub Apps have access

**Allowed apps:**

- renovate (dependency updates)
- dependabot (security updates)
- policy-bot (PR approval policies)

**Action:** Creates issue for unauthorized collaborators

### 4. Security Policy

**What it checks:**

- ✅ SECURITY.md exists in repository
- ✅ Security policy is accessible
- ✅ Vulnerability reporting instructions present

**Action:** Creates issue if missing

### 5. Dangerous Workflow Policy

**What it checks:**

- ❌ No dangerous `pull_request_target` triggers
- ❌ No unchecked code execution in workflows
- ❌ No exposed secrets in workflow files

**Action:** Creates issue for dangerous patterns

## How Allstar Works

1. **Continuous Monitoring:**
   - Allstar runs on every push, PR, and periodically
   - Checks all enabled policies against repository state

2. **Issue Creation:**
   - If policy violation found, creates GitHub issue with label `allstar`
   - Issue describes violation and remediation steps
   - Issue auto-closes when violation is resolved

3. **Auto-Remediation (Optional):**
   - When `action: fix` is set, Allstar can auto-fix certain issues
   - Currently set to `action: issue` (manual fixes required)
   - Examples of auto-fixes:
     - Enable branch protection
     - Add SECURITY.md file
     - Update workflow permissions

## Integration with Other Tools

Allstar works alongside our other governance tools:

```
GitHub Rulesets ──┐
CODEOWNERS       ─┼─> Branch protection
Policy-Bot       ─┘

Super-Linter     ──┐
OPA/Conftest     ─┼─> Code quality & policy
Scorecard        ─┤
Allstar          ─┘  Continuous monitoring

Renovate ─────────> Dependency security
SLSA Provenance ─> Supply chain security
```

## Monitoring Allstar

### Check for violations:

```bash
# View Allstar issues
gh issue list --label allstar

# Check Allstar status
# (If self-hosted, check logs)
docker logs allstar
```

### View Allstar activity:

1. Go to repository Insights → Security
2. Check "Security advisories" for Allstar findings
3. Review issues with `allstar` label

## Customizing Policies

### To enable auto-fix:

Edit `.allstar/allstar.yaml`:

```yaml
action: fix # Change from 'issue' to 'fix'
```

### To add custom policies:

Create new policy files in `.allstar/`:

```yaml
# .allstar/custom_policy.yaml
enabled: true
action: issue
# ... policy rules
```

### To exempt specific files:

```yaml
# .allstar/binary_artifacts.yaml
enabled: true
action: issue
allowedBinaryArtifacts:
  - 'scripts/tools/binary-tool'
  - 'vendor/third-party.exe'
```

## Troubleshooting

### Allstar not creating issues

1. **Check opt-in configuration:**
   - Verify repo listed in `optConfig.optInRepos`
   - Check `.allstar/allstar.yaml` exists

2. **Verify installation:**
   - Go to: https://github.com/settings/installations
   - Ensure Allstar has access to repository

3. **Check permissions:**
   - Allstar needs write access to create issues
   - Verify GitHub App permissions

### Too many issues created

1. **Temporarily disable policies:**

   ```yaml
   # .allstar/allstar.yaml
   policies:
     branch_protection:
       enabled: false # Temporarily disable
   ```

2. **Switch to log mode:**

   ```yaml
   action: log # Log instead of creating issues
   ```

3. **Fix violations, then re-enable:**
   - Resolve all current violations
   - Re-enable policies gradually

### False positives

1. **Customize policy configuration:**
   - Add exceptions in policy-specific YAML files
   - Whitelist specific patterns or files

2. **Report to Allstar:**
   - Open issue at: https://github.com/ossf/allstar/issues
   - Provide details about false positive

## Documentation

- Allstar GitHub: https://github.com/ossf/allstar
- Policy documentation: https://github.com/ossf/allstar/tree/main/pkg/policies
- OpenSSF homepage: https://openssf.org/

## Next Steps

After installing Allstar:

1. Monitor for created issues
2. Fix any policy violations
3. Consider enabling auto-remediation (`action: fix`)
4. Expand to other repositories in organization
5. Customize policies based on your security requirements
