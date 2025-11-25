# Policy-Bot Setup Guide

Policy-Bot is a GitHub App that enforces advanced PR approval policies based on file paths and custom rules.

## Installation

Policy-Bot requires installation as a GitHub App. Choose one of these options:

### Option 1: Install Palantir's Hosted Version (Recommended)

1. **Visit the GitHub App page:**
   - Go to: https://github.com/apps/policy-bot
   - Click "Install" or "Configure"

2. **Select repository:**
   - Choose "alaweimm90/alaweimm90" repository
   - Grant required permissions

3. **Verify installation:**
   - Policy-Bot will automatically read `.metaHub/policy-bot.yml`
   - Check PR status checks for "policy-bot" context

### Option 2: Self-Host Policy-Bot

If you prefer to self-host for greater control:

1. **Clone Policy-Bot:**
   ```bash
   git clone https://github.com/palantir/policy-bot.git
   cd policy-bot
   ```

2. **Create GitHub App:**
   - Go to GitHub Settings → Developer settings → GitHub Apps → New GitHub App
   - Set webhook URL to your server
   - Grant permissions:
     - Repository permissions:
       - Pull requests: Read & write
       - Checks: Read & write
       - Contents: Read
       - Metadata: Read
     - Subscribe to events:
       - Pull request
       - Pull request review
       - Push
       - Status

3. **Configure Policy-Bot:**
   ```yaml
   # config.yml
   server:
     address: "0.0.0.0"
     port: 8080

   github:
     app:
       integration_id: YOUR_APP_ID
       webhook_secret: YOUR_WEBHOOK_SECRET
       private_key: |
         -----BEGIN RSA PRIVATE KEY-----
         YOUR_PRIVATE_KEY_HERE
         -----END RSA PRIVATE KEY-----

   options:
     policy_path: .metaHub/policy-bot.yml
   ```

4. **Deploy:**
   ```bash
   # Using Docker
   docker build -t policy-bot .
   docker run -p 8080:8080 -v /path/to/config.yml:/config.yml policy-bot

   # Or using Docker Compose (add to root docker-compose.yml)
   ```

## Configuration Location

The Policy-Bot configuration is located at:
```
.metaHub/policy-bot.yml
```

## Current Rules

Our policy-bot.yml defines these approval rules:

### 1. Governance Changes (Critical)
- **Paths**: `.metaHub/**`, `.github/workflows/**`, `.github/CODEOWNERS`, `SECURITY.md`
- **Requires**: 2 approvals from security team
- **Author approval**: Not allowed
- **Invalidated on push**: Yes

### 2. Policy Changes
- **Paths**: `.metaHub/policies/**/*.rego`
- **Requires**: 1 security team approval
- **Author approval**: Not allowed

### 3. Docker Changes
- **Paths**: `Dockerfile*`, `docker-compose*.yml`
- **Requires**: 1 platform team approval
- **Author approval**: Not allowed

### 4. Dependency Changes
- **Paths**: `package.json`, `pnpm-workspace.yaml`, `.metaHub/renovate.json`
- **Requires**: 1 security approval
- **Invalidated on push**: No (allows lockfile updates)

### 5. Workflow Changes
- **Paths**: `.github/workflows/**`, `.github/actions/**`
- **Requires**: 1 DevOps approval
- **Author approval**: Not allowed

### 6. Organization Workspace Changes
- **Paths**: `organizations/**`, `alaweimm90/**`
- **Requires**: 1 org owner approval
- **Author approval**: Allowed (for personal workspace)

## Blocking Conditions

PRs are automatically blocked if they have these labels:
- `do-not-merge`
- `wip`
- `blocked`

## Required Status Checks

Policy-Bot enforces that these checks must pass:
- Super-Linter
- OpenSSF Scorecard
- OPA Policy Enforcement

## Auto-Labeling

PRs are automatically labeled based on file changes:
- `.metaHub/**` → `governance`, `infrastructure`
- `Dockerfile*` → `docker`, `infrastructure`
- `.github/workflows/**` → `ci-cd`, `automation`
- `**/*.rego` → `policy`, `security`
- `package.json` → `dependencies`

## Testing Policy-Bot

After installation, test by creating a PR that:

1. **Tests governance approval:**
   ```bash
   # Touch a governance file
   echo "# test" >> .metaHub/README.md
   git checkout -b test-policy-bot
   git add .metaHub/README.md
   git commit -m "test: policy-bot governance approval"
   git push origin test-policy-bot
   ```

2. **Verify status check:**
   - PR should show "policy-bot" status check
   - Status should indicate "Missing required approvals"
   - After approval from @alaweimm90, status should pass

## Integration with GitHub Rulesets

Policy-Bot works alongside GitHub Rulesets:
- **Rulesets**: Enforce branch protection, require PR, block force push
- **Policy-Bot**: Enforce custom approval rules based on file paths and conditions

Together they provide defense-in-depth:
1. Rulesets prevent direct commits
2. Policy-Bot ensures correct approvals
3. CODEOWNERS provides file-level ownership
4. Status checks (Super-Linter, Scorecard, OPA) validate code quality

## Troubleshooting

### Policy-Bot not showing up
- Check GitHub App installation at: https://github.com/settings/installations
- Verify repository is selected in installation settings
- Check webhook deliveries for errors

### Policy not being enforced
- Verify `.metaHub/policy-bot.yml` is valid YAML
- Check Policy-Bot logs (if self-hosted)
- Ensure file paths in rules match actual repository structure

### Status check failing incorrectly
- Review policy-bot.yml syntax
- Check team/user names are correct
- Verify required status checks exist in other workflows

## Documentation

- Policy-Bot GitHub: https://github.com/palantir/policy-bot
- Configuration reference: https://github.com/palantir/policy-bot/blob/develop/docs/configuration.md
- Approval rules: https://github.com/palantir/policy-bot/blob/develop/docs/approval-rules.md
