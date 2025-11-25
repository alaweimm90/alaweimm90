# Governance Troubleshooting Runbook

Quick reference for common issues with the governance toolchain.

---

## Issue: Super-Linter failing on valid code

**Symptoms**: PR blocked by Super-Linter for false positive

**Diagnosis**:

```bash
# Check which linter failed
gh pr view <pr-number> --json statusCheckRollup

# View detailed logs
gh run view <run-id> --log
```

**Solution**:

1. Review linter output in Actions tab
2. If specific validator is problematic, disable in `.github/workflows/super-linter.yml`:

   ```yaml
   env:
     VALIDATE_<LANGUAGE>: false
   ```

3. Or add exceptions in `.github/linters/` directory:

   ```bash
   # Example: Disable specific ESLint rules
   mkdir -p .github/linters
   cat > .github/linters/.eslintrc.json <<'EOF'
   {
     "rules": {
       "no-console": "off"
     }
   }
   EOF
   ```

4. Commit fix, re-run checks

**Common False Positives**:

- `VALIDATE_JSCPD`: Copy-paste detection too strict → Disable
- `VALIDATE_NATURAL_LANGUAGE`: Prose linting not needed → Disable
- `VALIDATE_CSS`: No CSS files in repo → Disable

---

## Issue: OPA policy rejecting valid Dockerfile

**Symptoms**: Conftest fails with policy violation on legitimate Dockerfile

**Diagnosis**:

```bash
# Test policy locally
cd .metaHub/policies
conftest test --policy docker-security.rego /path/to/Dockerfile

# View policy rules
cat docker-security.rego | grep -A 5 "deny\["
```

**Solution**:

1. Read policy violation message carefully
2. Check `.metaHub/policies/docker-security.rego` for the specific rule
3. Option A: Fix the Dockerfile to comply with policy
4. Option B: Add exception to policy:

   ```rego
   # Allow :latest for development images
   allow_latest_for_dev := {
     "node:latest",
     "python:latest"
   }

   deny[msg] {
     input.dockerfile
     line := input.dockerfile_lines[_]
     startswith(line, "FROM ")
     contains(line, ":latest")
     not allow_latest_for_dev[trim_space(line)]
     msg := sprintf("FROM uses :latest tag: %s", [line])
   }
   ```

5. Test locally: `conftest test --policy .metaHub/policies/ Dockerfile`
6. Commit policy update

**Common Violations**:

- `:latest` tag → Use specific version: `FROM node:20-alpine`
- Missing `USER` → Add before CMD: `USER node`
- Missing `HEALTHCHECK` → Add healthcheck command
- No `-y` flag → `RUN apt-get install -y curl`
- Secrets in ENV → Use runtime environment variables

---

## Issue: Renovate creating too many PRs

**Symptoms**: Overwhelmed with dependency update PRs (10+)

**Diagnosis**:

```bash
# Count open Renovate PRs
gh pr list --label dependencies --json number | jq '. | length'

# View Renovate dashboard
gh issue list --search "Dependency Dashboard in:title"
```

**Solution**:

1. **Temporary**: Close excess PRs, Renovate will recreate later
2. **Permanent**: Adjust `.metaHub/renovate.json`:

   ```json
   {
     "prConcurrentLimit": 2,  // Reduce from 5
     "prHourlyLimit": 1,       // Reduce from 2
     "schedule": [
       "after 10pm on friday"  // Less frequent
     ]
   }
   ```

3. **Pause**: Temporarily disable Renovate:

   ```json
   {
     "enabled": false
   }
   ```

4. **Group**: Combine related updates:

   ```json
   {
     "packageRules": [
       {
         "groupName": "all non-major dependencies",
         "matchUpdateTypes": ["minor", "patch"],
         "groupSlug": "all-minor-patch"
       }
     ]
   }
   ```

5. Commit changes, Renovate will respect new config on next run

---

## Issue: Policy-Bot not enforcing rules

**Symptoms**: PRs merge without required approvals

**Diagnosis**:

```bash
# Check if Policy-Bot is installed
gh api /repos/alaweimm90/alaweimm90/installation | jq '.app_slug'

# Check for Policy-Bot comment on PR
gh pr view <pr-number> --json comments | jq '.comments[] | select(.author.login == "policy-bot")'

# Verify webhook deliveries (if you have access)
# Go to: https://github.com/settings/installations
```

**Solution**:

1. **Verify GitHub App installed**:
   - Go to: https://github.com/settings/installations
   - Ensure Policy-Bot has access to repository

2. **Check config syntax**:

   ```bash
   # Validate YAML
   yamllint .metaHub/policy-bot.yml
   ```

3. **Review webhook delivery**:
   - In App settings, check webhook deliveries
   - Look for failed deliveries or errors

4. **Check Policy-Bot comment**:
   - Policy-Bot should comment on every PR
   - If no comment, webhook not working

5. **Verify in required status checks**:
   - Go to: https://github.com/alaweimm90/alaweimm90/settings/rules
   - Ensure `policy-bot` is in required status checks list

6. **Test with simple PR**:

   ```bash
   git checkout -b test-policy-bot
   echo "# test" >> .metaHub/README.md
   git commit -am "test: policy-bot enforcement"
   git push origin test-policy-bot
   gh pr create --title "Test Policy-Bot"
   ```

---

## Issue: Allstar creating false positive issues

**Symptoms**: Allstar issues for legitimate configurations

**Diagnosis**:

```bash
# List all Allstar issues
gh issue list --label allstar

# View specific issue details
gh issue view <issue-number>
```

**Solution**:

1. Review issue details - is it a real violation?

2. **If false positive**, adjust policy in `.allstar/<policy>.yaml`:

   **Disable policy**:

   ```yaml
   enabled: false
   ```

   **Add exceptions**:

   ```yaml
   # For binary artifacts
   allowedBinaryArtifacts:
     - "path/to/legitimate-binary"

   # For outside collaborators
   allowedCollaborators:
     - "trusted-external-user"

   # For GitHub Apps
   allowedGitHubApps:
     - "renovate"
     - "dependabot"
     - "policy-bot"
   ```

3. Close Allstar issue (will reopen if still violated after config change)

4. Commit config changes:

   ```bash
   git add .allstar/
   git commit -m "fix(allstar): add exception for legitimate use case"
   git push origin master
   ```

**Common False Positives**:

- Binary artifacts: Test fixtures, vendored binaries
- Outside collaborators: Legitimate contractors
- Branch protection: Custom protection rules

---

## Issue: SLSA provenance generation failing

**Symptoms**: Workflow fails on provenance step, no attestations created

**Diagnosis**:

```bash
# Check recent SLSA workflow runs
gh run list --workflow=slsa-provenance.yml --limit 5

# View failed run logs
gh run view <run-id> --log

# Check for stored provenances
ls -la .metaHub/security/slsa/
```

**Solution**:

1. **Check Actions logs** for specific error

2. **Common issues**:

   **Artifact too large**:
   - Reduce what's packaged in workflow
   - Exclude unnecessary files

   **Hash generation failed**:
   - Verify file paths exist
   - Check tar command succeeded

   **SLSA verifier download failed**:
   - Network issue, retry workflow
   - Check verifier version still available

   **Permission denied**:
   - Verify `contents: write` permission in workflow
   - Check branch protection allows workflow commits

3. **Temporary workaround**: Disable provenance temporarily:

   ```yaml
   # .github/workflows/slsa-provenance.yml
   on:
     workflow_dispatch:  # Manual trigger only
     # Comment out automatic triggers
   ```

4. **Test locally**:

   ```bash
   # Manually generate artifact
   tar -czf test-artifact.tar.gz .metaHub/policies/
   sha256sum test-artifact.tar.gz

   # Verify structure
   tar -tzf test-artifact.tar.gz
   ```

---

## Issue: GitHub Rulesets not blocking merge

**Symptoms**: Can merge PR despite failed checks or missing approvals

**Diagnosis**:

```bash
# Check ruleset status via API
gh api /repos/alaweimm90/alaweimm90/rulesets

# Check branch protection
gh api /repos/alaweimm90/alaweimm90/branches/master/protection
```

**Solution**:

1. **Go to Settings → Rules**:
   - https://github.com/alaweimm90/alaweimm90/settings/rules

2. **Verify ruleset is active**:
   - Not in "Evaluate" mode (test mode)
   - Status should be "Active"

3. **Check enforcement settings**:
   - Bypass list should be empty or minimal
   - All required checks listed correctly

4. **Verify status check names match exactly**:
   - Check Actions workflow for exact `name:` field
   - Example: "Super-Linter" not "super-linter"
   - Must match exactly (case-sensitive)

5. **Test with admin account**:
   - If you're admin, you may have bypass privileges
   - Check if non-admin users are properly blocked

**Common Issues**:

- Ruleset in "Evaluate" mode → Set to "Active"
- Status check name mismatch → Fix exact naming
- Admin bypass enabled → Disable for testing
- Wrong branch targeted → Verify branch pattern

---

## Issue: Pre-commit hook blocking commits

**Symptoms**: `git commit` fails with "COMMIT BLOCKED: Code quality issues"

**Diagnosis**:

```bash
# Check pre-commit hook
cat .husky/pre-commit

# Test ESLint directly
npx eslint .
```

**Solution**:

1. **Hook references old structure** (looking for `src/` directory that doesn't exist)

2. **Temporary bypass** for governance changes:

   ```bash
   git commit --no-verify -m "your message"
   ```

3. **Permanent fix**: Update `.husky/pre-commit` to match canonical structure:

   ```bash
   # Replace hardcoded "src" path with dynamic detection
   # Or disable specific checks that don't apply
   ```

4. **Alternative**: Disable hook temporarily:

   ```bash
   # Rename hook
   mv .husky/pre-commit .husky/pre-commit.disabled

   # Re-enable later
   mv .husky/pre-commit.disabled .husky/pre-commit
   ```

**Note**: GitHub-level checks (Super-Linter, OPA) will still run on PR, so bypass is safe for governance changes.

---

## Issue: Renovate PRs not auto-merging

**Symptoms**: PRs eligible for auto-merge but remain open

**Diagnosis**:

```bash
# Check PR labels
gh pr view <pr-number> --json labels

# Check if all checks passed
gh pr view <pr-number> --json statusCheckRollup
```

**Solution**:

1. **Verify conditions met**:
   - All status checks passed
   - `minimumReleaseAge` elapsed (default: 3 days)
   - Update type is `minor` or `patch` (not `major`)
   - Labeled correctly

2. **Check Renovate config**:

   ```json
   {
     "platformAutomerge": false,  // Should be false (use PR auto-merge)
     "automergeType": "pr",
     "automergeStrategy": "squash"
   }
   ```

3. **Verify auto-merge enabled on repository**:
   - Go to Settings → General
   - Ensure "Allow auto-merge" is checked

4. **Check if approval required**:
   - Renovate can't auto-merge if approval required by branch protection
   - Either approve manually or exempt Renovate bot

5. **Manual merge**:

   ```bash
   gh pr review <pr-number> --approve
   gh pr merge <pr-number> --squash --auto
   ```

---

## Issue: Backstage catalog not loading services

**Symptoms**: Backstage portal shows empty catalog or missing services

**Diagnosis**:

```bash
# Start Backstage with logs
cd .metaHub/backstage
node server.js

# Check catalog file syntax
yamllint .metaHub/backstage/catalog-info.yaml

# Test catalog processing
cat .metaHub/backstage/catalog-info.yaml | grep -A 5 "kind: Component"
```

**Solution**:

1. **Check app-config.yaml**:

   ```yaml
   catalog:
     locations:
       - type: file
         target: ../../.metaHub/backstage/catalog-info.yaml
   ```

2. **Verify catalog-info.yaml syntax**:
   - Valid YAML
   - Correct `apiVersion: backstage.io/v1alpha1`
   - Proper `kind` (Component, System, Resource, API)

3. **Check file paths**:
   - Relative paths in app-config must be correct
   - `target` should point to actual file

4. **Restart Backstage**:

   ```bash
   cd .metaHub/backstage
   # Kill existing process
   pkill -f "node server.js"
   # Restart
   node server.js
   ```

5. **Check browser console** for errors (F12)

---

## Emergency: Need to bypass governance for hotfix

**Scenario**: Production down, need immediate fix without waiting for governance

**Process**:

1. **Assess if truly emergency**:
   - Production outage?
   - Security breach?
   - Data loss imminent?

2. **If you're repository admin**:
   - You can bypass rulesets if configured
   - Go to PR and use admin override

3. **Create hotfix**:

   ```bash
   git checkout -b hotfix-production-outage
   # Make MINIMAL fix
   git commit -m "hotfix: fix production outage

   Emergency bypass: Production down, users unable to access service.
   Details: [brief description]
   Follow-up PR: [will create proper PR after]"

   git push origin hotfix-production-outage
   ```

4. **Use admin override** to merge (if available)

5. **IMMEDIATELY AFTER**:
   - Create follow-up PR with proper process
   - Document bypass in incident report
   - Notify team of governance bypass
   - Review why emergency bypass was needed
   - Update runbooks to prevent future emergencies

**Prevention**:

- Configure Rulesets with admin bypass option for emergencies
- Maintain staging environment to catch issues
- Have rollback plan ready
- Keep on-call engineer with admin access

---

## General Debugging Tips

### Check Tool Versions

```bash
# GitHub CLI
gh --version

# Node.js
node --version

# Git
git --version

# Check workflow action versions
grep "uses:" .github/workflows/*.yml
```

### View Workflow Logs

```bash
# List recent runs
gh run list --limit 10

# View specific run
gh run view <run-id>

# Download logs
gh run download <run-id>
```

### Test Locally Before Pushing

```bash
# Run Super-Linter locally
docker run -e RUN_LOCAL=true -v $(pwd):/tmp/lint github/super-linter:latest

# Test OPA policies
conftest test --policy .metaHub/policies/ <file>

# Run ESLint
npx eslint .

# Check YAML syntax
yamllint .
```

### Enable Debug Logging

Add to workflow:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

### Check GitHub Status

If tools are failing unexpectedly:

- https://www.githubstatus.com/
- Check if Actions are experiencing issues

---

## Getting Help

### Documentation

- [Governance Summary](./GOVERNANCE_SUMMARY.md) - Complete implementation guide
- [Developer Guide](./DEVELOPER_GUIDE.md) - How to work with tools
- [Monitoring Checklist](./MONITORING_CHECKLIST.md) - Regular maintenance

### Tool-Specific Docs

- Super-Linter: https://github.com/super-linter/super-linter
- OPA/Conftest: https://www.conftest.dev/
- Renovate: https://docs.renovatebot.com/
- Policy-Bot: https://github.com/palantir/policy-bot
- Allstar: https://github.com/ossf/allstar
- SLSA: https://slsa.dev/
- Scorecard: https://github.com/ossf/scorecard
- Backstage: https://backstage.io/docs/

### Community Support

- GitHub Community: https://github.community/
- OpenSSF Slack: https://openssf.slack.com/
- Backstage Discord: https://discord.gg/backstage

### Contact

For policy exceptions or urgent issues:
- @alaweimm90
- Create issue with label: `governance-help`

---

Last updated: [DATE]
Version: 1.0
