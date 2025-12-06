# Requires GitHub CLI (gh) and repository admin permissions.
# Usage: ./scripts/branch-protection.ps1 <repo_full_name>
param(
  [Parameter(Mandatory=$true)][string]$Repo
)

Write-Host "Configuring branch protection for $Repo"

# Protect main
gh api -X PUT repos/$Repo/branches/main/protection -F required_status_checks='{"strict":true,"contexts":["CI","CodeQL","Commit Message Lint","SBOM"]}' -F enforce_admins=true -F required_pull_request_reviews='{"dismiss_stale_reviews":true,"required_approving_review_count":1}' -F restrictions='null' -F allow_force_pushes=false -F allow_deletions=false

# Protect develop (if exists)
try {
  gh api -X PUT repos/$Repo/branches/develop/protection -F required_status_checks='{"strict":true,"contexts":["CI","CodeQL","Commit Message Lint","SBOM"]}' -F enforce_admins=true -F required_pull_request_reviews='{"dismiss_stale_reviews":true,"required_approving_review_count":1}' -F restrictions='null' -F allow_force_pushes=false -F allow_deletions=false
} catch {
  Write-Host "Develop branch not found; skipped protection." -ForegroundColor Yellow
}

