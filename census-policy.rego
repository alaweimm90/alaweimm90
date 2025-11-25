package portfolio

# census-policy.rego — Enforce no orphan projects or drafts without repos
# Usage: Run against inventory.json in CI

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Default allow (pass if no violations)
default pass = true

# =============================================================================
# RULE 1: No projects without repo linkage
# =============================================================================
deny[msg] if {
    some i
    orphan := input.projects_without_repo[i]

    # Exception list (from standards/EXCEPTIONS.md)
    not in_exception_list(orphan.org, orphan.project_number)

    msg := sprintf(
        "VIOLATION: Project %s (org: %s) has no repo linkage - reason: %s",
        [orphan.project_number, orphan.org, orphan.reason]
    )
}

# =============================================================================
# RULE 2: No orphaned DraftIssues
# =============================================================================
deny[msg] if {
    some i
    draft := input.orphan_drafts[i]

    # Exception list
    not in_draft_exception(draft.org, draft.project)

    msg := sprintf(
        "VIOLATION: DraftIssue in project '%s' (org: %s) has no repo - draft: '%s'",
        [draft.project, draft.org, draft.draft_title]
    )
}

# =============================================================================
# RULE 3: All repos must have required metadata
# =============================================================================
deny[msg] if {
    some org
    some i
    repo := input.repos_by_org[org].repos[i]

    # Check if repo follows prefix taxonomy
    not matches_prefix(repo.full_name)

    msg := sprintf(
        "VIOLATION: Repo '%s' does not match prefix taxonomy (core-, lib-, adapter-, tool-, template-, demo-, infra-, paper-)",
        [repo.full_name]
    )
}

# =============================================================================
# RULE 4: Projects should have at least one linked item
# =============================================================================
deny[msg] if {
    some org
    some i
    project := input.projects_v2_by_org[org].projects[i]

    # If project exists but has no linked repos, warn (not error)
    # This is a softer rule than projects_without_repo

    msg := sprintf(
        "WARNING: Project '%s' in org '%s' should link to at least one repo",
        [project.title, org]
    )
}

# =============================================================================
# HELPER: Exception list lookup
# =============================================================================
in_exception_list(org, proj_num) if {
    # Load exceptions from standards/EXCEPTIONS.md if available
    # Format: "org/proj_num: reason"
    exceptions := {
        # Add entries here or load from config
    }
    exceptions[sprintf("%s/%s", [org, proj_num])]
}

in_draft_exception(org, project) if {
    exceptions := {
        # Add draft exceptions here
    }
    exceptions[sprintf("%s/%s", [org, project])]
}

# =============================================================================
# HELPER: Prefix taxonomy validation
# =============================================================================
matches_prefix(repo_name) if {
    prefixes := [
        "core-",
        "lib-",
        "adapter-",
        "tool-",
        "template-",
        "demo-",
        "infra-",
        "paper-"
    ]

    some prefix in prefixes
    endswith(split(repo_name, "/")[0], split(prefix, "-")[0])  # Extract prefix
}

matches_prefix(repo_name) if {
    # Special cases: .github, alaweimm90, etc (metadata repos)
    metadata_repos := {
        ".github",
        "alaweimm90",
        "standards",
        "AlaweinOS",
        "alaweimm90-science",
        "alaweimm90-business",
        "alaweimm90-tools"
    }

    repo := split(repo_name, "/")[1]
    repo in metadata_repos
}

# =============================================================================
# REPORT
# =============================================================================
report[summary] if {
    total_repos := count(flatten([input.repos_by_org[org].repos | _ := input.repos_by_org[_]]))
    orphan_count := count(input.projects_without_repo)
    draft_count := count(input.orphan_drafts)

    summary := {
        "total_repositories": total_repos,
        "orphan_projects": orphan_count,
        "orphan_drafts": draft_count,
        "status": "PASS" if orphan_count == 0 and draft_count == 0 else "FAIL"
    }
}

# =============================================================================
# Final decision
# =============================================================================
pass if {
    count(deny) == 0
}

fail if {
    count(deny) > 0
}
