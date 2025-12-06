# Repository Structure Policy — WARNINGS ONLY
# This repo enforces governance contract. Violations are WARNINGS, not blockers.

package repo_structure

# Root files allowed in this meta-governance repo
allowed_roots := {
    ".github",
    ".metaHub",
    ".allstar",
    ".gitignore",
    ".gitattributes",
    "README.md",
    "LICENSE"
}

# .metaHub subdirectories allowed
allowed_metahub_paths := {
    ".metaHub/policies",
    ".metaHub/schemas",
    ".metaHub/infra/examples",
    ".metaHub/templates",
    ".metaHub/guides",
    ".metaHub/examples",
    ".metaHub/docs",
    ".metaHub/security"
}

# Warn (don't deny) about structure violations
warn[msg] {
    input.file.path
    not startswith(input.file.path, ".")
    parts := split(input.file.path, "/")
    root := parts[0]
    not allowed_roots[root]
    msg := sprintf("LINT: File '%s' outside governance contract. Move to dedicated repo or delete.", [input.file.path])
}

warn[msg] {
    input.file.path
    startswith(input.file.path, ".metaHub/")
    not metahub_path_allowed(input.file.path)
    msg := sprintf("LINT: '%s' not in governance contract. Move to separate repo or delete.", [input.file.path])
}

metahub_path_allowed(path) {
    allowed_path := allowed_metahub_paths[_]
    startswith(path, allowed_path)
}

# Warn about large files
warn[msg] {
    input.file.size > 10485760
    msg := sprintf("LINT: '%s' is large (>10MB). Consider Git LFS or move to data repo.", [input.file.path])
}

# Allow everything (no hard denials)
pass = true

# Hint: suggest structured locations for configs and tools
warn[msg] {
    input.file.path
    endswith(input.file.path, "-config.yaml")
    not startswith(input.file.path, ".config/")
    msg := sprintf("LINT: Config '%s' should live under .config/** instead of repo root.", [input.file.path])
}

warn[msg] {
    input.file.path
    endswith(input.file.path, ".ts")
    re_match("^[^/]+-(cli|script)\\.ts$", input.file.path)
    msg := sprintf("LINT: CLI '%s' should live under tools/** instead of repo root.", [input.file.path])
}
