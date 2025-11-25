# Repository Structure Policy - STRICT ENFORCEMENT
# Enforces CANONICAL root directory structure for multi-org monorepo

package repo_structure

# CANONICAL STRUCTURE: Only these root directories/files allowed
allowed_roots := {
    ".github",
    ".metaHub",
    "alaweimm90",
    "organizations",
    ".husky",
    "SECURITY.md",
    "README.md",
    "LICENSE",
    "package.json",
    "package-lock.json",
    "pnpm-workspace.yaml",
    "turbo.json",
    "docker-compose.yml",
    "docker-compose.dev.yml",
    "docker-compose.test.yml",
    ".dockerignore",
    ".gitignore",
    "Makefile"
}

# STRICT: Only these specific paths allowed in .metaHub
allowed_metahub_paths := {
    ".metaHub/backstage",
    ".metaHub/policies",
    ".metaHub/security",
    ".metaHub/renovate.json",
    ".metaHub/service-catalog.json"
}

# Define forbidden patterns
forbidden_patterns := {
    ".DS_Store",
    "Thumbs.db",
    "*.log",
    "node_modules",
    ".env",
    ".env.local",
    "*.swp",
    "*.swo",
    "*.bak"
}

# Deny files in root that aren't in allowed_roots
deny[msg] {
    input.file.path
    not startswith(input.file.path, ".")
    parts := split(input.file.path, "/")
    root := parts[0]
    not allowed_roots[root]
    msg := sprintf("File '%s' violates repository structure. Root '%s' is not in allowed_roots", [input.file.path, root])
}

# STRICT ENFORCEMENT: Deny ANY file in .metaHub that's not explicitly allowed
deny[msg] {
    input.file.path
    startswith(input.file.path, ".metaHub/")
    not metahub_path_allowed(input.file.path)
    msg := sprintf("BLOCKED: '%s' not allowed in .metaHub. Only backstage/, policies/, security/, renovate.json, service-catalog.json permitted", [input.file.path])
}

# Helper: Check if .metaHub path is allowed
metahub_path_allowed(path) {
    allowed_path := allowed_metahub_paths[_]
    startswith(path, allowed_path)
}

# Deny forbidden file patterns anywhere
deny[msg] {
    input.file.path
    forbidden_patterns[pattern]
    contains(input.file.path, pattern)
    msg := sprintf("File '%s' matches forbidden pattern '%s'", [input.file.path, pattern])
}

# Deny Dockerfiles outside allowed directories
deny[msg] {
    input.file.path
    endswith(input.file.path, "Dockerfile")
    not dockerfile_in_allowed_location(input.file.path)
    msg := sprintf("Dockerfile at '%s' must be in service directory, not root or arbitrary locations", [input.file.path])
}

dockerfile_in_allowed_location(path) {
    startswith(path, "organizations/")
}

dockerfile_in_allowed_location(path) {
    startswith(path, "alaweimm90/")
}

dockerfile_in_allowed_location(path) {
    startswith(path, ".metaHub/")
}

# Warn about large files (>10MB)
warn[msg] {
    input.file.size > 10485760
    not is_allowed_large_file(input.file.path)
    msg := sprintf("File '%s' is %d bytes (>10MB). Consider using Git LFS", [input.file.path, input.file.size])
}

is_allowed_large_file(path) {
    # Allow large files in specific directories
    startswith(path, ".metaHub/cache/")
}

is_allowed_large_file(path) {
    endswith(path, ".pdf")
}
