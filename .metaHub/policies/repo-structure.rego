# Repository Structure Policy
# Enforces allowed root directory structure for multi-org monorepo

package repo_structure

# Define allowed root directories
allowed_roots := {
    ".github",
    ".metaHub",
    ".config",
    "apps",
    "packages",
    "alaweimm90",
    "ops",
    "scripts",
    "templates",
    "docs",
    "SECURITY.md",
    "README.md",
    "package.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "turbo.json",
    "docker-compose.yml",
    ".dockerignore",
    ".gitignore",
    ".editorconfig",
    "Makefile",
    "LICENSE"
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
    "*.swo"
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
    startswith(path, ".config/organizations/")
}

dockerfile_in_allowed_location(path) {
    startswith(path, "apps/")
}

dockerfile_in_allowed_location(path) {
    startswith(path, "alaweimm90/")
}

dockerfile_in_allowed_location(path) {
    startswith(path, ".metaHub/")
}

dockerfile_in_allowed_location(path) {
    startswith(path, "ops/")
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
    startswith(path, "docs/assets/")
}

is_allowed_large_file(path) {
    endswith(path, ".pdf")
}
