# Docker Security Policy
# Enforces security best practices in Dockerfiles

package docker_security

# Deny Dockerfiles that run as root
deny[msg] {
    input.dockerfile
    not has_user_directive(input.dockerfile)
    msg := "Dockerfile must include USER directive to run as non-root user"
}

has_user_directive(content) {
    contains(content, "USER ")
    not contains(content, "USER root")
}

# Deny Dockerfiles without HEALTHCHECK
deny[msg] {
    input.dockerfile
    not has_healthcheck(input.dockerfile)
    msg := "Dockerfile must include HEALTHCHECK directive for container monitoring"
}

has_healthcheck(content) {
    contains(content, "HEALTHCHECK")
}

# Deny untagged or :latest base images
deny[msg] {
    input.dockerfile
    line := input.dockerfile_lines[_]
    startswith(line, "FROM ")
    contains(line, ":latest")
    msg := sprintf("FROM directive uses ':latest' tag: %s. Use specific version tags", [line])
}

deny[msg] {
    input.dockerfile
    line := input.dockerfile_lines[_]
    startswith(line, "FROM ")
    not contains(line, ":")
    not contains(line, "@sha256")
    msg := sprintf("FROM directive missing tag: %s. Use specific version tags", [line])
}

# Deny ADD when COPY should be used
warn[msg] {
    input.dockerfile
    line := input.dockerfile_lines[_]
    startswith(line, "ADD ")
    not contains(line, ".tar")
    not contains(line, ".zip")
    not startswith(line, "ADD http")
    msg := sprintf("Use COPY instead of ADD: %s", [line])
}

# Deny apt-get without -y flag
deny[msg] {
    input.dockerfile
    line := input.dockerfile_lines[_]
    contains(line, "apt-get install")
    not contains(line, "-y")
    msg := sprintf("apt-get install must use -y flag: %s", [line])
}

# Deny missing apt-get clean
warn[msg] {
    input.dockerfile
    has_apt_install(input.dockerfile)
    not has_apt_clean(input.dockerfile)
    msg := "Dockerfile with apt-get should include 'apt-get clean && rm -rf /var/lib/apt/lists/*'"
}

has_apt_install(content) {
    contains(content, "apt-get install")
}

has_apt_clean(content) {
    contains(content, "apt-get clean")
    contains(content, "rm -rf /var/lib/apt/lists")
}

# Require chown for COPY operations with USER
warn[msg] {
    input.dockerfile
    has_user_directive(input.dockerfile)
    line := input.dockerfile_lines[_]
    startswith(line, "COPY ")
    not contains(line, "--chown")
    msg := sprintf("COPY with USER should use --chown flag: %s", [line])
}

# Deny exposed privileged ports
deny[msg] {
    input.dockerfile
    line := input.dockerfile_lines[_]
    startswith(line, "EXPOSE ")
    port := to_number(trim_space(substring(line, 7, -1)))
    port < 1024
    msg := sprintf("EXPOSE uses privileged port %d. Use ports >= 1024", [port])
}

# Deny secrets in ENV
deny[msg] {
    input.dockerfile
    line := input.dockerfile_lines[_]
    startswith(line, "ENV ")
    contains_secret_pattern(line)
    msg := sprintf("ENV may contain secrets: %s. Use Docker secrets or runtime config", [line])
}

contains_secret_pattern(line) {
    patterns := ["PASSWORD", "SECRET", "TOKEN", "API_KEY", "PRIVATE_KEY"]
    pattern := patterns[_]
    contains(upper(line), pattern)
}

# Require multi-stage builds for production images
warn[msg] {
    input.dockerfile
    not has_multistage_build(input.dockerfile)
    msg := "Consider using multi-stage builds to reduce image size"
}

has_multistage_build(content) {
    count([line | line := input.dockerfile_lines[_]; startswith(line, "FROM ")]) > 1
}
