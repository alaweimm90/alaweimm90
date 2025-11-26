# Dependency Security Policy
# Enforces security best practices for dependency management

package dependency_security

# Deny package.json without lockfile
deny[msg] {
    input.package_json
    not input.has_lockfile
    msg := "package.json requires package-lock.json or yarn.lock for reproducible builds"
}

# Deny requirements.txt without pinned versions
warn[msg] {
    input.requirements_txt
    line := input.requirements_lines[_]
    not contains(line, "==")
    not contains(line, ">=")
    not startswith(line, "#")
    not startswith(line, "-")
    trim_space(line) != ""
    msg := sprintf("requirements.txt: pin version for '%s'", [trim_space(line)])
}

# Deny pyproject.toml without version constraints
warn[msg] {
    input.pyproject_toml
    dep := input.dependencies[_]
    not contains(dep.version, ">=")
    not contains(dep.version, "==")
    not contains(dep.version, "^")
    not contains(dep.version, "~")
    msg := sprintf("pyproject.toml: add version constraint for '%s'", [dep.name])
}

# Deny known vulnerable packages
deny[msg] {
    input.dependencies[_].name == vulnerable_packages[_]
    msg := sprintf("Vulnerable package detected: %s. Update or replace.", [input.dependencies[_].name])
}

vulnerable_packages := [
    "lodash<4.17.21",
    "axios<0.21.1",
    "minimist<1.2.6"
]

# Warn about outdated Node.js versions in package.json
warn[msg] {
    input.package_json
    engines := input.package_json.engines
    node_version := engines.node
    is_outdated_node(node_version)
    msg := sprintf("Node.js version '%s' is outdated. Use Node.js 18+ for LTS support.", [node_version])
}

is_outdated_node(version) {
    contains(version, "14")
}

is_outdated_node(version) {
    contains(version, "16")
}

# Warn about outdated Python versions
warn[msg] {
    input.pyproject_toml
    python_version := input.pyproject_toml.project["requires-python"]
    is_outdated_python(python_version)
    msg := sprintf("Python version '%s' is outdated. Use Python 3.10+ for security updates.", [python_version])
}

is_outdated_python(version) {
    contains(version, "3.7")
}

is_outdated_python(version) {
    contains(version, "3.8")
}

is_outdated_python(version) {
    contains(version, "3.9")
}

# Require dependency update tool configuration
warn[msg] {
    not input.has_renovate
    not input.has_dependabot
    msg := "No dependency update tool configured. Add renovate.json or dependabot.yml"
}

# Deny npm scripts with dangerous commands
deny[msg] {
    input.package_json
    script := input.package_json.scripts[name]
    contains_dangerous_command(script)
    msg := sprintf("Dangerous command in npm script '%s': %s", [name, script])
}

contains_dangerous_command(script) {
    contains(script, "rm -rf /")
}

contains_dangerous_command(script) {
    contains(script, "curl | sh")
}

contains_dangerous_command(script) {
    contains(script, "wget | sh")
}

# Warn about missing security audit in CI
warn[msg] {
    input.ci_config
    not has_security_audit(input.ci_config)
    msg := "CI configuration should include security audit step (npm audit, pip-audit, etc.)"
}

has_security_audit(config) {
    contains(config, "npm audit")
}

has_security_audit(config) {
    contains(config, "pip-audit")
}

has_security_audit(config) {
    contains(config, "safety check")
}

has_security_audit(config) {
    contains(config, "cargo audit")
}

has_security_audit(config) {
    contains(config, "govulncheck")
}
