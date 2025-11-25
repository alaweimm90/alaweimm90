#!/bin/bash
# OPA Policy Enforcement Pre-commit Hook
# Validates repository structure and Docker security

set -e

POLICY_DIR=".metaHub/policies"
FAILED=0

echo "🔍 Running OPA policy checks..."

# Check if OPA is installed
if ! command -v opa &> /dev/null; then
    echo "⚠️  OPA not installed. Installing..."

    # Detect OS
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    if [ "$ARCH" == "x86_64" ]; then
        ARCH="amd64"
    elif [ "$ARCH" == "aarch64" ]; then
        ARCH="arm64"
    fi

    # Download OPA
    OPA_VERSION="0.68.0"
    curl -L -o /tmp/opa "https://openpolicyagent.org/downloads/v${OPA_VERSION}/opa_${OS}_${ARCH}"
    chmod +x /tmp/opa
    sudo mv /tmp/opa /usr/local/bin/opa

    echo "✅ OPA installed successfully"
fi

# Get staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
    echo "✅ No files staged, skipping OPA checks"
    exit 0
fi

echo "📝 Checking $(echo "$STAGED_FILES" | wc -l) staged files..."

# Check repository structure policy
echo ""
echo "1️⃣  Validating repository structure..."

for FILE in $STAGED_FILES; do
    # Skip deleted files
    if [ ! -f "$FILE" ]; then
        continue
    fi

    # Get file size
    SIZE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE" 2>/dev/null || echo 0)

    # Create JSON input for OPA
    cat > /tmp/opa-input.json <<EOF
{
  "file": {
    "path": "$FILE",
    "size": $SIZE
  }
}
EOF

    # Run OPA eval
    RESULT=$(opa eval -d "$POLICY_DIR/repo-structure.rego" -i /tmp/opa-input.json "data.repo_structure.deny" -f json)

    # Check for denials
    DENIALS=$(echo "$RESULT" | jq -r '.result[0].expressions[0].value // []')

    if [ "$DENIALS" != "[]" ] && [ "$DENIALS" != "null" ]; then
        echo "❌ Policy violation in $FILE:"
        echo "$DENIALS" | jq -r '.[]'
        FAILED=1
    fi

    # Check for warnings
    WARNINGS=$(opa eval -d "$POLICY_DIR/repo-structure.rego" -i /tmp/opa-input.json "data.repo_structure.warn" -f json | jq -r '.result[0].expressions[0].value // []')

    if [ "$WARNINGS" != "[]" ] && [ "$WARNINGS" != "null" ]; then
        echo "⚠️  Warning for $FILE:"
        echo "$WARNINGS" | jq -r '.[]'
    fi
done

# Check Docker security policy for Dockerfiles
echo ""
echo "2️⃣  Validating Dockerfile security..."

DOCKERFILES=$(echo "$STAGED_FILES" | grep -E 'Dockerfile$' || true)

if [ -n "$DOCKERFILES" ]; then
    for DOCKERFILE in $DOCKERFILES; do
        if [ ! -f "$DOCKERFILE" ]; then
            continue
        fi

        echo "   Checking $DOCKERFILE..."

        # Read Dockerfile content
        CONTENT=$(cat "$DOCKERFILE")

        # Create lines array
        mapfile -t LINES < "$DOCKERFILE"
        LINES_JSON=$(printf '%s\n' "${LINES[@]}" | jq -R . | jq -s .)

        # Create JSON input
        cat > /tmp/opa-docker-input.json <<EOF
{
  "dockerfile": $(echo "$CONTENT" | jq -Rs .),
  "dockerfile_lines": $LINES_JSON
}
EOF

        # Run Docker security policy
        DOCKER_DENIALS=$(opa eval -d "$POLICY_DIR/docker-security.rego" -i /tmp/opa-docker-input.json "data.docker_security.deny" -f json | jq -r '.result[0].expressions[0].value // []')

        if [ "$DOCKER_DENIALS" != "[]" ] && [ "$DOCKER_DENIALS" != "null" ]; then
            echo "   ❌ Security violation in $DOCKERFILE:"
            echo "$DOCKER_DENIALS" | jq -r '.[]' | sed 's/^/      /'
            FAILED=1
        fi

        # Check warnings
        DOCKER_WARNINGS=$(opa eval -d "$POLICY_DIR/docker-security.rego" -i /tmp/opa-docker-input.json "data.docker_security.warn" -f json | jq -r '.result[0].expressions[0].value // []')

        if [ "$DOCKER_WARNINGS" != "[]" ] && [ "$DOCKER_WARNINGS" != "null" ]; then
            echo "   ⚠️  Warning for $DOCKERFILE:"
            echo "$DOCKER_WARNINGS" | jq -r '.[]' | sed 's/^/      /'
        else
            echo "   ✅ Dockerfile security checks passed"
        fi
    done
else
    echo "   No Dockerfiles changed"
fi

# Cleanup
rm -f /tmp/opa-input.json /tmp/opa-docker-input.json

# Final result
echo ""
if [ $FAILED -eq 1 ]; then
    echo "❌ OPA policy checks FAILED. Fix violations before committing."
    exit 1
else
    echo "✅ All OPA policy checks passed!"
    exit 0
fi
