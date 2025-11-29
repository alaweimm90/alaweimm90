#!/bin/bash
# Check if protected files are being modified without explicit approval

# Load protected files from policy
POLICY_FILE=".metaHub/policies/protected-files.yaml"

if [ ! -f "$POLICY_FILE" ]; then
  exit 0  # No policy file, skip check
fi

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only)

# Strict protected files - these should rarely change
STRICT_FILES=(
  "README.md"
  "LICENSE"
  "CODEOWNERS"
  ".github/CODEOWNERS"
)

# Check for workflow files
WORKFLOW_PATTERN=".github/workflows/*.yml"

WARNINGS=""

for file in $STAGED_FILES; do
  # Check strict files
  for protected in "${STRICT_FILES[@]}"; do
    if [ "$file" == "$protected" ]; then
      WARNINGS="$WARNINGS\n  - $file (strict protection)"
    fi
  done

  # Check workflow files
  if [[ "$file" == .github/workflows/*.yml ]]; then
    WARNINGS="$WARNINGS\n  - $file (workflow protection)"
  fi

  # Check policy files
  if [[ "$file" == .metaHub/policies/*.yaml ]]; then
    WARNINGS="$WARNINGS\n  - $file (policy protection)"
  fi
done

if [ -n "$WARNINGS" ]; then
  echo ""
  echo "⚠️  PROTECTED FILES MODIFIED:"
  echo -e "$WARNINGS"
  echo ""
  echo "These files have special protection. Ensure this change was intentional."
  echo "To proceed anyway, the commit will continue (warning only)."
  echo ""
fi

exit 0  # Warning only, don't block
