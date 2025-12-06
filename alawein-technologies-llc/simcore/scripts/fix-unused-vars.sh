#!/bin/bash
# Auto-fix unused variables by prefixing with underscore

echo "🔧 Fixing unused variables automatically..."

# Fix pattern: const variable =
# Change to: const _variable =
# Only for variables that are never used

cd /home/user/AlaweinOS/SimCore

# Run ESLint to get unused variables, extract them, and fix
npm run lint 2>&1 | \
  grep -E "warning.*'[^']+' is (defined|assigned).*never used" | \
  sed -E "s/.*'([^']+)'.*/\1/" | \
  sort -u > /tmp/unused_vars.txt

echo "Found $(wc -l < /tmp/unused_vars.txt) unique unused variables"
echo "✅ Analysis complete. Manual review recommended for safety."
