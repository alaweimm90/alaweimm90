#!/bin/bash
# Script to help identify and replace console.log statements with logger

set -e

echo "🔍 Scanning for console statements..."
echo ""

# Count console statements by type
echo "📊 Console Statement Breakdown:"
echo "  console.log:   $(grep -r "console\.log" src/ --include="*.ts" --include="*.tsx" | wc -l)"
echo "  console.error: $(grep -r "console\.error" src/ --include="*.ts" --include="*.tsx" | wc -l)"
echo "  console.warn:  $(grep -r "console\.warn" src/ --include="*.ts" --include="*.tsx" | wc -l)"
echo "  console.info:  $(grep -r "console\.info" src/ --include="*.ts" --include="*.tsx" | wc -l)"
echo ""

# Show files with most console statements
echo "📁 Top Files with Console Statements:"
grep -r "console\." src/ --include="*.ts" --include="*.tsx" -l | while read file; do
  count=$(grep "console\." "$file" | wc -l)
  echo "  $count: $file"
done | sort -rn | head -10
echo ""

# Check for files without logger import
echo "⚠️  Files using console but missing logger import:"
grep -r "console\." src/ --include="*.ts" --include="*.tsx" -l | while read file; do
  if ! grep -q "from '@/lib/logger'" "$file" && ! grep -q "from './logger'" "$file"; then
    count=$(grep "console\." "$file" | wc -l)
    echo "  $file ($count statements)"
  fi
done
echo ""

echo "✅ Scan complete!"
echo ""
echo "💡 Next steps:"
echo "  1. Add logger imports: import { logger } from '@/lib/logger';"
echo "  2. Replace console.log with logger.info or logger.debug"
echo "  3. Replace console.error with logger.error"
echo "  4. Replace console.warn with logger.warn"
echo "  5. Run npm run lint to verify"
