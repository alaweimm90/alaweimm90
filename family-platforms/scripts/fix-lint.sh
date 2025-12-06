#!/bin/bash

# Fix all markdown lint issues automatically
echo "🔧 Fixing markdown lint issues..."

# Install markdownlint-cli2 if not already installed
npm install -g markdownlint-cli2

# Fix all markdown files in the project
echo "📝 Fixing markdown files..."
find . -name "*.md" -not -path "./node_modules/*" -not -path "./dist/*" | xargs markdownlint-cli2 --fix

# Run prettier on all files
echo "💅 Running prettier..."
npx prettier --write . --ignore-path .gitignore

# Run ESLint fix on TypeScript/JavaScript files
echo "🔍 Running ESLint fix..."
npx eslint . --ext .ts,.tsx,.js,.jsx --fix

echo "✅ Lint issues fixed automatically!"
echo "📊 Summary:"
echo "- Markdown formatting fixed with markdownlint-cli2"
echo "- Code formatting fixed with prettier"
echo "- ESLint issues fixed automatically"
echo ""
echo "🎯 Next steps:"
echo "1. Run 'npm run lint' to verify fixes"
echo "2. Commit the formatted files"
echo "3. Continue with development"
