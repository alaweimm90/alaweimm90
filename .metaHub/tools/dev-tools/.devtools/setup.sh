#!/bin/bash
# Centralized devtools setup script

DEVTOOLS_DIR=".devtools"
TOOLS=("amazonq" "cursor" "continue" "windsurf" "cline")

echo "🚀 Setting up centralized devtools..."
echo ""

# Create symlinks for each tool
for tool in "${TOOLS[@]}"; do
  echo "Setting up .$tool..."
  mkdir -p ".$tool"
  
  # Link rules
  if [ -L ".$tool/rules" ]; then
    echo "  ⚠️  Rules already linked for $tool"
  else
    ln -sf "../$DEVTOOLS_DIR/rules" ".$tool/rules" 2>/dev/null || \
    cmd //c "mklink /D .$tool\\rules ..\\$DEVTOOLS_DIR\\rules" > /dev/null 2>&1
    echo "  ✓ Linked rules for $tool"
  fi
  
  # Link integrations
  if [ -L ".$tool/integrations" ]; then
    echo "  ⚠️  Integrations already linked for $tool"
  else
    ln -sf "../$DEVTOOLS_DIR/integrations" ".$tool/integrations" 2>/dev/null || \
    cmd //c "mklink /D .$tool\\integrations ..\\$DEVTOOLS_DIR\\integrations" > /dev/null 2>&1
    echo "  ✓ Linked integrations for $tool"
  fi
  
  echo ""
done

echo "✅ Centralized devtools setup complete!"
echo ""
echo "📁 Structure created:"
echo "   .devtools/          (source of truth)"
echo "   ├── rules/          (5 rule files)"
echo "   ├── mcps/           (2 config files)"
echo "   └── integrations/   (5 config files)"
echo ""
echo "🔗 Symlinks created for: ${TOOLS[*]}"
echo ""
echo "📖 See .devtools/README.md for usage"
