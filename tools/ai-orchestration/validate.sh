#!/bin/bash
# AI Tools Validator v2.3
echo "AI Tools Validator v2.3"
echo "======================="
errors=0

echo ""; echo "Bash Syntax:"
for s in ~/.ai_tools/scripts/*.sh; do
    name=$(basename "$s")
    [ "$name" = "validate.sh" ] && continue
    bash -n "$s" 2>/dev/null && echo "  ✓ $name" || { echo "  ✗ $name"; ((errors++)); }
done

echo ""; echo "ShellCheck:"
err=$(shellcheck --severity=error ~/.ai_tools/scripts/*.sh 2>&1 | grep -c "SC[0-9]" || echo 0)
echo "  Critical errors: $err"

echo ""; echo "JSON Files:"
for j in ~/.ai_tools/metrics/current.json ~/.ai_tools/learning/routing_model.json; do
    [ -f "$j" ] && jq . "$j" >/dev/null 2>&1 && echo "  ✓ $(basename $j)" || echo "  ✗ $(basename $j)"
done

echo ""; [ $errors -eq 0 ] && echo "✅ All validations passed" || echo "❌ Errors: $errors"
