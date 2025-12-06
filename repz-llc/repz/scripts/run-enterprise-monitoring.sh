#!/bin/bash
# Enterprise Monitoring Automation Script
# Run this daily to maintain system health

echo "🏢 Starting Enterprise Monitoring Suite..."
echo "================================================"

# Set error handling
set -e

# Change to project directory
cd "$(dirname "$0")/.."

echo "📊 Running Health Monitor..."
node scripts/enterprise-health-monitor.mjs || echo "⚠️  Health monitor completed with warnings"

echo ""
echo "🔒 Running Security Scanner..." 
node scripts/enterprise-security-scanner.mjs || echo "⚠️  Security scanner completed with warnings"

echo ""
echo "⚡ Running Build Optimizer..."
node scripts/enterprise-build-optimizer.mjs || echo "⚠️  Build optimizer completed with warnings"

echo ""
echo "📚 Running Documentation Standardizer..."
node scripts/documentation-standardizer.mjs || echo "⚠️  Documentation standardizer completed with warnings"

echo ""
echo "🏢 Generating Enterprise Dashboard..."
node scripts/enterprise-dashboard.mjs

echo ""
echo "================================================"
echo "✅ Enterprise monitoring complete!"
echo "📄 Check dashboard-report.json for detailed results"

# Optional: Send notifications (uncomment to enable)
# if [ -f "dashboard-report.json" ]; then
#   # Send Slack notification
#   # curl -X POST -H 'Content-type: application/json' \
#   #   --data '{"text":"Enterprise monitoring complete. Check dashboard for details."}' \
#   #   $SLACK_WEBHOOK_URL
#   
#   # Send email notification
#   # echo "Enterprise monitoring report attached" | mail -s "Daily Enterprise Report" \
#   #   -A dashboard-report.json admin@company.com
# fi
