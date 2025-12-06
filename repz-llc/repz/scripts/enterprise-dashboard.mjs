#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🏢 Enterprise Monitoring Dashboard\n');

const dashboard = {
  timestamp: new Date().toISOString(),
  overview: {
    status: 'unknown',
    score: 0,
    alerts: 0,
    criticalIssues: 0
  },
  modules: {
    health: { status: 'unknown', score: 0, lastRun: null },
    security: { status: 'unknown', score: 0, lastRun: null },
    build: { status: 'unknown', score: 0, lastRun: null },
    documentation: { status: 'unknown', score: 0, lastRun: null },
    cleanup: { status: 'unknown', score: 0, lastRun: null }
  },
  metrics: {
    codebase: {},
    performance: {},
    security: {},
    compliance: {}
  },
  trends: [],
  recommendations: []
};

// 1. LOAD EXISTING REPORTS
console.log('📊 Loading existing reports...\n');

function loadReports() {
  const reports = {
    health: path.join(rootDir, 'health-report.json'),
    security: path.join(rootDir, 'security-report.json'),
    build: path.join(rootDir, 'build-optimization-report.json'),
    documentation: path.join(rootDir, 'documentation-quality-report.json'),
    cleanup: path.join(rootDir, 'enterprise-analysis-report.json')
  };
  
  Object.entries(reports).forEach(([type, reportPath]) => {
    if (fs.existsSync(reportPath)) {
      try {
        const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
        processReport(type, report);
        console.log(`✅ Loaded ${type} report`);
      } catch (error) {
        console.log(`⚠️  Failed to load ${type} report`);
        dashboard.modules[type].status = 'error';
      }
    } else {
      console.log(`❌ ${type} report not found`);
      dashboard.modules[type].status = 'missing';
    }
  });
  
  console.log();
}

function processReport(type, report) {
  dashboard.modules[type].lastRun = report.timestamp;
  
  switch (type) {
    case 'health':
      dashboard.modules[type].status = report.summary?.overallHealth || 'unknown';
      dashboard.modules[type].score = calculateHealthScore(report);
      dashboard.metrics.codebase = {
        files: report.codebase?.files || 0,
        lines: report.codebase?.lines || 0,
        components: report.codebase?.components || 0,
        features: report.codebase?.features || 0
      };
      dashboard.metrics.performance = {
        buildTime: report.performance?.buildTime || 0,
        bundleSize: report.performance?.bundleSize || 0
      };
      break;
      
    case 'security':
      dashboard.modules[type].status = report.summary?.securityScore > 80 ? 'healthy' : 
                                      report.summary?.securityScore > 60 ? 'warning' : 'critical';
      dashboard.modules[type].score = report.summary?.securityScore || 0;
      dashboard.metrics.security = {
        vulnerabilities: report.summary?.totalIssues || 0,
        criticalVulns: report.summary?.criticalVulns || 0,
        secrets: report.secrets?.length || 0
      };
      dashboard.metrics.compliance = report.compliance || {};
      break;
      
    case 'build':
      dashboard.modules[type].status = report.alerts?.filter(a => a.level === 'critical').length > 0 ? 'critical' :
                                      report.alerts?.filter(a => a.level === 'high').length > 0 ? 'warning' : 'healthy';
      dashboard.modules[type].score = calculateBuildScore(report);
      break;
      
    case 'documentation':
      const docScore = calculateDocScore(report);
      dashboard.modules[type].status = docScore > 80 ? 'healthy' : docScore > 60 ? 'warning' : 'critical';
      dashboard.modules[type].score = docScore;
      break;
      
    case 'cleanup':
      const cleanupScore = calculateCleanupScore(report);
      dashboard.modules[type].status = cleanupScore > 80 ? 'healthy' : cleanupScore > 60 ? 'warning' : 'critical';
      dashboard.modules[type].score = cleanupScore;
      break;
  }
}

function calculateHealthScore(report) {
  let score = 100;
  
  // Deduct for alerts
  const criticalAlerts = report.alerts?.filter(a => a.level === 'critical').length || 0;
  const highAlerts = report.alerts?.filter(a => a.level === 'high').length || 0;
  const warningAlerts = report.alerts?.filter(a => a.level === 'warning').length || 0;
  
  score -= (criticalAlerts * 30);
  score -= (highAlerts * 15);
  score -= (warningAlerts * 5);
  
  return Math.max(0, score);
}

function calculateBuildScore(report) {
  let score = 100;
  
  // Deduct for performance issues
  if (report.performance?.buildTime > 60) score -= 20;
  if (report.performance?.bundleSize > 5) score -= 15;
  if (report.performance?.firstLoadJs > 2) score -= 15;
  
  // Deduct for missing optimizations
  if (!report.optimization?.treeshaking?.enabled) score -= 10;
  if (!report.optimization?.codesplitting?.enabled) score -= 10;
  if (!report.optimization?.compression?.enabled) score -= 10;
  
  return Math.max(0, score);
}

function calculateDocScore(report) {
  const requiredFiles = report.summary?.requiredFiles || { existing: 0, total: 5 };
  const brokenLinks = report.summary?.qualityIssues?.brokenLinks || 0;
  
  let score = (requiredFiles.existing / requiredFiles.total) * 80;
  score -= (brokenLinks * 5);
  
  return Math.max(0, Math.min(100, score));
}

function calculateCleanupScore(report) {
  let score = 100;
  
  // Deduct for issues found
  score -= (report.summary?.ghostRoutes || 0) * 2;
  score -= (report.summary?.deadPages || 0) * 5;
  score -= (report.summary?.redundantFiles || 0) * 3;
  score -= (report.summary?.deprecatedComponents || 0) * 1;
  
  return Math.max(0, score);
}

loadReports();

// 2. CALCULATE OVERALL STATUS
console.log('📈 Calculating overall status...\n');

function calculateOverallStatus() {
  const scores = Object.values(dashboard.modules)
    .filter(module => module.status !== 'missing' && module.status !== 'error')
    .map(module => module.score);
  
  if (scores.length === 0) {
    dashboard.overview.status = 'unknown';
    dashboard.overview.score = 0;
    return;
  }
  
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  dashboard.overview.score = Math.round(avgScore);
  
  // Count critical issues across all modules
  const criticalModules = Object.values(dashboard.modules)
    .filter(module => module.status === 'critical').length;
  
  const warningModules = Object.values(dashboard.modules)
    .filter(module => module.status === 'warning').length;
  
  dashboard.overview.criticalIssues = criticalModules;
  dashboard.overview.alerts = criticalModules + warningModules;
  
  if (criticalModules > 0) {
    dashboard.overview.status = 'critical';
  } else if (warningModules > 0) {
    dashboard.overview.status = 'warning';
  } else if (avgScore > 80) {
    dashboard.overview.status = 'healthy';
  } else {
    dashboard.overview.status = 'warning';
  }
  
  console.log(`📊 Overall Status: ${dashboard.overview.status.toUpperCase()}`);
  console.log(`🎯 Overall Score: ${dashboard.overview.score}/100`);
  console.log(`🚨 Critical Issues: ${dashboard.overview.criticalIssues}`);
  console.log(`⚠️  Total Alerts: ${dashboard.overview.alerts}`);
  console.log();
}

calculateOverallStatus();

// 3. GENERATE SYSTEM RECOMMENDATIONS
console.log('💡 Generating system recommendations...\n');

function generateSystemRecommendations() {
  const recommendations = [];
  
  // Module-specific recommendations
  Object.entries(dashboard.modules).forEach(([module, status]) => {
    if (status.status === 'missing') {
      recommendations.push({
        priority: 'high',
        category: module,
        action: `Run ${module} analysis to establish baseline metrics`,
        command: `node scripts/enterprise-${module}-*.mjs`
      });
    } else if (status.status === 'critical') {
      recommendations.push({
        priority: 'critical',
        category: module,
        action: `Address critical issues in ${module} module immediately`,
        command: `node scripts/enterprise-${module}-*.mjs`
      });
    } else if (status.status === 'warning') {
      recommendations.push({
        priority: 'medium',
        category: module,
        action: `Review and resolve warnings in ${module} module`,
        command: `node scripts/enterprise-${module}-*.mjs`
      });
    }
  });
  
  // Security-specific recommendations
  if (dashboard.metrics.security?.criticalVulns > 0) {
    recommendations.push({
      priority: 'critical',
      category: 'security',
      action: 'Address critical security vulnerabilities immediately',
      command: 'npm audit fix --force'
    });
  }
  
  // Performance recommendations
  if (dashboard.metrics.performance?.buildTime > 60) {
    recommendations.push({
      priority: 'medium',
      category: 'performance',
      action: 'Optimize build performance to reduce deployment times',
      command: 'node scripts/enterprise-build-optimizer.mjs'
    });
  }
  
  // General recommendations
  recommendations.push({
    priority: 'low',
    category: 'monitoring',
    action: 'Set up automated daily monitoring with CI/CD integration',
    command: 'Setup GitHub Actions workflow for daily monitoring'
  });
  
  recommendations.push({
    priority: 'low',
    category: 'reporting',
    action: 'Configure alert notifications for critical issues',
    command: 'Setup Slack/email notifications for dashboard alerts'
  });
  
  dashboard.recommendations = recommendations
    .sort((a, b) => {
      const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  
  if (recommendations.length > 0) {
    console.log('💡 System Recommendations:');
    recommendations.forEach((rec, i) => {
      const priorityIcon = rec.priority === 'critical' ? '🔴' :
                          rec.priority === 'high' ? '🟠' :
                          rec.priority === 'medium' ? '🟡' : '🔵';
      console.log(`  ${priorityIcon} ${rec.priority.toUpperCase()}: ${rec.action}`);
      console.log(`     💻 Command: ${rec.command}\n`);
    });
  } else {
    console.log('✅ No immediate recommendations - system is well maintained!');
  }
  
  console.log();
}

generateSystemRecommendations();

// 4. CREATE MONITORING AUTOMATION
console.log('🤖 Creating monitoring automation...\n');

function createMonitoringAutomation() {
  const monitoringScript = `#!/bin/bash
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
#   # curl -X POST -H 'Content-type: application/json' \\
#   #   --data '{"text":"Enterprise monitoring complete. Check dashboard for details."}' \\
#   #   $SLACK_WEBHOOK_URL
#   
#   # Send email notification
#   # echo "Enterprise monitoring report attached" | mail -s "Daily Enterprise Report" \\
#   #   -A dashboard-report.json admin@company.com
# fi
`;

  fs.writeFileSync(path.join(rootDir, 'scripts/run-enterprise-monitoring.sh'), monitoringScript);
  
  try {
    fs.chmodSync(path.join(rootDir, 'scripts/run-enterprise-monitoring.sh'), '755');
  } catch (error) {
    // chmod might not work on all systems
  }
  
  // Create GitHub Actions workflow
  const githubWorkflow = `name: Enterprise Monitoring

on:
  schedule:
    - cron: '0 8 * * *' # Run daily at 8 AM UTC
  workflow_dispatch: # Allow manual trigger

jobs:
  enterprise-monitoring:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Run Enterprise Monitoring
        run: |
          chmod +x scripts/run-enterprise-monitoring.sh
          ./scripts/run-enterprise-monitoring.sh
        
      - name: Upload monitoring reports
        uses: actions/upload-artifact@v4
        with:
          name: enterprise-monitoring-reports
          path: |
            health-report.json
            security-report.json
            build-optimization-report.json
            documentation-quality-report.json
            dashboard-report.json
          retention-days: 30
          
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            if (fs.existsSync('dashboard-report.json')) {
              const dashboard = JSON.parse(fs.readFileSync('dashboard-report.json', 'utf8'));
              const status = dashboard.overview.status;
              const score = dashboard.overview.score;
              const alerts = dashboard.overview.alerts;
              
              const statusIcon = status === 'healthy' ? '✅' : 
                               status === 'warning' ? '⚠️' : '🔴';
              
              const comment = \`## 🏢 Enterprise Monitoring Report
              
\${statusIcon} **Overall Status**: \${status.toUpperCase()}
📊 **Health Score**: \${score}/100
🚨 **Active Alerts**: \${alerts}

### Module Status:
\${Object.entries(dashboard.modules).map(([module, data]) => 
  \`- \${module}: \${data.status} (\${data.score}/100)\`
).join('\\n')}

View detailed reports in the workflow artifacts.\`;
              
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: comment
              });
            }
`;
  
  const workflowDir = path.join(rootDir, '.github/workflows');
  if (!fs.existsSync(workflowDir)) {
    fs.mkdirSync(workflowDir, { recursive: true });
  }
  
  fs.writeFileSync(path.join(workflowDir, 'enterprise-monitoring.yml'), githubWorkflow);
  
  console.log('✅ Created monitoring automation scripts:');
  console.log('  • scripts/run-enterprise-monitoring.sh');
  console.log('  • .github/workflows/enterprise-monitoring.yml');
  console.log();
}

createMonitoringAutomation();

// 5. SAVE DASHBOARD REPORT
const dashboardReport = {
  ...dashboard,
  generatedBy: 'Enterprise Dashboard v1.0',
  nextRecommendedRun: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
  automationEnabled: fs.existsSync(path.join(rootDir, 'scripts/run-enterprise-monitoring.sh'))
};

fs.writeFileSync(
  path.join(rootDir, 'dashboard-report.json'),
  JSON.stringify(dashboardReport, null, 2)
);

// 6. DISPLAY ENTERPRISE DASHBOARD
console.log('🏢 Enterprise Monitoring Dashboard');
console.log('═'.repeat(80));
console.log(`📊 Overall Status: ${dashboard.overview.status.toUpperCase()}`);
console.log(`🎯 Health Score: ${dashboard.overview.score}/100`);
console.log(`🚨 Critical Issues: ${dashboard.overview.criticalIssues}`);
console.log(`⚠️  Active Alerts: ${dashboard.overview.alerts}`);
console.log(`🕐 Last Updated: ${new Date().toLocaleString()}`);
console.log('═'.repeat(80));

console.log('\n📋 Module Status:');
Object.entries(dashboard.modules).forEach(([module, data]) => {
  const statusIcon = data.status === 'healthy' ? '✅' :
                    data.status === 'warning' ? '⚠️' :
                    data.status === 'critical' ? '🔴' :
                    data.status === 'missing' ? '❌' : '⚪';
  
  const lastRun = data.lastRun ? new Date(data.lastRun).toLocaleDateString() : 'Never';
  console.log(`${statusIcon} ${module.padEnd(15)} | ${data.status.padEnd(10)} | Score: ${data.score.toString().padStart(3)}/100 | Last: ${lastRun}`);
});

console.log('\n📊 Key Metrics:');
if (dashboard.metrics.codebase.files) {
  console.log(`🗂️  Codebase: ${dashboard.metrics.codebase.files} files, ${dashboard.metrics.codebase.components} components`);
}
if (dashboard.metrics.performance.buildTime) {
  console.log(`⚡ Performance: ${dashboard.metrics.performance.buildTime}s build, ${dashboard.metrics.performance.bundleSize}MB bundle`);
}
if (dashboard.metrics.security.vulnerabilities !== undefined) {
  console.log(`🔒 Security: ${dashboard.metrics.security.vulnerabilities} vulnerabilities, ${dashboard.metrics.security.criticalVulns} critical`);
}

console.log('\n🚀 Quick Actions:');
console.log('  • Run full monitoring suite: ./scripts/run-enterprise-monitoring.sh');
console.log('  • View detailed reports: ls *-report.json');
console.log('  • Check GitHub Actions: .github/workflows/enterprise-monitoring.yml');

if (dashboard.recommendations.length > 0) {
  console.log('\n⚡ Top Priority Actions:');
  dashboard.recommendations.slice(0, 3).forEach((rec, i) => {
    const priorityIcon = rec.priority === 'critical' ? '🔴' :
                        rec.priority === 'high' ? '🟠' : '🟡';
    console.log(`  ${priorityIcon} ${rec.action}`);
  });
}

console.log('\n📄 Dashboard report saved to: dashboard-report.json');
console.log('\n🏢 Enterprise dashboard complete!');

// Exit with appropriate code
process.exit(dashboard.overview.criticalIssues > 0 ? 1 : 0);