#!/usr/bin/env node

/**
 * ATLAS Dashboard Generator
 * Creates visual HTML dashboards from analysis metrics
 */

const fs = require('fs');
const path = require('path');

class DashboardGenerator {
  constructor(dashboardDir = null) {
    this.dashboardDir = dashboardDir || path.join(__dirname, '..', 'dashboards');
    this.ensureDashboardDir();
  }

  ensureDashboardDir() {
    if (!fs.existsSync(this.dashboardDir)) {
      fs.mkdirSync(this.dashboardDir, { recursive: true });
    }
  }

  generateHTMLDashboard(metricsFile, outputFile = null) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputPath = outputFile || path.join(this.dashboardDir, `dashboard-${timestamp}.html`);

    // Read metrics data
    let metricsData = {};
    if (fs.existsSync(metricsFile)) {
      try {
        metricsData = JSON.parse(fs.readFileSync(metricsFile, 'utf8'));
      } catch (e) {
        console.error('Error reading metrics file:', e.message);
        return null;
      }
    }

    const html = this.createHTML(metricsData, timestamp);
    fs.writeFileSync(outputPath, html);

    console.log(`Dashboard generated: ${outputPath}`);
    return outputPath;
  }

  createHTML(metricsData, timestamp) {
    const repositories = metricsData.repositories || {};
    const benchmarks = metricsData.benchmarks || {};

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATLAS Demo Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #007bff;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.2em;
        }
        .metric-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 5px 0;
        }
        .metric-label {
            font-weight: 500;
            color: #666;
        }
        .metric-value {
            font-weight: 600;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-bad { color: #dc3545; }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .chart-container h3 {
            margin-top: 0;
            color: #333;
            text-align: center;
        }
        .performance-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .performance-table th,
        .performance-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .performance-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        .performance-table tr:hover {
            background: #f8f9fa;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            border-top: 1px solid #eee;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ATLAS Demo Dashboard</h1>
            <p>Comprehensive Code Analysis & Optimization Results</p>
        </div>

        <div class="content">
            <div class="metrics-grid" id="metrics"></div>

            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Complexity Scores</h3>
                    <canvas id="complexityChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Analysis Performance</h3>
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>

            <div class="chart-container">
                <h3>Performance Benchmarks</h3>
                <table class="performance-table">
                    <thead>
                        <tr>
                            <th>Repository</th>
                            <th>Analysis Time</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="performanceTable"></tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            Generated on ${new Date().toLocaleString()} | ATLAS Demo Environment
        </div>
    </div>

    <script>
        // Dashboard data
        const metricsData = ${JSON.stringify(metricsData)};

        // Populate metrics cards
        const metricsDiv = document.getElementById('metrics');
        const repositories = metricsData.repositories || {};

        Object.entries(repositories).forEach(([repo, metrics]) => {
            const card = document.createElement('div');
            card.className = 'metric-card';

            const complexityClass = getStatusClass(metrics.complexityScore);
            const chaosClass = getStatusClass(metrics.chaosLevel);
            const maintainabilityClass = getStatusClass(metrics.maintainabilityIndex, true);

            card.innerHTML = \`
                <h3>\${repo.replace('-', ' ').toUpperCase()}</h3>
                <div class="metric-item">
                    <span class="metric-label">Files Analyzed:</span>
                    <span class="metric-value">\${metrics.filesAnalyzed || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Total Lines:</span>
                    <span class="metric-value">\${metrics.totalLines || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Complexity Score:</span>
                    <span class="metric-value \${complexityClass}">\${(metrics.complexityScore || 0).toFixed(2)}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Chaos Level:</span>
                    <span class="metric-value \${chaosClass}">\${(metrics.chaosLevel || 0).toFixed(2)}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Maintainability:</span>
                    <span class="metric-value \${maintainabilityClass}">\${(metrics.maintainabilityIndex || 0).toFixed(2)}</span>
                </div>
            \`;

            metricsDiv.appendChild(card);
        });

        // Complexity Chart
        const ctx = document.getElementById('complexityChart').getContext('2d');
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: Object.keys(repositories),
                datasets: [{
                    label: 'Complexity Score',
                    data: Object.values(repositories).map(m => m.complexityScore || 0),
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(54, 162, 235, 1)'
                }, {
                    label: 'Chaos Level',
                    data: Object.values(repositories).map(m => m.chaosLevel || 0),
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(255, 99, 132, 1)'
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 10
                    }
                }
            }
        });

        // Performance Chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        const benchmarks = metricsData.benchmarks || {};
        new Chart(perfCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(benchmarks),
                datasets: [{
                    label: 'Analysis Time (seconds)',
                    data: Object.values(benchmarks),
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // Performance Table
        const tableBody = document.getElementById('performanceTable');
        Object.entries(benchmarks).forEach(([repo, time]) => {
            const row = document.createElement('tr');
            row.innerHTML = \`
                <td>\${repo}</td>
                <td>\${time}</td>
                <td><span class="status-good">✓ Completed</span></td>
            \`;
            tableBody.appendChild(row);
        });

        function getStatusClass(value, invert = false) {
            if (value === undefined || value === null) return 'status-warning';
            if (invert) {
                return value >= 8 ? 'status-good' : value >= 6 ? 'status-warning' : 'status-bad';
            }
            return value <= 2 ? 'status-good' : value <= 5 ? 'status-warning' : 'status-bad';
        }
    </script>
</body>
</html>`;
  }

  generateComparisonDashboard(beforeFile, afterFile, outputFile = null) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputPath = outputFile || path.join(this.dashboardDir, `comparison-${timestamp}.html`);

    let beforeData = {};
    let afterData = {};

    if (fs.existsSync(beforeFile)) {
      try {
        beforeData = JSON.parse(fs.readFileSync(beforeFile, 'utf8'));
      } catch (e) {
        console.error('Error reading before file:', e.message);
      }
    }

    if (fs.existsSync(afterFile)) {
      try {
        afterData = JSON.parse(fs.readFileSync(afterFile, 'utf8'));
      } catch (e) {
        console.error('Error reading after file:', e.message);
      }
    }

    const html = this.createComparisonHTML(beforeData, afterData, timestamp);
    fs.writeFileSync(outputPath, html);

    console.log(`Comparison dashboard generated: ${outputPath}`);
    return outputPath;
  }

  createComparisonHTML(beforeData, afterData, timestamp) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATLAS Before/After Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .metric-card h3 { margin: 0 0 10px 0; color: #333; }
        .metric-value { font-size: 1.5em; font-weight: bold; }
        .improvement { color: #28a745; }
        .decline { color: #dc3545; }
        .chart-container { margin: 20px 0; padding: 20px; background: white; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ATLAS Before/After Comparison</h1>
        <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>

        <div class="comparison-grid" id="comparison"></div>

        <div class="chart-container">
            <canvas id="comparisonChart"></canvas>
        </div>
    </div>

    <script>
        const beforeData = ${JSON.stringify(beforeData)};
        const afterData = ${JSON.stringify(afterData)};

        // Populate comparison
        const comparisonDiv = document.getElementById('comparison');

        Object.keys(beforeData.repositories || {}).forEach(repo => {
            const before = beforeData.repositories[repo] || {};
            const after = afterData.repositories[repo] || {};

            const complexityDiff = ((after.complexityScore || 0) - (before.complexityScore || 0)).toFixed(2);
            const chaosDiff = ((after.chaosLevel || 0) - (before.chaosLevel || 0)).toFixed(2);
            const maintainabilityDiff = ((after.maintainabilityIndex || 0) - (before.maintainabilityIndex || 0)).toFixed(2);

            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = \`
                <h3>\${repo.replace('-', ' ').toUpperCase()}</h3>
                <p>Complexity: \${before.complexityScore?.toFixed(2) || 0} → \${after.complexityScore?.toFixed(2) || 0}
                   <span class="\${complexityDiff < 0 ? 'improvement' : 'decline'}">(\${complexityDiff})</span></p>
                <p>Chaos Level: \${before.chaosLevel?.toFixed(2) || 0} → \${after.chaosLevel?.toFixed(2) || 0}
                   <span class="\${chaosDiff < 0 ? 'improvement' : 'decline'}">(\${chaosDiff})</span></p>
                <p>Maintainability: \${before.maintainabilityIndex?.toFixed(2) || 0} → \${after.maintainabilityIndex?.toFixed(2) || 0}
                   <span class="\${maintainabilityDiff > 0 ? 'improvement' : 'decline'}">(\${maintainabilityDiff})</span></p>
            \`;

            comparisonDiv.appendChild(card);
        });

        // Comparison Chart
        const ctx = document.getElementById('comparisonChart').getContext('2d');
        const repos = Object.keys(beforeData.repositories || {});

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: repos,
                datasets: [{
                    label: 'Before Complexity',
                    data: repos.map(r => beforeData.repositories[r]?.complexityScore || 0),
                    backgroundColor: 'rgba(255, 99, 132, 0.5)'
                }, {
                    label: 'After Complexity',
                    data: repos.map(r => afterData.repositories[r]?.complexityScore || 0),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                }]
            },
            options: {
                scales: { y: { beginAtZero: true } }
            }
        });
    </script>
</body>
</html>`;
  }
}

module.exports = DashboardGenerator;

// CLI usage
if (require.main === module) {
  const generator = new DashboardGenerator();

  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.log('Usage:');
    console.log('  node generate-dashboard.js <metrics-file> [output-file]');
    console.log('  node generate-dashboard.js compare <before-file> <after-file> [output-file]');
    process.exit(1);
  }

  if (args[0] === 'compare') {
    generator.generateComparisonDashboard(args[1], args[2], args[3]);
  } else {
    generator.generateHTMLDashboard(args[0], args[1]);
  }
}
