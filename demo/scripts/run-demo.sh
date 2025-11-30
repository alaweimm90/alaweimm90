#!/bin/bash

# ATLAS Comprehensive Demo Script
# This script demonstrates the complete ATLAS workflow:
# 1. Repository analysis
# 2. Code quality assessment
# 3. Complexity analysis
# 4. Chaos analysis
# 5. Performance benchmarking
# 6. Before/after comparisons

set -e

# Configuration
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$DEMO_DIR/test-repos"
LOG_DIR="$DEMO_DIR/logs"
DASHBOARD_DIR="$DEMO_DIR/dashboards"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/demo_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $*" | tee -a "$LOG_FILE"
}

# Header function
header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    log "Starting: $1"
}

# Success function
success() {
    echo -e "${GREEN}✓ $1${NC}"
    log "SUCCESS: $1"
}

# Error function
error() {
    echo -e "${RED}✗ $1${NC}"
    log "ERROR: $1"
}

# Warning function
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    log "WARNING: $1"
}

# Setup function
setup() {
    header "Setting up ATLAS Demo Environment"

    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"

    # Check if ATLAS CLI is available
    if ! command -v atlas &> /dev/null; then
        error "ATLAS CLI not found. Please ensure ATLAS is properly installed."
        exit 1
    fi

    success "ATLAS CLI found"

    # Check test repositories
    if [ ! -d "$REPO_DIR/messy-python" ] || [ ! -d "$REPO_DIR/complex-js" ] || [ ! -d "$REPO_DIR/spaghetti-ts" ]; then
        error "Test repositories not found. Please run setup first."
        exit 1
    fi

    success "Test repositories verified"
}

# Run analysis on a repository
analyze_repo() {
    local repo_name="$1"
    local repo_path="$REPO_DIR/$repo_name"

    header "Analyzing $repo_name Repository"

    log "Running repository analysis on $repo_path"

    # Full repository analysis
    if atlas analyze repo "$repo_path" --format json > "$LOG_DIR/${repo_name}_analysis_$TIMESTAMP.json" 2>> "$LOG_FILE"; then
        success "Repository analysis completed for $repo_name"
    else
        error "Repository analysis failed for $repo_name"
        return 1
    fi

    # Complexity analysis
    if atlas analyze complexity "$repo_path" > "$LOG_DIR/${repo_name}_complexity_$TIMESTAMP.txt" 2>> "$LOG_FILE"; then
        success "Complexity analysis completed for $repo_name"
    else
        warning "Complexity analysis failed for $repo_name"
    fi

    # Chaos analysis
    if atlas analyze chaos "$repo_path" --detailed > "$LOG_DIR/${repo_name}_chaos_$TIMESTAMP.txt" 2>> "$LOG_FILE"; then
        success "Chaos analysis completed for $repo_name"
    else
        warning "Chaos analysis failed for $repo_name"
    fi

    # Quick scan
    if atlas analyze scan "$repo_path" > "$LOG_DIR/${repo_name}_scan_$TIMESTAMP.txt" 2>> "$LOG_FILE"; then
        success "Quick scan completed for $repo_name"
    else
        warning "Quick scan failed for $repo_name"
    fi
}

# Generate performance benchmarks
benchmark_performance() {
    header "Running Performance Benchmarks"

    log "Starting performance benchmarking"

    # Measure analysis time for each repository
    for repo in "messy-python" "complex-js" "spaghetti-ts"; do
        local repo_path="$REPO_DIR/$repo"
        local start_time=$(date +%s.%3N)

        if atlas analyze repo "$repo_path" --format json > /dev/null 2>> "$LOG_FILE"; then
            local end_time=$(date +%s.%3N)
            local duration=$(echo "$end_time - $start_time" | bc)
            echo "$repo: ${duration}s" >> "$LOG_DIR/benchmark_$TIMESTAMP.txt"
            success "Benchmarked $repo: ${duration}s"
        else
            error "Benchmark failed for $repo"
        fi
    done
}

# Generate dashboard data
generate_dashboard() {
    header "Generating Dashboard Data"

    log "Creating dashboard metrics"

    # Collect all analysis results
    local dashboard_data="{
  \"timestamp\": \"$TIMESTAMP\",
  \"repositories\": {
    \"messy-python\": $(cat \"$LOG_DIR/messy-python_analysis_$TIMESTAMP.json\" 2>/dev/null || echo '{}'),
    \"complex-js\": $(cat \"$LOG_DIR/complex-js_analysis_$TIMESTAMP.json\" 2>/dev/null || echo '{}'),
    \"spaghetti-ts\": $(cat \"$LOG_DIR/spaghetti-ts_analysis_$TIMESTAMP.json\" 2>/dev/null || echo '{}')
  },
  \"benchmarks\": $(cat \"$LOG_DIR/benchmark_$TIMESTAMP.txt\" 2>/dev/null | jq -R -s 'split(\"\n\") | map(select(. != \"\")) | map(split(\": \") | {(.[0]): .[1]}) | add' 2>/dev/null || echo '{}')
}"

    echo "$dashboard_data" > "$DASHBOARD_DIR/metrics_$TIMESTAMP.json"
    success "Dashboard data generated"
}

# Generate HTML dashboard
generate_html_dashboard() {
    header "Generating HTML Dashboard"

    local html_file="$DASHBOARD_DIR/dashboard_$TIMESTAMP.html"

    cat > "$html_file" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATLAS Demo Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .metric-card { background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 5px; border-left: 4px solid #007bff; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .chart-container { margin: 20px 0; padding: 20px; background: white; border-radius: 5px; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-bad { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ATLAS Demo Dashboard</h1>
        <p><strong>Generated:</strong> <span id="timestamp"></span></p>

        <div class="metric-grid" id="metrics"></div>

        <div class="chart-container">
            <canvas id="complexityChart"></canvas>
        </div>

        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>
    </div>

    <script>
        // Load dashboard data
        fetch('./metrics_${TIMESTAMP}.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('timestamp').textContent = new Date(data.timestamp * 1000).toLocaleString();

                // Display metrics
                const metricsDiv = document.getElementById('metrics');
                Object.entries(data.repositories).forEach(([repo, metrics]) => {
                    const card = document.createElement('div');
                    card.className = 'metric-card';
                    card.innerHTML = `
                        <h3>${repo.replace('-', ' ').toUpperCase()}</h3>
                        <p><strong>Files:</strong> ${metrics.filesAnalyzed || 0}</p>
                        <p><strong>Lines:</strong> ${metrics.totalLines || 0}</p>
                        <p><strong>Complexity:</strong> <span class="${getStatusClass(metrics.complexityScore)}">${(metrics.complexityScore || 0).toFixed(2)}</span></p>
                        <p><strong>Chaos Level:</strong> <span class="${getStatusClass(metrics.chaosLevel)}">${(metrics.chaosLevel || 0).toFixed(2)}</span></p>
                        <p><strong>Maintainability:</strong> <span class="${getStatusClass(metrics.maintainabilityIndex, true)}">${(metrics.maintainabilityIndex || 0).toFixed(2)}</span></p>
                    `;
                    metricsDiv.appendChild(card);
                });

                // Complexity Chart
                const ctx = document.getElementById('complexityChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(data.repositories),
                        datasets: [{
                            label: 'Complexity Score',
                            data: Object.values(data.repositories).map(m => m.complexityScore || 0),
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });

                // Performance Chart
                const perfCtx = document.getElementById('performanceChart').getContext('2d');
                new Chart(perfCtx, {
                    type: 'line',
                    data: {
                        labels: Object.keys(data.benchmarks || {}),
                        datasets: [{
                            label: 'Analysis Time (seconds)',
                            data: Object.values(data.benchmarks || {}),
                            backgroundColor: 'rgba(255, 99, 132, 0.5)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2,
                            fill: false
                        }]
                    },
                    options: {
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
            })
            .catch(error => console.error('Error loading dashboard data:', error));

        function getStatusClass(value, invert = false) {
            if (value === undefined || value === null) return 'status-warning';
            if (invert) {
                return value >= 8 ? 'status-good' : value >= 6 ? 'status-warning' : 'status-bad';
            }
            return value <= 2 ? 'status-good' : value <= 5 ? 'status-warning' : 'status-bad';
        }
    </script>
</body>
</html>
EOF

    success "HTML dashboard generated: $html_file"
}

# Main demo workflow
main() {
    header "ATLAS Comprehensive Demo"
    log "Starting ATLAS demo workflow"

    # Setup
    setup

    # Analyze each test repository
    for repo in "messy-python" "complex-js" "spaghetti-ts"; do
        analyze_repo "$repo"
    done

    # Performance benchmarking
    benchmark_performance

    # Generate dashboard
    generate_dashboard
    generate_html_dashboard

    # Final summary
    header "Demo Complete"
    success "ATLAS demo workflow completed successfully"
    echo ""
    echo -e "${BLUE}Results:${NC}"
    echo "  Logs: $LOG_DIR/"
    echo "  Dashboard: $DASHBOARD_DIR/dashboard_$TIMESTAMP.html"
    echo "  Metrics: $DASHBOARD_DIR/metrics_$TIMESTAMP.json"
    echo ""
    echo -e "${GREEN}Open the HTML dashboard in your browser to view the results!${NC}"

    log "Demo workflow completed"
}

# Run main function
main "$@"