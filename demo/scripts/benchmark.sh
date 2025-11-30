#!/bin/bash

# ATLAS Performance Benchmark Script
# Measures analysis performance across different repository sizes and types

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$DEMO_DIR/test-repos"
LOG_DIR="$DEMO_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BENCHMARK_FILE="$LOG_DIR/benchmark_$TIMESTAMP.csv"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Initialize benchmark file
echo "repository,operation,duration_seconds,timestamp" > "$BENCHMARK_FILE"

# Benchmark function
benchmark_operation() {
    local repo="$1"
    local operation="$2"
    local command="$3"

    echo -e "${BLUE}Benchmarking $operation on $repo...${NC}"

    local start_time=$(date +%s.%3N)
    if eval "$command" > /dev/null 2>&1; then
        local end_time=$(date +%s.%3N)
        local duration=$(echo "$end_time - $start_time" | bc)
        echo "$repo,$operation,$duration,$TIMESTAMP" >> "$BENCHMARK_FILE"
        echo -e "${GREEN}✓ $operation: ${duration}s${NC}"
    else
        echo "ERROR: $operation failed on $repo" >&2
    fi
}

# Main benchmark workflow
main() {
    echo "ATLAS Performance Benchmark"
    echo "=========================="

    # Create test data if needed
    if [ ! -f "$REPO_DIR/messy-python/data.json" ]; then
        echo '{"items": []}' > "$REPO_DIR/messy-python/data.json"
    fi

    # Benchmark each repository
    for repo in "messy-python" "complex-js" "spaghetti-ts"; do
        repo_path="$REPO_DIR/$repo"

        # Repository analysis
        benchmark_operation "$repo" "repo_analysis" "atlas analyze repo '$repo_path' --format json"

        # Complexity analysis
        benchmark_operation "$repo" "complexity_analysis" "atlas analyze complexity '$repo_path'"

        # Chaos analysis
        benchmark_operation "$repo" "chaos_analysis" "atlas analyze chaos '$repo_path'"

        # Quick scan
        benchmark_operation "$repo" "quick_scan" "atlas analyze scan '$repo_path'"
    done

    echo ""
    echo "Benchmark complete!"
    echo "Results saved to: $BENCHMARK_FILE"

    # Generate summary
    echo ""
    echo "Summary:"
    echo "--------"
    awk -F',' '
        NR > 1 {
            repo = $1
            operation = $2
            duration = $3
            sum[repo] += duration
            count[repo]++
            if (operation == "repo_analysis") {
                repo_times[repo] = duration
            }
        }
        END {
            for (repo in sum) {
                avg = sum[repo] / count[repo]
                printf "%s: Total %.3fs, Average %.3fs, Repo Analysis %.3fs\n",
                    repo, sum[repo], avg, repo_times[repo]
            }
        }
    ' "$BENCHMARK_FILE"
}

main "$@"