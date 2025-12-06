#!/bin/bash

# Performance Optimization Script
# Runs comprehensive optimization for all applications

set -e

echo "🚀 Starting Performance Optimization Suite"
echo "========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APPS=("scientific-tinder" "chaos-engine" "ghost-researcher")
BASE_DIR="/home/user/CrazyIdeas"
OPTIMIZATION_DIR="$BASE_DIR/optimization"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run bundle analysis
analyze_bundle() {
    local app=$1
    print_status "Analyzing bundle for $app..."

    cd "$BASE_DIR/frontend/$app"

    # Build the application
    npm run build

    # Analyze bundle size
    ANALYZE=true npm run build 2>&1 | tee "$OPTIMIZATION_DIR/bundle-analysis/reports/$app-bundle.log"

    # Generate bundle stats
    npx webpack-bundle-analyzer .next/analyze/client.html \
        --mode static \
        --report "$OPTIMIZATION_DIR/bundle-analysis/reports/$app-bundle-report.html" \
        --no-open

    print_success "Bundle analysis complete for $app"
}

# Function to optimize dependencies
optimize_dependencies() {
    local app=$1
    print_status "Optimizing dependencies for $app..."

    cd "$BASE_DIR/frontend/$app"

    # Remove unused dependencies
    npx depcheck --json > "$OPTIMIZATION_DIR/bundle-analysis/reports/$app-depcheck.json"

    # Deduplicate dependencies
    npm dedupe

    # Update lock file
    npm install

    print_success "Dependencies optimized for $app"
}

# Function to optimize images
optimize_images() {
    local app=$1
    print_status "Optimizing images for $app..."

    cd "$BASE_DIR/frontend/$app/public"

    # Optimize PNG images
    find . -name "*.png" -exec npx imagemin {} --out-dir=. --plugin=pngquant \;

    # Optimize JPEG images
    find . -name "*.jpg" -o -name "*.jpeg" -exec npx imagemin {} --out-dir=. --plugin=mozjpeg \;

    # Convert to WebP
    for img in *.{jpg,jpeg,png}; do
        if [ -f "$img" ]; then
            npx imagemin "$img" --out-dir=. --plugin=webp
        fi
    done

    print_success "Images optimized for $app"
}

# Function to run Lighthouse audit
run_lighthouse() {
    local app=$1
    local port=$2
    print_status "Running Lighthouse audit for $app..."

    # Start the app in background
    cd "$BASE_DIR/frontend/$app"
    npm run start &
    local pid=$!

    # Wait for server to start
    sleep 10

    # Run Lighthouse
    npx lighthouse "http://localhost:$port" \
        --output html \
        --output-path "$OPTIMIZATION_DIR/continuous-improvement/reports/$app-lighthouse.html" \
        --chrome-flags="--headless"

    # Stop the app
    kill $pid

    print_success "Lighthouse audit complete for $app"
}

# Function to analyze performance metrics
analyze_metrics() {
    local app=$1
    print_status "Analyzing performance metrics for $app..."

    # Create metrics report
    cat > "$OPTIMIZATION_DIR/continuous-improvement/reports/$app-metrics.json" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "application": "$app",
  "metrics": {
    "bundleSize": $(du -sk "$BASE_DIR/frontend/$app/.next" | cut -f1),
    "nodeModulesSize": $(du -sk "$BASE_DIR/frontend/$app/node_modules" | cut -f1),
    "publicAssetsSize": $(du -sk "$BASE_DIR/frontend/$app/public" | cut -f1)
  }
}
EOF

    print_success "Metrics analysis complete for $app"
}

# Function to generate optimization report
generate_report() {
    local app=$1
    print_status "Generating optimization report for $app..."

    local report_file="$OPTIMIZATION_DIR/docs/$app-optimization-report.md"

    cat > "$report_file" <<EOF
# Optimization Report: $app

Generated: $(date)

## Bundle Analysis

- Total Bundle Size: $(du -sh "$BASE_DIR/frontend/$app/.next" | cut -f1)
- Node Modules Size: $(du -sh "$BASE_DIR/frontend/$app/node_modules" | cut -f1)
- Public Assets Size: $(du -sh "$BASE_DIR/frontend/$app/public" | cut -f1)

## Recommendations

1. **Code Splitting**
   - Implement dynamic imports for heavy components
   - Lazy load routes that are not immediately needed
   - Split vendor bundles by frequency of change

2. **Image Optimization**
   - Convert images to WebP/AVIF format
   - Implement responsive images with srcset
   - Use next/image for automatic optimization

3. **Caching Strategy**
   - Implement service worker for offline support
   - Configure proper cache headers
   - Use stale-while-revalidate pattern

4. **Performance Monitoring**
   - Set up Core Web Vitals tracking
   - Monitor bundle size in CI/CD
   - Track performance regressions

## Next Steps

- [ ] Review bundle analysis report
- [ ] Implement recommended optimizations
- [ ] Re-run performance audit
- [ ] Deploy optimizations to production

EOF

    print_success "Report generated: $report_file"
}

# Main execution
main() {
    print_status "Creating report directories..."
    mkdir -p "$OPTIMIZATION_DIR/bundle-analysis/reports"
    mkdir -p "$OPTIMIZATION_DIR/continuous-improvement/reports"
    mkdir -p "$OPTIMIZATION_DIR/docs"

    # Process each application
    for i in "${!APPS[@]}"; do
        app="${APPS[$i]}"
        port=$((3000 + i))

        echo ""
        echo "========================================="
        echo "Processing: $app"
        echo "========================================="

        # Check if app directory exists
        if [ ! -d "$BASE_DIR/frontend/$app" ]; then
            print_error "Application directory not found: $app"
            continue
        fi

        # Run optimization tasks
        analyze_bundle "$app"
        optimize_dependencies "$app"
        # optimize_images "$app"  # Commented out as it requires imagemin
        analyze_metrics "$app"
        # run_lighthouse "$app" "$port"  # Commented out as it requires running server
        generate_report "$app"

        print_success "Optimization complete for $app"
    done

    echo ""
    echo "========================================="
    print_success "All optimizations complete!"
    echo "Reports available in: $OPTIMIZATION_DIR"
    echo "========================================="
}

# Run main function
main "$@"