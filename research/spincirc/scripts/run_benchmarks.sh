#!/bin/bash
# SpinCirc Benchmarks - Hub-Spoke Pattern
set -e
cd "$(dirname "$0")/.."
../../../.metaHub/clis/bench config config/benchmarks.yaml
echo "✅ SpinCirc benchmarks complete!"
