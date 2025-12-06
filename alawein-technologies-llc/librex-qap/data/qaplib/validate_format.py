#!/usr/bin/env python3
"""
Quick validation script for QAPLIB format files
"""

import sys
from pathlib import Path

def validate_qaplib_file(filepath):
    """Validate a QAPLIB format file"""
    print(f"\nValidating: {filepath.name}")
    print("-" * 50)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse problem size
    n = int(lines[0].strip())
    print(f"Problem size (n): {n}")

    # Expected structure: 1 (size) + 1 (blank) + n (flow) + 1 (blank) + n (distance)
    expected_lines = 1 + 1 + n + 1 + n
    actual_lines = len(lines)

    # Count non-blank lines
    non_blank = sum(1 for line in lines if line.strip())
    expected_non_blank = 1 + n + n  # size + flow matrix + distance matrix

    print(f"Total lines: {actual_lines} (expected: ~{expected_lines})")
    print(f"Non-blank lines: {non_blank} (expected: {expected_non_blank})")

    # Parse flow matrix (starts at line 2, after blank line at line 1)
    flow_matrix = []
    idx = 2  # Skip size line and blank line
    for i in range(n):
        row = [int(x) for x in lines[idx + i].split()]
        if len(row) != n:
            print(f"ERROR: Flow matrix row {i} has {len(row)} elements, expected {n}")
            return False
        flow_matrix.append(row)

    print(f"Flow matrix: {n}×{n} ✓")

    # Parse distance matrix (starts after flow matrix + blank line)
    dist_matrix = []
    idx = 2 + n + 1  # Skip size, blank, flow matrix, and another blank
    for i in range(n):
        row = [int(x) for x in lines[idx + i].split()]
        if len(row) != n:
            print(f"ERROR: Distance matrix row {i} has {len(row)} elements, expected {n}")
            return False
        dist_matrix.append(row)

    print(f"Distance matrix: {n}×{n} ✓")
    print("Format validation: PASSED ✓")

    return True

def main():
    qaplib_dir = Path("/home/user/QAP-CLAUDE-CODE/data/qaplib")
    dat_files = sorted(qaplib_dir.glob("*.dat"))

    print("=" * 50)
    print("QAPLIB Format Validation")
    print("=" * 50)

    all_valid = True
    for filepath in dat_files:
        try:
            if not validate_qaplib_file(filepath):
                all_valid = False
        except Exception as e:
            print(f"ERROR validating {filepath.name}: {e}")
            all_valid = False

    print("\n" + "=" * 50)
    if all_valid:
        print("OVERALL: ALL FILES VALID ✓")
    else:
        print("OVERALL: SOME FILES FAILED VALIDATION ✗")
    print("=" * 50)

    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())
