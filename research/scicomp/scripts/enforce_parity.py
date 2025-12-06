#!/usr/bin/env python3
"""
SciComp Directory Structure Enforcement Script
Ensures all platforms have identical subject directories with required files.
"""
import os
from pathlib import Path
from typing import List, Dict
# Canonical subject directories
SUBJECTS = [
    'Control',
    'Crystallography',
    'Elasticity',
    'FEM',
    'Linear_Algebra',
    'Machine_Learning',
    'Monte_Carlo',
    'Multiphysics',
    'ODE_PDE',
    'Optics',
    'Optimization',
    'Plotting',
    'Quantum',
    'Quantum_Materials',
    'QuantumOptics',
    'Signal_Processing',
    'Spintronics',
    'Stochastic',
    'Symbolic_Algebra',
    'Thermal_Transport'
]
PLATFORMS = ['Python', 'MATLAB', 'Mathematica']
def create_directory_structure(base_path: Path) -> Dict[str, List[str]]:
    """Create missing directories and return status report."""
    created = {'Python': [], 'MATLAB': [], 'Mathematica': []}
    for platform in PLATFORMS:
        platform_path = base_path / platform
        platform_path.mkdir(exist_ok=True)
        for subject in SUBJECTS:
            subject_path = platform_path / subject
            if not subject_path.exists():
                subject_path.mkdir(parents=True)
                created[platform].append(subject)
                # Create required subdirectories
                (subject_path / 'core').mkdir(exist_ok=True)
                (subject_path / 'examples' / 'beginner').mkdir(parents=True, exist_ok=True)
                (subject_path / 'examples' / 'intermediate').mkdir(parents=True, exist_ok=True)
                (subject_path / 'examples' / 'advanced').mkdir(parents=True, exist_ok=True)
                (subject_path / 'tests').mkdir(exist_ok=True)
                (subject_path / 'benchmarks').mkdir(exist_ok=True)
                # Create placeholder README
                readme_path = subject_path / 'README.md'
                if not readme_path.exists():
                    with open(readme_path, 'w') as f:
                        f.write(f"# {subject} - {platform}\n\n")
                        f.write("## Theory Background\n")
                        f.write("TODO: Add mathematical foundations\n\n")
                        f.write("## Implementation Overview\n")
                        f.write("TODO: Add algorithm descriptions\n\n")
                        f.write("## Canonical Problems\n")
                        f.write("TODO: Add classic problems\n\n")
                        f.write("## Platform-Specific Notes\n")
                        f.write("TODO: Add optimization notes\n\n")
                        f.write("## References\n")
                        f.write("TODO: Add citations\n")
    return created
def audit_existing_structure(base_path: Path) -> Dict[str, Dict[str, bool]]:
    """Audit which subjects exist in each platform."""
    status = {}
    for platform in PLATFORMS:
        status[platform] = {}
        platform_path = base_path / platform
        for subject in SUBJECTS:
            subject_path = platform_path / subject
            status[platform][subject] = subject_path.exists()
    return status
def generate_parity_report(status: Dict[str, Dict[str, bool]]) -> str:
    """Generate markdown parity report."""
    report = "# SciComp Platform Parity Report\n\n"
    report += "| Subject | Python | MATLAB | Mathematica | Status |\n"
    report += "|---------|--------|--------|-------------|--------|\n"
    for subject in SUBJECTS:
        py_status = "✓" if status['Python'][subject] else "✗"
        ml_status = "✓" if status['MATLAB'][subject] else "✗"
        mm_status = "✓" if status['Mathematica'][subject] else "✗"
        complete = all(status[platform][subject] for platform in PLATFORMS)
        overall = "Complete" if complete else "Missing"
        report += f"| {subject} | {py_status} | {ml_status} | {mm_status} | {overall} |\n"
    return report
if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    print("SciComp Directory Structure Enforcement")
    print("=" * 40)
    # Audit current state
    print("Auditing current structure...")
    pre_status = audit_existing_structure(base_path)
    # Create missing directories
    print("Creating missing directories...")
    created = create_directory_structure(base_path)
    # Report what was created
    for platform, subjects in created.items():
        if subjects:
            print(f"{platform}: Created {len(subjects)} subjects: {', '.join(subjects)}")
        else:
            print(f"{platform}: All subjects already exist")
    # Generate final report
    post_status = audit_existing_structure(base_path)
    report = generate_parity_report(post_status)
    with open(base_path / "PARITY_REPORT.md", "w") as f:
        f.write(report)
    print("\nParity report generated: PARITY_REPORT.md")
    print("Directory structure enforcement complete!")