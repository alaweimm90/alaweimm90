#!/usr/bin/env python3
"""
Berkeley SciComp Command Line Interface
======================================
Professional command-line interface for the UC Berkeley Scientific Computing
Framework, providing unified access to all computational tools, examples,
and utilities with Berkeley branding and academic standards.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import argparse
import json
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib.util
# Berkeley colors for terminal output
BERKELEY_BLUE = '\033[38;2;0;50;98m'
CALIFORNIA_GOLD = '\033[38;2;253;181;21m'
RESET = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
class BerkeleyCLI:
    """SciComp Command Line Interface."""
    def __init__(self):
        self.framework_root = Path(__file__).parent.parent.parent
        self.config = self._load_config()
        self.version = self._get_version()
    def _load_config(self) -> Dict[str, Any]:
        """Load Berkeley configuration."""
        config_file = self.framework_root / "berkeley_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    def _get_version(self) -> str:
        """Get framework version."""
        return self.config.get('framework', {}).get('version', '1.0.0')
    def print_banner(self):
        """Print Berkeley SciComp banner."""
        banner = f"""
{BERKELEY_BLUE}{BOLD}================================================================
SciComp - Command Line Interface
================================================================{RESET}
{CALIFORNIA_GOLD}University of California, Berkeley{RESET}
{BERKELEY_BLUE}Meshal Alawein (meshal@berkeley.edu){RESET}
Version: {self.version}
Framework: Multi-platform Scientific Computing (Python, MATLAB, Mathematica)
License: MIT
{BERKELEY_BLUE}Available commands: run, test, demo, config, docs, style{RESET}
"""
        print(banner)
    def run_quantum_physics(self, args):
        """Run quantum physics simulations."""
        print(f"{BERKELEY_BLUE}Running Quantum Physics Simulation...{RESET}")
        if args.example == "harmonic_oscillator":
            try:
                from Python.quantum_physics.harmonic_oscillator import HarmonicOscillator, HarmonicOscillatorConfig
                config = HarmonicOscillatorConfig(
                    n_max=args.n_max or 10,
                    omega=args.omega or 1.0,
                    mass=args.mass or 1.0,
                    hbar=args.hbar or 1.0
                )
                ho = HarmonicOscillator(config)
                energies = ho.get_energy_levels()
                print(f"{GREEN}✓ Harmonic Oscillator Simulation Complete{RESET}")
                print(f"Energy levels (n=0 to {args.n_max or 10}):")
                for i, E in enumerate(energies):
                    print(f"  E_{i} = {E:.6f}")
            except ImportError as e:
                print(f"{RED}✗ Error importing quantum physics modules: {e}{RESET}")
        elif args.example == "tunneling":
            print(f"{YELLOW}Quantum tunneling simulation not yet implemented{RESET}")
        else:
            print(f"{RED}✗ Unknown quantum physics example: {args.example}{RESET}")
    def run_ml_physics(self, args):
        """Run machine learning physics simulations."""
        print(f"{BERKELEY_BLUE}Running ML Physics Simulation...{RESET}")
        if args.example == "pinn_schrodinger":
            try:
                from Python.ml_physics.pinns.schrodinger_pinn import SchrodingerPINN, SchrodingerConfig
                config = SchrodingerConfig(
                    x_domain=(-2.0, 2.0),
                    t_domain=(0.0, 1.0),
                    nx=args.nx or 50,
                    nt=args.nt or 20,
                    epochs=args.epochs or 100
                )
                pinn = SchrodingerPINN(config)
                print(f"{GREEN}✓ Schrödinger PINN configured{RESET}")
                print(f"Domain: x ∈ {config.x_domain}, t ∈ {config.t_domain}")
                print(f"Grid: {config.nx} × {config.nt}")
                print(f"Training epochs: {config.epochs}")
            except ImportError as e:
                print(f"{RED}✗ Error importing ML physics modules: {e}{RESET}")
        elif args.example == "fno":
            print(f"{YELLOW}Fourier Neural Operator example not yet implemented{RESET}")
        else:
            print(f"{RED}✗ Unknown ML physics example: {args.example}{RESET}")
    def run_quantum_computing(self, args):
        """Run quantum computing algorithms."""
        print(f"{BERKELEY_BLUE}Running Quantum Computing Algorithm...{RESET}")
        if args.example == "grover":
            try:
                from Python.quantum_computing.algorithms.grover import GroverSearch, GroverConfig
                config = GroverConfig(
                    n_qubits=args.n_qubits or 3,
                    n_shots=args.n_shots or 1000,
                    target_items=args.target_items or ['101', '110']
                )
                grover = GroverSearch(config)
                print(f"{GREEN}✓ Grover's Algorithm configured{RESET}")
                print(f"Qubits: {config.n_qubits}")
                print(f"Target items: {config.target_items}")
                print(f"Shots: {config.n_shots}")
            except ImportError as e:
                print(f"{RED}✗ Error importing quantum computing modules: {e}{RESET}")
        else:
            print(f"{RED}✗ Unknown quantum computing example: {args.example}{RESET}")
    def run_tests(self, args):
        """Run framework tests."""
        print(f"{BERKELEY_BLUE}Running Berkeley SciComp Tests...{RESET}")
        if args.platform == "all" or args.platform == "python":
            print(f"{BERKELEY_BLUE}Running Python tests...{RESET}")
            test_files = [
                "tests/python/test_quantum_physics.py",
                "tests/python/test_ml_physics.py",
                "tests/python/test_quantum_computing.py"
            ]
            for test_file in test_files:
                test_path = self.framework_root / test_file
                if test_path.exists():
                    try:
                        result = subprocess.run([sys.executable, str(test_path)],
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            print(f"{GREEN}✓ {test_file}{RESET}")
                        else:
                            print(f"{RED}✗ {test_file}{RESET}")
                    except Exception as e:
                        print(f"{RED}✗ Error running {test_file}: {e}{RESET}")
                else:
                    print(f"{YELLOW}⚠ Test file not found: {test_file}{RESET}")
        if args.platform == "all" or args.platform == "matlab":
            print(f"{BERKELEY_BLUE}MATLAB tests require MATLAB installation{RESET}")
        if args.platform == "all" or args.platform == "mathematica":
            print(f"{BERKELEY_BLUE}Mathematica tests require Mathematica installation{RESET}")
    def run_demo(self, args):
        """Run framework demonstrations."""
        print(f"{BERKELEY_BLUE}Running Berkeley SciComp Demonstrations...{RESET}")
        demo_scripts = {
            "quantum": "examples/python/quantum_tunneling_demo.py",
            "ml": "examples/python/ml_physics_demo.py",
            "computing": "examples/python/quantum_computing_demo.py",
            "style": "assets/berkeley_style.py"
        }
        if args.type in demo_scripts:
            demo_path = self.framework_root / demo_scripts[args.type]
            if demo_path.exists():
                try:
                    if args.type == "style":
                        # Run style demo
                        subprocess.run([sys.executable, str(demo_path)])
                    else:
                        # Run example demo
                        subprocess.run([sys.executable, str(demo_path)])
                    print(f"{GREEN}✓ Demo completed: {args.type}{RESET}")
                except Exception as e:
                    print(f"{RED}✗ Error running demo: {e}{RESET}")
            else:
                print(f"{RED}✗ Demo file not found: {demo_path}{RESET}")
        else:
            print(f"{RED}✗ Unknown demo type: {args.type}{RESET}")
            print(f"Available demos: {', '.join(demo_scripts.keys())}")
    def show_config(self, args):
        """Show framework configuration."""
        print(f"{BERKELEY_BLUE}SciComp Configuration{RESET}")
        print("=" * 50)
        if args.section:
            if args.section in self.config:
                print(f"{CALIFORNIA_GOLD}{args.section.title()}:{RESET}")
                self._print_config_section(self.config[args.section])
            else:
                print(f"{RED}✗ Configuration section not found: {args.section}{RESET}")
        else:
            for section, content in self.config.items():
                print(f"{CALIFORNIA_GOLD}{section.title()}:{RESET}")
                self._print_config_section(content, indent=2)
                print()
    def _print_config_section(self, section: Dict[str, Any], indent: int = 0):
        """Print configuration section with proper formatting."""
        prefix = " " * indent
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    self._print_config_section(value, indent + 2)
                elif isinstance(value, list):
                    print(f"{prefix}{key}: [{', '.join(map(str, value))}]")
                else:
                    print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{section}")
    def show_docs(self, args):
        """Show documentation."""
        print(f"{BERKELEY_BLUE}Berkeley SciComp Documentation{RESET}")
        docs_dir = self.framework_root / "docs"
        if args.topic:
            doc_files = {
                "quantum": "theory/quantum_mechanics_theory.md",
                "ml": "theory/ml_physics_theory.md",
                "methods": "theory/computational_methods.md",
                "engineering": "theory/engineering_applications.md",
                "style": "../assets/STYLE_GUIDE.md"
            }
            if args.topic in doc_files:
                doc_path = docs_dir / doc_files[args.topic]
                if doc_path.exists():
                    print(f"Documentation: {doc_path}")
                    if args.open:
                        try:
                            if sys.platform.startswith('darwin'):  # macOS
                                subprocess.run(['open', str(doc_path)])
                            elif sys.platform.startswith('win'):   # Windows
                                subprocess.run(['start', str(doc_path)], shell=True)
                            else:  # Linux
                                subprocess.run(['xdg-open', str(doc_path)])
                        except Exception as e:
                            print(f"{RED}✗ Error opening documentation: {e}{RESET}")
                else:
                    print(f"{RED}✗ Documentation file not found: {doc_path}{RESET}")
            else:
                print(f"{RED}✗ Unknown documentation topic: {args.topic}{RESET}")
                print(f"Available topics: {', '.join(doc_files.keys())}")
        else:
            print("Available documentation:")
            print(f"  {GREEN}quantum{RESET}     - Quantum mechanics theory")
            print(f"  {GREEN}ml{RESET}          - Machine learning physics theory")
            print(f"  {GREEN}methods{RESET}     - Computational methods")
            print(f"  {GREEN}engineering{RESET} - Engineering applications")
            print(f"  {GREEN}style{RESET}       - Visual identity style guide")
    def apply_style(self, args):
        """Apply Berkeley styling."""
        print(f"{BERKELEY_BLUE}Applying Berkeley Visual Identity...{RESET}")
        if args.platform == "python":
            try:
                from assets.berkeley_style import BerkeleyPlotStyle
                BerkeleyPlotStyle.apply_style()
                print(f"{GREEN}✓ Berkeley Python styling applied{RESET}")
            except ImportError as e:
                print(f"{RED}✗ Error importing Berkeley style: {e}{RESET}")
        elif args.platform == "matlab":
            print(f"{BERKELEY_BLUE}Run 'berkeley_style()' in MATLAB{RESET}")
        elif args.platform == "mathematica":
            print(f"{BERKELEY_BLUE}Load '<< \"assets/BerkeleyStyle.wl\"' in Mathematica{RESET}")
        else:
            print(f"{RED}✗ Unknown platform: {args.platform}{RESET}")
def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="berkeley-scicomp",
        description="SciComp Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{BERKELEY_BLUE}Examples:{RESET}
  berkeley-scicomp run quantum harmonic_oscillator --n_max 20
  berkeley-scicomp run ml pinn_schrodinger --epochs 500
  berkeley-scicomp test --platform python
  berkeley-scicomp demo --type quantum
  berkeley-scicomp config --section visual_identity
  berkeley-scicomp docs --topic quantum --open
  berkeley-scicomp style --platform python
{CALIFORNIA_GOLD}University of California, Berkeley{RESET}
{BERKELEY_BLUE}Meshal Alawein (meshal@berkeley.edu){RESET}
        """
    )
    parser.add_argument('--version', action='version', version='Berkeley SciComp 1.0.0')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    # Run command
    run_parser = subparsers.add_parser('run', help='Run simulations and computations')
    run_subparsers = run_parser.add_subparsers(dest='module', help='Computational modules')
    # Quantum physics
    quantum_parser = run_subparsers.add_parser('quantum', help='Quantum physics simulations')
    quantum_parser.add_argument('example', choices=['harmonic_oscillator', 'tunneling'])
    quantum_parser.add_argument('--n_max', type=int, help='Maximum quantum number')
    quantum_parser.add_argument('--omega', type=float, help='Angular frequency')
    quantum_parser.add_argument('--mass', type=float, help='Particle mass')
    quantum_parser.add_argument('--hbar', type=float, help='Reduced Planck constant')
    # ML physics
    ml_parser = run_subparsers.add_parser('ml', help='Machine learning physics')
    ml_parser.add_argument('example', choices=['pinn_schrodinger', 'fno'])
    ml_parser.add_argument('--nx', type=int, help='Spatial grid points')
    ml_parser.add_argument('--nt', type=int, help='Temporal grid points')
    ml_parser.add_argument('--epochs', type=int, help='Training epochs')
    # Quantum computing
    qc_parser = run_subparsers.add_parser('computing', help='Quantum computing algorithms')
    qc_parser.add_argument('example', choices=['grover', 'vqe', 'qaoa'])
    qc_parser.add_argument('--n_qubits', type=int, help='Number of qubits')
    qc_parser.add_argument('--n_shots', type=int, help='Number of measurement shots')
    qc_parser.add_argument('--target_items', nargs='+', help='Target items for search')
    # Test command
    test_parser = subparsers.add_parser('test', help='Run framework tests')
    test_parser.add_argument('--platform', choices=['all', 'python', 'matlab', 'mathematica'],
                           default='all', help='Testing platform')
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstrations')
    demo_parser.add_argument('--type', choices=['quantum', 'ml', 'computing', 'style'],
                           required=True, help='Demo type')
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--section', help='Configuration section to display')
    # Docs command
    docs_parser = subparsers.add_parser('docs', help='Show documentation')
    docs_parser.add_argument('--topic', choices=['quantum', 'ml', 'methods', 'engineering', 'style'],
                           help='Documentation topic')
    docs_parser.add_argument('--open', action='store_true', help='Open documentation file')
    # Style command
    style_parser = subparsers.add_parser('style', help='Apply Berkeley styling')
    style_parser.add_argument('--platform', choices=['python', 'matlab', 'mathematica'],
                            required=True, help='Platform for styling')
    return parser
def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    cli = BerkeleyCLI()
    if not args.command:
        cli.print_banner()
        parser.print_help()
        return
    # Route commands
    if args.command == 'run':
        if args.module == 'quantum':
            cli.run_quantum_physics(args)
        elif args.module == 'ml':
            cli.run_ml_physics(args)
        elif args.module == 'computing':
            cli.run_quantum_computing(args)
        else:
            print(f"{RED}✗ Unknown module: {args.module}{RESET}")
    elif args.command == 'test':
        cli.run_tests(args)
    elif args.command == 'demo':
        cli.run_demo(args)
    elif args.command == 'config':
        cli.show_config(args)
    elif args.command == 'docs':
        cli.show_docs(args)
    elif args.command == 'style':
        cli.apply_style(args)
    else:
        print(f"{RED}✗ Unknown command: {args.command}{RESET}")
        parser.print_help()
if __name__ == "__main__":
    main()