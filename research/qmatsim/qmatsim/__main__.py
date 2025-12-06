#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---- Utilities ----
def get_project_root() -> Path:
    return Path(__file__).parent.parent

def validate_file_exists(file_path: Path, description: str) -> bool:
    """Check if required file exists and provide helpful error messages."""
    if not file_path.exists():
        print(f"❌ Missing {description}: {file_path}")
        print(f"   Please ensure the file exists or check your project structure")
        return False
    return True

def run_script_safely(script_path: str, args: list, description: str) -> None:
    project_root = get_project_root()
    script_full_path = project_root / script_path
    
    if not validate_file_exists(script_full_path, f"{description} script"):
        sys.exit(1)
    
    try:
        # Change to project root directory for script execution
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Run the script
        result = subprocess.run(["bash", script_path] + args, 
                              capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"❌ {description} failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout}")
            sys.exit(result.returncode)
        else:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(result.stdout)
                
    except FileNotFoundError:
        print(f"❌ Error: bash command not found. Please ensure bash is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error running {description}: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

# ---- DFT ----
def run_dft(args):
    print(f"🔬 Starting DFT relaxation for {args.material} ({args.structure})")
    run_script_safely("scripts/run-DFT.sh", [args.material, args.structure], "DFT simulation")

# ---- MD ----
def run_md(args):
    structure = args.structure
    mode = args.mode
    project_root = get_project_root()
    
    print(f"⚛️ Starting MD simulation for {structure} (mode: {mode})")
    
    # Validate data file exists
    data_file = project_root / f"lammps/data/{structure}.data"
    if not validate_file_exists(data_file, "LAMMPS data file"):
        print(f"   Available data files:")
        data_dir = project_root / "lammps/data"
        if data_dir.exists():
            for f in data_dir.glob("*.data"):
                print(f"     - {f.name}")
        sys.exit(1)

    if mode == "compress":
        input_file = project_root / "lammps/in/compress_y.in"
        if not validate_file_exists(input_file, "LAMMPS compress input script"):
            sys.exit(1)
        run_script_safely("scripts/compress-MD.sh", [structure], "MD compression simulation")

    elif mode == "all":
        # Check for required input files
        missing = []
        required_files = ["compress_y.in", "deformation.in", "minimization.in"]
        for fname in required_files:
            if not (project_root / f"lammps/in/{fname}").exists():
                missing.append(fname)
        
        if missing:
            print(f"❌ Missing LAMMPS input files: {', '.join(missing)}")
            print(f"   Please ensure these files exist in lammps/in/")
            sys.exit(1)
            
        run_script_safely("scripts/run-MD.sh", [structure], "MD simulation suite")
    else:
        print(f"❌ Unknown mode '{mode}'. Use '--mode compress' or '--mode all'.")
        sys.exit(1)

# ---- Postprocessing ----
def run_post(args):
    print(f"📊 Starting postprocessing analysis for {args.material} ({args.structure})")
    run_script_safely("scripts/run-postprocessing.sh", [args.material, args.structure], "Postprocessing analysis")

# ---- CLI Entrypoint ----
def main():
    parser = argparse.ArgumentParser(
        description="QMatSim CLI — Multiscale DFT + MD toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  qmatsim relax --material MoS2 --structure 1x10_rectangular
  qmatsim minimize --structure 1x10_rectangular --mode compress
  qmatsim analyze --material MoS2 --structure 1x10_rectangular

For more information, visit: https://github.com/alawein/QMatSim"""
    )
    
    subparsers = parser.add_subparsers(
        title="Available Commands", 
        dest="command",
        help="QMatSim simulation workflows"
    )

    # DFT command
    p_dft = subparsers.add_parser(
        "relax", 
        help="Run SIESTA DFT simulation",
        description="Perform electronic structure calculations using SIESTA"
    )
    p_dft.add_argument("--material", required=True, 
                      help="Material name (supported: MoS2, MoSe2, WS2, WSe2)")
    p_dft.add_argument("--structure", required=True, 
                      help="Structure type (e.g., 1x1_primitive, 1x10_rectangular)")
    p_dft.set_defaults(func=run_dft)

    # MD command
    p_md = subparsers.add_parser(
        "minimize", 
        help="Run LAMMPS MD simulation",
        description="Perform molecular dynamics simulations using LAMMPS"
    )
    p_md.add_argument("--structure", required=True, 
                     help="Structure name (must match a .data file in lammps/data/)")
    p_md.add_argument("--mode", choices=["compress", "all"], default="all", 
                     help="Simulation mode: 'compress' for Y-compression only, 'all' for full suite [default: all]")
    p_md.set_defaults(func=run_md)

    # Postprocessing command
    p_post = subparsers.add_parser(
        "analyze", 
        help="Run DFT postprocessing analysis",
        description="Analyze DFT results and generate plots/data"
    )
    p_post.add_argument("--material", required=True,
                       help="Material name (must match DFT calculation material)")
    p_post.add_argument("--structure", required=True,
                       help="Structure type (must match DFT calculation structure)")
    p_post.set_defaults(func=run_post)

    # Parse arguments and run
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
