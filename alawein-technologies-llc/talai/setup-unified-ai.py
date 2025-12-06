#!/usr/bin/env python3
"""
Setup Unified AI/IDE Configuration
One-time setup script to configure all AI and IDE integrations
"""

import subprocess
import sys
from pathlib import Path
import json
import yaml

def main():
    """Setup unified AI/IDE configuration"""
    print("🚀 Setting up Unified AI/IDE Configuration")
    print("=" * 50)

    root_path = Path.cwd()

    # 1. Install pre-commit if not available
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run(['pre-commit', '--version'], check=True, capture_output=True)
        print("  ✅ pre-commit already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  📦 Installing pre-commit...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pre-commit'], check=True)

    # 2. Install pre-commit hooks
    print("\n🔗 Setting up git hooks...")
    try:
        subprocess.run(['pre-commit', 'install'], check=True, cwd=root_path)
        subprocess.run(['pre-commit', 'install', '--hook-type', 'commit-msg'], check=True, cwd=root_path)
        print("  ✅ Pre-commit hooks installed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to install hooks: {e}")

    # 3. Set git commit template
    print("\n📝 Configuring git...")
    try:
        subprocess.run(['git', 'config', '--local', 'commit.template', '.gitmessage'],
                      check=True, cwd=root_path)
        print("  ✅ Git commit template configured")
    except subprocess.CalledProcessError:
        print("  ⚠️  Could not set git commit template")

    # 4. Run unified AI enforcer
    print("\n🤖 Enforcing AI/IDE configurations...")
    try:
        subprocess.run([sys.executable, '.meta/scripts/unified-ai-enforcer.py', '--fix'],
                      check=True, cwd=root_path)
        print("  ✅ AI/IDE configurations enforced")
    except subprocess.CalledProcessError:
        print("  ⚠️  Some configurations may need manual setup")

    # 5. Test pre-commit setup
    print("\n🧪 Testing configuration...")
    try:
        subprocess.run(['pre-commit', 'run', '--all-files'],
                      check=False, cwd=root_path, capture_output=True)
        print("  ✅ Pre-commit test completed")
    except subprocess.CalledProcessError:
        print("  ⚠️  Pre-commit test had issues (this is normal for first run)")

    # 6. Setup organization configs
    print("\n🏛️ Setting up organization configurations...")
    orgs_path = root_path / "ORGANIZATIONS"
    if orgs_path.exists():
        for org_dir in orgs_path.iterdir():
            if org_dir.is_dir():
                # Install pre-commit in each organization
                try:
                    subprocess.run(['pre-commit', 'install'],
                                  check=True, cwd=org_dir, capture_output=True)
                    print(f"  ✅ {org_dir.name} hooks installed")
                except subprocess.CalledProcessError:
                    print(f"  ⚠️  {org_dir.name} hooks setup failed")

    print("\n🎉 Unified AI/IDE Configuration Setup Complete!")
    print("\nNext steps:")
    print("1. Restart your IDE to pick up new settings")
    print("2. Install recommended VSCode extensions")
    print("3. Run 'pre-commit run --all-files' to test")
    print("4. Make a test commit to verify hooks work")

if __name__ == "__main__":
    main()
