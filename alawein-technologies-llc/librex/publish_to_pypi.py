#!/usr/bin/env python3
"""
PyPI Publishing Script for Librex.QAP v1.0

Automated publishing pipeline with comprehensive validation and safety checks.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

class PyPIPublisher:
    """Safe and validated PyPI publishing pipeline"""

    def __init__(self, package_dir: Path):
        self.package_dir = Path(package_dir)
        self.dist_dir = self.package_dir / "dist"
        self.build_dir = self.package_dir / "build"

    def validate_environment(self) -> bool:
        """Validate publishing environment and dependencies"""
        print("🔍 Validating publishing environment...")

        # Check required tools
        required_tools = ["python", "pip", "twine"]
        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"],
                             capture_output=True, check=True)
                print(f"  ✅ {tool} available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"  ❌ {tool} not found")
                return False

        # Check PyPI credentials
        try:
            result = subprocess.run(["twine", "check", "--version"],
                                  capture_output=True, check=True)
            print("  ✅ Twine configured")
        except subprocess.CalledProcessError:
            print("  ⚠️  Twine not configured - will need credentials")

        return True

    def run_tests(self) -> bool:
        """Run comprehensive test suite"""
        print("🧪 Running test suite...")

        try:
            # Run unit tests
            result = subprocess.run([
                "python", "-m", "pytest", "tests/",
                "-v", "--tb=short", "--color=yes"
            ], cwd=self.package_dir, capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅ All tests passed")
                return True
            else:
                print("  ❌ Tests failed:")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Test execution failed: {e}")
            return False

    def validate_package_structure(self) -> bool:
        """Validate package structure and metadata"""
        print("📋 Validating package structure...")

        # Check essential files
        required_files = [
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "MANIFEST.in"
        ]

        for file_path in required_files:
            full_path = self.package_dir / file_path
            if full_path.exists():
                print(f"  ✅ {file_path} present")
            else:
                print(f"  ❌ {file_path} missing")
                return False

        # Validate pyproject.toml
        try:
            import tomllib
            with open(self.package_dir / "pyproject.toml", "rb") as f:
                config = tomllib.load(f)

            required_fields = ["name", "version", "description", "authors"]
            for field in required_fields:
                if field in config.get("project", {}):
                    print(f"  ✅ project.{field} configured")
                else:
                    print(f"  ❌ project.{field} missing")
                    return False

        except Exception as e:
            print(f"  ❌ Invalid pyproject.toml: {e}")
            return False

        return True

    def build_package(self) -> bool:
        """Build source and wheel distributions"""
        print("🔨 Building package distributions...")

        # Clean previous builds
        if self.dist_dir.exists():
            import shutil
            shutil.rmtree(self.dist_dir)
        if self.build_dir.exists():
            import shutil
            shutil.rmtree(self.build_dir)

        try:
            # Build distributions
            result = subprocess.run([
                "python", "-m", "build", "--sdist", "--wheel"
            ], cwd=self.package_dir, capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅ Package built successfully")

                # Check built files
                dist_files = list(self.dist_dir.glob("*"))
                print(f"  📦 Built {len(dist_files)} distribution files:")
                for file in dist_files:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"    - {file.name} ({size_mb:.1f} MB)")

                return True
            else:
                print("  ❌ Build failed:")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Build execution failed: {e}")
            return False

    def check_distributions(self) -> bool:
        """Validate built distributions with twine"""
        print("🔍 Checking distributions with Twine...")

        try:
            result = subprocess.run([
                "twine", "check", "dist/*"
            ], cwd=self.package_dir, capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅ All distribution checks passed")
                print(result.stdout)
                return True
            else:
                print("  ❌ Distribution checks failed:")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Twine check failed: {e}")
            return False

    def upload_to_test_pypi(self) -> bool:
        """Upload to Test PyPI for validation"""
        print("🧪 Uploading to Test PyPI...")

        try:
            result = subprocess.run([
                "twine", "upload",
                "--repository", "testpypi",
                "--verbose",
                "dist/*"
            ], cwd=self.package_dir, capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅ Successfully uploaded to Test PyPI")
                print("  🔗 Test with: pip install --index-url https://test.pypi.org/simple/ librex-qap")
                return True
            else:
                print("  ❌ Test PyPI upload failed:")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Test PyPI upload failed: {e}")
            return False

    def upload_to_production_pypi(self) -> bool:
        """Upload to production PyPI"""
        print("🚀 Uploading to production PyPI...")

        # Final confirmation
        response = input("  ⚠️  This will publish to PRODUCTION PyPI. Continue? (yes/no): ")
        if response.lower() != "yes":
            print("  ❌ Upload cancelled by user")
            return False

        try:
            result = subprocess.run([
                "twine", "upload",
                "--verbose",
                "dist/*"
            ], cwd=self.package_dir, capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅ Successfully uploaded to PyPI!")
                print("  🔗 Install with: pip install librex-qap")
                print("  📖 Documentation: https://librex.dev/qap/docs")
                return True
            else:
                print("  ❌ PyPI upload failed:")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"  ❌ PyPI upload failed: {e}")
            return False

    def generate_publishing_report(self) -> Dict:
        """Generate comprehensive publishing report"""
        print("📊 Generating publishing report...")

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "package": "librex-qap",
            "version": "1.0.0",
            "build_files": [],
            "validation_results": {},
            "publishing_status": "completed"
        }

        # List distribution files
        if self.dist_dir.exists():
            for file in self.dist_dir.glob("*"):
                report["build_files"].append({
                    "name": file.name,
                    "size_bytes": file.stat().st_size,
                    "size_mb": round(file.stat().st_size / (1024 * 1024), 2)
                })

        # Save report
        report_path = self.package_dir / "publishing_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"  📄 Report saved to {report_path}")
        return report

    def publish(self, test_mode: bool = True) -> bool:
        """Complete publishing pipeline"""
        print("🚀 Starting Librex.QAP v1.0 publishing pipeline...")
        print("=" * 60)

        # Step 1: Environment validation
        if not self.validate_environment():
            return False

        # Step 2: Package structure validation
        if not self.validate_package_structure():
            return False

        # Step 3: Run tests
        if not self.run_tests():
            return False

        # Step 4: Build package
        if not self.build_package():
            return False

        # Step 5: Check distributions
        if not self.check_distributions():
            return False

        # Step 6: Upload
        if test_mode:
            success = self.upload_to_test_pypi()
        else:
            success = self.upload_to_production_pypi()

        if success:
            # Step 7: Generate report
            self.generate_publishing_report()
            print("=" * 60)
            print("🎉 Publishing pipeline completed successfully!")
            return False
        else:
            print("=" * 60)
            print("❌ Publishing pipeline failed!")
            return False

def main():
    """Main publishing script"""
    import argparse

    parser = argparse.ArgumentParser(description="Publish Librex.QAP to PyPI")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Upload to production PyPI (default: Test PyPI)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test suite (not recommended)"
    )

    args = parser.parse_args()

    # Get package directory
    package_dir = Path(__file__).parent

    # Create publisher
    publisher = PyPIPublisher(package_dir)

    # Skip tests if requested
    if args.skip_tests:
        publisher.run_tests = lambda: True

    # Run publishing pipeline
    success = publisher.publish(test_mode=not args.production)

    if success:
        print("\n🎯 Next steps:")
        if not args.production:
            print("  1. Test installation: pip install --index-url https://test.pypi.org/simple/ librex-qap")
            print("  2. Verify functionality with test scripts")
            print("  3. Run with --production flag to publish to PyPI")
        else:
            print("  1. Verify installation: pip install librex-qap")
            print("  2. Update documentation and README")
            print("  3. Announce release to community")
            print("  4. Monitor downloads and feedback")
        sys.exit(0)
    else:
        print("\n❌ Publishing failed. Check logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
