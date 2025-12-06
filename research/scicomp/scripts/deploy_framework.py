#!/usr/bin/env python3
"""
SciComp - Deployment Script
==============================================
Automated deployment script for the SciComp.
Handles PyPI publishing, Docker image building, and documentation deployment.
Author: UC Berkeley SciComp Team
Date: 2025
License: MIT
"""
import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse
# Berkeley colors for output
class Colors:
    BERKELEY_BLUE = '\033[94m'
    CALIFORNIA_GOLD = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
def print_berkeley_header():
    """Print Berkeley deployment header."""
    print(f"\n{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BERKELEY_BLUE}{Colors.BOLD}🐻 SciComp - Deployment Automation 🐻{Colors.RESET}")
    print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.CALIFORNIA_GOLD}University of California, Berkeley{Colors.RESET}")
    print(f"{Colors.CALIFORNIA_GOLD}Deploying Scientific Computing Excellence{Colors.RESET}")
    print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}\n")
class DeploymentManager:
    """Manages framework deployment operations."""
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.version = self._get_version()
        self.deployment_config = self._load_config()
    def _get_version(self) -> str:
        """Extract version from setup.py or pyproject.toml."""
        try:
            # Try to get version from setup.py
            setup_py = self.root_path / "setup.py"
            if setup_py.exists():
                import ast
                with open(setup_py) as f:
                    content = f.read()
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'setup':
                            for keyword in node.keywords:
                                if keyword.arg == 'version':
                                    if isinstance(keyword.value, ast.Str):
                                        return keyword.value.s
            # Default version
            return "1.0.0"
        except Exception:
            return "1.0.0"
    def _load_config(self) -> Dict:
        """Load deployment configuration."""
        config_file = self.root_path / "deployment_config.json"
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "pypi": {
                    "repository": "https://upload.pypi.org/legacy/",
                    "test_repository": "https://test.pypi.org/legacy/"
                },
                "docker": {
                    "registry": "docker.io",
                    "namespace": "berkeley",
                    "image_name": "scicomp"
                },
                "docs": {
                    "build_dir": "docs/_build",
                    "deploy_branch": "gh-pages"
                }
            }
    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Run shell command and return success status."""
        try:
            if cwd is None:
                cwd = self.root_path
            print(f"{Colors.CALIFORNIA_GOLD}Running: {' '.join(command)}{Colors.RESET}")
            result = subprocess.run(command, cwd=cwd, check=True,
                                  capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}Command failed: {e}{Colors.RESET}")
            if e.stderr:
                print(f"{Colors.RED}Error output: {e.stderr}{Colors.RESET}")
            return False
    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites."""
        print(f"{Colors.BOLD}🔍 Checking Prerequisites{Colors.RESET}")
        print("-" * 40)
        prerequisites = {
            "python": ["python", "--version"],
            "pip": ["pip", "--version"],
            "build": ["python", "-m", "build", "--version"],
            "twine": ["twine", "--version"],
            "docker": ["docker", "--version"],
            "git": ["git", "--version"]
        }
        all_good = True
        for name, command in prerequisites.items():
            try:
                subprocess.run(command, check=True, capture_output=True)
                print(f"{Colors.GREEN}✓{Colors.RESET} {name}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"{Colors.RED}✗{Colors.RESET} {name} - Not available")
                all_good = False
        if not all_good:
            print(f"\n{Colors.RED}Please install missing prerequisites before deployment.{Colors.RESET}")
        return all_good
    def clean_build_artifacts(self):
        """Clean previous build artifacts."""
        print(f"\n{Colors.BOLD}🧹 Cleaning Build Artifacts{Colors.RESET}")
        print("-" * 40)
        directories_to_clean = ["build", "dist", "*.egg-info"]
        for pattern in directories_to_clean:
            if pattern.startswith("*"):
                # Handle glob patterns
                import glob
                paths = glob.glob(str(self.root_path / pattern))
                for path in paths:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Removed directory: {path}")
            else:
                path = self.root_path / pattern
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"Removed directory: {path}")
                    else:
                        path.unlink()
                        print(f"Removed file: {path}")
    def run_tests(self) -> bool:
        """Run comprehensive tests before deployment."""
        print(f"\n{Colors.BOLD}🧪 Running Tests{Colors.RESET}")
        print("-" * 40)
        # Check if pytest is available and run tests
        test_commands = [
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            ["python", "scripts/validate_framework.py"]
        ]
        for command in test_commands:
            if not self.run_command(command):
                print(f"{Colors.RED}Tests failed! Deployment aborted.{Colors.RESET}")
                return False
        print(f"{Colors.GREEN}All tests passed!{Colors.RESET}")
        return True
    def build_package(self) -> bool:
        """Build Python package for PyPI."""
        print(f"\n{Colors.BOLD}📦 Building Package{Colors.RESET}")
        print("-" * 40)
        # Build source and wheel distributions
        if not self.run_command(["python", "-m", "build"]):
            return False
        # Verify build artifacts
        dist_dir = self.root_path / "dist"
        if not dist_dir.exists():
            print(f"{Colors.RED}Distribution directory not found!{Colors.RESET}")
            return False
        dist_files = list(dist_dir.glob("*"))
        print(f"Built {len(dist_files)} distribution files:")
        for file in dist_files:
            print(f"  - {file.name}")
        return True
    def publish_to_pypi(self, test: bool = True) -> bool:
        """Publish package to PyPI."""
        repository = "testpypi" if test else "pypi"
        print(f"\n{Colors.BOLD}🚀 Publishing to {'Test ' if test else ''}PyPI{Colors.RESET}")
        print("-" * 40)
        # Check credentials
        if test:
            repo_url = self.deployment_config["pypi"]["test_repository"]
        else:
            repo_url = self.deployment_config["pypi"]["repository"]
        command = [
            "twine", "upload",
            "--repository", repository,
            "dist/*"
        ]
        if self.run_command(command):
            print(f"{Colors.GREEN}Successfully published to {'Test ' if test else ''}PyPI!{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}Failed to publish to {'Test ' if test else ''}PyPI{Colors.RESET}")
            return False
    def build_docker_image(self) -> bool:
        """Build Docker image."""
        print(f"\n{Colors.BOLD}🐳 Building Docker Image{Colors.RESET}")
        print("-" * 40)
        config = self.deployment_config["docker"]
        image_name = f"{config['namespace']}/{config['image_name']}:{self.version}"
        latest_name = f"{config['namespace']}/{config['image_name']}:latest"
        # Build image
        build_command = [
            "docker", "build",
            "-t", image_name,
            "-t", latest_name,
            "."
        ]
        if not self.run_command(build_command):
            return False
        print(f"{Colors.GREEN}Docker image built: {image_name}{Colors.RESET}")
        return True
    def push_docker_image(self) -> bool:
        """Push Docker image to registry."""
        print(f"\n{Colors.BOLD}🚀 Pushing Docker Image{Colors.RESET}")
        print("-" * 40)
        config = self.deployment_config["docker"]
        image_name = f"{config['namespace']}/{config['image_name']}"
        # Push both version tag and latest
        for tag in [self.version, "latest"]:
            full_name = f"{image_name}:{tag}"
            if not self.run_command(["docker", "push", full_name]):
                return False
        print(f"{Colors.GREEN}Docker images pushed successfully!{Colors.RESET}")
        return True
    def build_documentation(self) -> bool:
        """Build documentation."""
        print(f"\n{Colors.BOLD}📚 Building Documentation{Colors.RESET}")
        print("-" * 40)
        docs_dir = self.root_path / "docs"
        if not docs_dir.exists():
            print(f"{Colors.CALIFORNIA_GOLD}No docs directory found, skipping...{Colors.RESET}")
            return True
        # Build Sphinx documentation
        build_command = ["make", "html"]
        if not self.run_command(build_command, cwd=docs_dir):
            # Try alternative build method
            build_command = ["sphinx-build", "-b", "html", ".", "_build/html"]
            if not self.run_command(build_command, cwd=docs_dir):
                return False
        print(f"{Colors.GREEN}Documentation built successfully!{Colors.RESET}")
        return True
    def create_github_release(self) -> bool:
        """Create GitHub release."""
        print(f"\n{Colors.BOLD}🏷️ Creating GitHub Release{Colors.RESET}")
        print("-" * 40)
        # Check if gh CLI is available
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Colors.CALIFORNIA_GOLD}GitHub CLI not available, skipping release creation...{Colors.RESET}")
            return True
        # Create release
        release_command = [
            "gh", "release", "create", f"v{self.version}",
            "--title", f"SciComp v{self.version}",
            "--notes", f"Release v{self.version} of the SciComp",
            "--draft"
        ]
        if self.run_command(release_command):
            print(f"{Colors.GREEN}GitHub release created: v{self.version}{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}Failed to create GitHub release{Colors.RESET}")
            return False
    def deploy_full(self, test_pypi: bool = True) -> bool:
        """Run full deployment pipeline."""
        print_berkeley_header()
        print(f"Deploying SciComp v{self.version}")
        print(f"Target: {'Test PyPI' if test_pypi else 'Production PyPI'}")
        # Step-by-step deployment
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Clean Build", lambda: (self.clean_build_artifacts(), True)[1]),
            ("Run Tests", self.run_tests),
            ("Build Package", self.build_package),
            ("Build Documentation", self.build_documentation)
        ]
        # Add PyPI deployment
        if test_pypi:
            steps.append(("Publish to Test PyPI", lambda: self.publish_to_pypi(test=True)))
        else:
            steps.append(("Publish to PyPI", lambda: self.publish_to_pypi(test=False)))
        # Add Docker deployment
        steps.extend([
            ("Build Docker Image", self.build_docker_image),
            ("Create GitHub Release", self.create_github_release)
        ])
        # Execute deployment steps
        failed_steps = []
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    print(f"{Colors.RED}Step failed: {step_name}{Colors.RESET}")
            except Exception as e:
                failed_steps.append(step_name)
                print(f"{Colors.RED}Step error: {step_name} - {e}{Colors.RESET}")
        # Final result
        if not failed_steps:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 DEPLOYMENT SUCCESSFUL! 🎉{Colors.RESET}")
            print(f"{Colors.BERKELEY_BLUE}SciComp v{self.version} deployed successfully!{Colors.RESET}")
            print(f"{Colors.CALIFORNIA_GOLD}🐻💙💛 Go Bears! 💙💛🐻{Colors.RESET}")
            return True
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ DEPLOYMENT FAILED ❌{Colors.RESET}")
            print(f"Failed steps: {', '.join(failed_steps)}")
            return False
def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="SciComp Deployment")
    parser.add_argument("--production", action="store_true",
                       help="Deploy to production PyPI (default: test PyPI)")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip test execution (not recommended)")
    parser.add_argument("--docker-only", action="store_true",
                       help="Only build Docker image")
    parser.add_argument("--docs-only", action="store_true",
                       help="Only build documentation")
    args = parser.parse_args()
    # Get root path
    root_path = Path(__file__).parent.parent
    # Create deployment manager
    deployer = DeploymentManager(root_path)
    # Execute requested deployment
    if args.docker_only:
        print_berkeley_header()
        success = deployer.build_docker_image()
    elif args.docs_only:
        print_berkeley_header()
        success = deployer.build_documentation()
    else:
        success = deployer.deploy_full(test_pypi=not args.production)
    # Exit with appropriate code
    return 0 if success else 1
if __name__ == "__main__":
    exit(main())