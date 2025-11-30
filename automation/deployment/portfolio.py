#!/usr/bin/env python3
"""
Portfolio Deployment System
Automated deployment of portfolio sites with accessibility enhancements.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class DeploymentConfig:
    """Configuration for portfolio deployment."""
    name: str
    source_path: Path
    output_path: Path
    platform: str = "netlify"  # netlify, vercel, github-pages
    domain: Optional[str] = None
    build_command: Optional[str] = None
    publish_dir: str = "dist"
    environment: str = "production"
    accessibility_level: str = "AA"  # A, AA, AAA
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    url: Optional[str] = None
    deploy_id: Optional[str] = None
    duration_seconds: float = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    accessibility_score: Optional[float] = None


class PortfolioDeployer:
    """Deploy portfolio sites with best practices."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.log_entries: List[Dict[str, Any]] = []

    def log(self, message: str, level: str = "INFO"):
        """Log deployment activity."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.log_entries.append(entry)
        print(f"[{entry['timestamp']}] {level}: {message}")

    def validate_source(self) -> bool:
        """Validate source directory exists and has required files."""
        if not self.config.source_path.exists():
            self.log(f"Source path does not exist: {self.config.source_path}", "ERROR")
            return False

        # Check for common required files
        required_files = ["index.html", "package.json", "README.md"]
        found_files = []

        for f in required_files:
            if (self.config.source_path / f).exists():
                found_files.append(f)

        if not found_files:
            self.log("No standard entry files found (index.html, package.json)", "WARNING")
        else:
            self.log(f"Found entry files: {', '.join(found_files)}")

        return True

    def prepare_build(self) -> bool:
        """Prepare the build environment."""
        self.log("Preparing build environment...")

        # Create output directory
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        # Check for package.json and install dependencies
        package_json = self.config.source_path / "package.json"
        if package_json.exists():
            self.log("Installing npm dependencies...")
            try:
                subprocess.run(
                    ["npm", "install"],
                    cwd=self.config.source_path,
                    check=True,
                    capture_output=True
                )
                self.log("Dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to install dependencies: {e}", "ERROR")
                return False
            except FileNotFoundError:
                self.log("npm not found, skipping dependency installation", "WARNING")

        return True

    def run_build(self) -> bool:
        """Run the build command."""
        if not self.config.build_command:
            self.log("No build command specified, using source directly")
            # Copy source to output
            if self.config.source_path != self.config.output_path:
                shutil.copytree(
                    self.config.source_path,
                    self.config.output_path / self.config.publish_dir,
                    dirs_exist_ok=True
                )
            return True

        self.log(f"Running build: {self.config.build_command}")
        try:
            subprocess.run(
                self.config.build_command,
                shell=True,
                cwd=self.config.source_path,
                check=True,
                capture_output=True
            )
            self.log("Build completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Build failed: {e.stderr.decode()}", "ERROR")
            return False

    def enhance_accessibility(self) -> Dict[str, Any]:
        """Apply accessibility enhancements to HTML files."""
        self.log("Applying accessibility enhancements...")

        enhancements = {
            "files_processed": 0,
            "issues_fixed": 0,
            "changes": []
        }

        publish_path = self.config.output_path / self.config.publish_dir
        if not publish_path.exists():
            publish_path = self.config.output_path

        for html_file in publish_path.rglob("*.html"):
            content = html_file.read_text(encoding='utf-8', errors='ignore')
            original_content = content

            # Add lang attribute if missing
            if '<html' in content and 'lang=' not in content:
                content = content.replace('<html', '<html lang="en"')
                enhancements["changes"].append(f"{html_file.name}: Added lang attribute")
                enhancements["issues_fixed"] += 1

            # Add viewport meta if missing
            if '<head>' in content and 'viewport' not in content:
                content = content.replace(
                    '<head>',
                    '<head>\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">'
                )
                enhancements["changes"].append(f"{html_file.name}: Added viewport meta")
                enhancements["issues_fixed"] += 1

            # Add skip link if missing
            if '<body>' in content and 'skip' not in content.lower():
                skip_link = '''
    <a href="#main-content" class="skip-link" style="position:absolute;left:-9999px;top:auto;width:1px;height:1px;overflow:hidden;">Skip to main content</a>
    <style>.skip-link:focus{position:fixed;top:0;left:0;width:auto;height:auto;padding:10px;background:#000;color:#fff;z-index:9999;}</style>'''
                content = content.replace('<body>', f'<body>{skip_link}')
                enhancements["changes"].append(f"{html_file.name}: Added skip link")
                enhancements["issues_fixed"] += 1

            # Save if changed
            if content != original_content:
                html_file.write_text(content, encoding='utf-8')
                enhancements["files_processed"] += 1

        self.log(f"Accessibility: {enhancements['issues_fixed']} issues fixed in {enhancements['files_processed']} files")
        return enhancements

    def generate_manifest(self) -> Dict[str, Any]:
        """Generate deployment manifest."""
        manifest = {
            "name": self.config.name,
            "version": "1.0.0",
            "deployed_at": datetime.now().isoformat(),
            "platform": self.config.platform,
            "environment": self.config.environment,
            "source": str(self.config.source_path),
            "output": str(self.config.output_path),
            "accessibility_level": self.config.accessibility_level,
            "files": []
        }

        publish_path = self.config.output_path / self.config.publish_dir
        if not publish_path.exists():
            publish_path = self.config.output_path

        for f in publish_path.rglob("*"):
            if f.is_file():
                manifest["files"].append({
                    "path": str(f.relative_to(publish_path)),
                    "size": f.stat().st_size
                })

        manifest["total_files"] = len(manifest["files"])
        manifest["total_size"] = sum(f["size"] for f in manifest["files"])

        # Save manifest
        manifest_path = self.config.output_path / "deployment-manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.log(f"Generated manifest: {manifest['total_files']} files, {manifest['total_size']} bytes")
        return manifest

    def deploy_to_netlify(self) -> DeploymentResult:
        """Deploy to Netlify."""
        self.log("Deploying to Netlify...")

        publish_path = self.config.output_path / self.config.publish_dir
        if not publish_path.exists():
            publish_path = self.config.output_path

        try:
            cmd = ["netlify", "deploy", "--prod", "--dir", str(publish_path)]
            if self.config.domain:
                cmd.extend(["--site", self.config.domain])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Parse URL from output
                url = None
                for line in result.stdout.split('\n'):
                    if 'Website URL:' in line or 'Live URL:' in line:
                        url = line.split()[-1]
                        break

                return DeploymentResult(
                    success=True,
                    url=url,
                    deploy_id=f"netlify-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
            else:
                return DeploymentResult(
                    success=False,
                    errors=[result.stderr]
                )
        except FileNotFoundError:
            return DeploymentResult(
                success=False,
                errors=["Netlify CLI not found. Install with: npm install -g netlify-cli"]
            )

    def deploy(self) -> DeploymentResult:
        """Run full deployment pipeline."""
        start_time = datetime.now()
        self.log(f"Starting deployment: {self.config.name}")

        # Validate
        if not self.validate_source():
            return DeploymentResult(success=False, errors=["Source validation failed"])

        # Prepare
        if not self.prepare_build():
            return DeploymentResult(success=False, errors=["Build preparation failed"])

        # Build
        if not self.run_build():
            return DeploymentResult(success=False, errors=["Build failed"])

        # Accessibility
        accessibility_result = self.enhance_accessibility()

        # Generate manifest
        self.generate_manifest()

        # Deploy based on platform
        if self.config.platform == "netlify":
            result = self.deploy_to_netlify()
        else:
            # For other platforms, just mark as ready
            result = DeploymentResult(
                success=True,
                deploy_id=f"local-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

        # Calculate duration
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        result.accessibility_score = 1.0 if accessibility_result["issues_fixed"] == 0 else 0.9

        self.log(f"Deployment {'succeeded' if result.success else 'failed'} in {result.duration_seconds:.1f}s")

        return result


def create_portfolio_config(
    name: str,
    source: str,
    output: str = None,
    platform: str = "netlify"
) -> DeploymentConfig:
    """Create a deployment configuration."""
    source_path = Path(source)
    output_path = Path(output) if output else source_path / "deploy"

    return DeploymentConfig(
        name=name,
        source_path=source_path,
        output_path=output_path,
        platform=platform
    )


if __name__ == "__main__":
    # Demo
    print("Portfolio Deployer - Demo Mode")
    print("=" * 40)

    config = create_portfolio_config(
        name="demo-portfolio",
        source=".",
        platform="local"
    )

    deployer = PortfolioDeployer(config)
    result = deployer.deploy()

    print(f"\nResult: {'Success' if result.success else 'Failed'}")
    if result.errors:
        print(f"Errors: {result.errors}")
