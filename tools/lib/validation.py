#!/usr/bin/env python3
"""
validation.py - Unified Validation Library

Provides validation functionality for schemas, structures, and compliance.
Consolidates logic from:
- tools/governance/enforce.py
- tools/governance/compliance_validator.py

Usage:
    from tools.lib.validation import Validator
    
    validator = Validator()
    is_valid, errors = validator.validate_schema(data, schema)
    is_valid, errors = validator.validate_structure(path, requirements)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yaml
import jsonschema


class Validator:
    """Unified validator for schemas, structures, and compliance."""
    
    # Tier-based requirements
    TIER_REQUIREMENTS = {
        1: {  # Mission-critical
            "required_files": [".meta/repo.yaml", "README.md", ".github/CODEOWNERS",
                               ".github/workflows/ci.yml", "tests/"],
            "coverage_min": 90,
            "docs_profile": "standard"
        },
        2: {  # Important
            "required_files": [".meta/repo.yaml", "README.md", ".github/CODEOWNERS",
                               ".github/workflows/ci.yml"],
            "coverage_min": 80,
            "docs_profile": "standard"
        },
        3: {  # Experimental
            "required_files": [".meta/repo.yaml", "README.md"],
            "coverage_min": 60,
            "docs_profile": "minimal"
        },
        4: {  # Unknown
            "required_files": [".meta/repo.yaml", "README.md"],
            "coverage_min": 0,
            "docs_profile": "minimal"
        }
    }
    
    # Docker security patterns
    DOCKER_SECURITY_PATTERNS = {
        "has_user": re.compile(r'^USER\s+(?!root)', re.MULTILINE),
        "has_healthcheck": re.compile(r'^HEALTHCHECK\s+', re.MULTILINE),
        "latest_tag": re.compile(r'^FROM\s+\S+:latest', re.MULTILINE),
        "untagged_from": re.compile(r'^FROM\s+([^:\s@]+)\s*$', re.MULTILINE),
        "secret_in_env": re.compile(r'^ENV\s+\S*(PASSWORD|SECRET|TOKEN|API_KEY|PRIVATE_KEY)', 
                                   re.MULTILINE | re.IGNORECASE),
    }
    
    def __init__(self):
        """Initialize validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_schema(self, data: dict, schema: dict) -> Tuple[bool, List[str]]:
        """
        Validate data against JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON schema
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            jsonschema.validate(data, schema)
            return True, []
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            if e.path:
                errors.append(f"  Path: {'.'.join(str(p) for p in e.path)}")
            return False, errors
        except jsonschema.SchemaError as e:
            errors.append(f"Invalid schema: {e.message}")
            return False, errors
    
    def validate_structure(self, path: Path, requirements: dict) -> Tuple[bool, List[str]]:
        """
        Validate repository structure against requirements.
        
        Args:
            path: Path to repository
            requirements: Dictionary of requirements (tier-based or custom)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        path = Path(path)
        
        if not path.exists():
            return False, [f"Path does not exist: {path}"]
        
        # Check required files
        required_files = requirements.get("required_files", [])
        for required in required_files:
            file_path = path / required
            
            # Handle directory requirements (ending with /)
            if required.endswith("/"):
                if not file_path.parent.exists() and not (path / required.rstrip("/")).is_dir():
                    errors.append(f"Missing required directory: {required}")
            elif not file_path.exists():
                errors.append(f"Missing required file: {required}")
        
        return len(errors) == 0, errors
    
    def validate_metadata(self, path: Path, schema: Optional[dict] = None) -> Tuple[bool, List[str]]:
        """
        Validate .meta/repo.yaml file.
        
        Args:
            path: Path to repository
            schema: Optional JSON schema for validation
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        meta_file = Path(path) / ".meta" / "repo.yaml"
        
        if not meta_file.exists():
            return False, [".meta/repo.yaml is required"]
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return False, [f"Invalid YAML in .meta/repo.yaml: {e}"]
        
        if not metadata:
            return False, [".meta/repo.yaml is empty"]
        
        # Schema validation if provided
        if schema:
            is_valid, schema_errors = self.validate_schema(metadata, schema)
            if not is_valid:
                errors.extend(schema_errors)
        
        # Required fields check
        required_fields = ["type", "language"]
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
        
        # Tier validation
        tier = metadata.get("tier", 4)
        if not isinstance(tier, int) or tier < 1 or tier > 4:
            errors.append(f"Tier should be integer 1-4, got: {tier}")
        
        return len(errors) == 0, errors
    
    def validate_docker(self, dockerfile_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate Dockerfile against security best practices.
        
        Args:
            dockerfile_path: Path to Dockerfile
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        warnings = []
        
        if not dockerfile_path.exists():
            return False, ["Dockerfile not found"]
        
        try:
            content = dockerfile_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = dockerfile_path.read_text(encoding='latin-1')
        
        # Check for USER directive (non-root)
        if not self.DOCKER_SECURITY_PATTERNS["has_user"].search(content):
            errors.append("Dockerfile must run as non-root user (USER directive)")
        
        # Check for HEALTHCHECK
        if not self.DOCKER_SECURITY_PATTERNS["has_healthcheck"].search(content):
            warnings.append("Dockerfile should include HEALTHCHECK")
        
        # Check for :latest tag
        if self.DOCKER_SECURITY_PATTERNS["latest_tag"].search(content):
            errors.append("Do not use :latest tag in FROM directive")
        
        # Check for untagged FROM
        if self.DOCKER_SECURITY_PATTERNS["untagged_from"].search(content):
            errors.append("FROM directive must specify version tag")
        
        # Check for secrets in ENV
        if self.DOCKER_SECURITY_PATTERNS["secret_in_env"].search(content):
            errors.append("Do not hardcode secrets in ENV directives")
        
        # Combine errors and warnings
        all_issues = errors + warnings
        return len(errors) == 0, all_issues
    
    def validate_tier_compliance(self, path: Path, tier: int) -> Tuple[bool, List[str]]:
        """
        Validate repository compliance for specific tier.
        
        Args:
            path: Path to repository
            tier: Tier level (1-4)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        requirements = self.TIER_REQUIREMENTS.get(tier, self.TIER_REQUIREMENTS[4])
        return self.validate_structure(path, requirements)
    
    def validate_codeowners(self, path: Path) -> Tuple[bool, List[str]]:
        """
        Validate CODEOWNERS file exists and has content.
        
        Args:
            path: Path to repository
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        codeowners = Path(path) / ".github" / "CODEOWNERS"
        
        if not codeowners.exists():
            return False, ["CODEOWNERS file not found"]
        
        try:
            content = codeowners.read_text(encoding='utf-8')
            if not content.strip():
                errors.append("CODEOWNERS file is empty")
            
            # Check for at least one valid ownership rule
            has_rule = False
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    has_rule = True
                    break
            
            if not has_rule:
                errors.append("CODEOWNERS has no ownership rules")
        except Exception as e:
            errors.append(f"Error reading CODEOWNERS: {e}")
        
        return len(errors) == 0, errors
    
    def validate_workflows(self, path: Path) -> Tuple[bool, List[str]]:
        """
        Validate CI/CD workflow configuration.
        
        Args:
            path: Path to repository
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        warnings = []
        workflows_dir = Path(path) / ".github" / "workflows"
        
        if not workflows_dir.exists():
            return False, ["CI/CD workflows directory not found"]
        
        # Check for ci.yml or similar
        ci_files = list(workflows_dir.glob("ci*.yml")) + list(workflows_dir.glob("ci*.yaml"))
        if not ci_files:
            warnings.append("No CI workflow found (ci.yml)")
        
        # Validate workflow files
        for wf_file in workflows_dir.glob("*.y*ml"):
            try:
                with open(wf_file, 'r', encoding='utf-8') as f:
                    workflow = yaml.safe_load(f)
                
                if workflow:
                    # Check for permissions block (security hardening)
                    if 'permissions' not in workflow:
                        warnings.append(f"Workflow {wf_file.name} missing permissions block")
            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML in {wf_file.name}: {e}")
        
        all_issues = errors + warnings
        return len(errors) == 0, all_issues