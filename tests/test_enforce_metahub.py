"""
Tests for .metaHub/scripts/enforce.py

Tests cover:
- PolicyEnforcer class
- Schema validation
- Docker security checks
- Repository structure validation
- CODEOWNERS validation
- Workflow validation
- CLI interface
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

# Add .metaHub/scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / ".metaHub" / "scripts"))

from enforce import PolicyEnforcer, enforce_organization


class TestPolicyEnforcer:
    """Tests for PolicyEnforcer class."""

    @pytest.fixture
    def enforcer_setup(self, tmp_path):
        """Create a basic enforcer setup for testing."""
        # Create central repo structure
        central = tmp_path / "central"
        central.mkdir()
        metahub = central / ".metaHub"
        metahub.mkdir()

        # Create schema
        schemas = metahub / "schemas"
        schemas.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["type", "language"],
            "properties": {
                "type": {"type": "string"},
                "language": {"type": "string"},
                "tier": {"type": "integer", "minimum": 1, "maximum": 4}
            },
            "additionalProperties": False
        }
        (schemas / "repo-schema.json").write_text(json.dumps(schema))

        # Create repo to test
        repo = tmp_path / "test-repo"
        repo.mkdir()

        return {
            "central": central,
            "repo": repo,
            "schema_path": schemas / "repo-schema.json"
        }

    def test_check_metadata_missing(self, enforcer_setup):
        """Should report violation when .meta/repo.yaml is missing."""
        enforcer = PolicyEnforcer(
            enforcer_setup["repo"],
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_metadata()

        assert result is False
        assert len(enforcer.violations) == 1
        assert "required" in enforcer.violations[0]["message"].lower()

    def test_check_metadata_valid(self, enforcer_setup):
        """Should pass when .meta/repo.yaml is valid."""
        repo = enforcer_setup["repo"]
        meta_dir = repo / ".meta"
        meta_dir.mkdir()

        metadata = {
            "type": "library",
            "language": "python"
        }
        (meta_dir / "repo.yaml").write_text(yaml.dump(metadata))

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_metadata()

        assert result is True
        assert len(enforcer.violations) == 0

    def test_check_metadata_invalid_yaml(self, enforcer_setup):
        """Should report violation for invalid YAML."""
        repo = enforcer_setup["repo"]
        meta_dir = repo / ".meta"
        meta_dir.mkdir()
        (meta_dir / "repo.yaml").write_text("invalid: yaml: content:")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_metadata()

        assert result is False
        assert len(enforcer.violations) >= 1

    def test_check_metadata_schema_violation(self, enforcer_setup):
        """Should report violation when schema validation fails."""
        repo = enforcer_setup["repo"]
        meta_dir = repo / ".meta"
        meta_dir.mkdir()

        # Missing required 'language' field
        metadata = {"type": "library"}
        (meta_dir / "repo.yaml").write_text(yaml.dump(metadata))

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_metadata()

        assert result is False
        assert any("language" in v["message"].lower() for v in enforcer.violations)

    def test_check_docker_no_dockerfiles(self, enforcer_setup):
        """Should pass when no Dockerfiles exist."""
        enforcer = PolicyEnforcer(
            enforcer_setup["repo"],
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_docker()

        assert result is True
        assert len(enforcer.violations) == 0

    def test_check_docker_missing_user(self, enforcer_setup):
        """Should report violation for Dockerfile without USER directive."""
        repo = enforcer_setup["repo"]
        dockerfile = repo / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
WORKDIR /app
COPY . .
CMD ["python", "app.py"]
""")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_docker()

        assert result is False
        assert any("non-root" in v["message"].lower() for v in enforcer.violations)

    def test_check_docker_latest_tag(self, enforcer_setup):
        """Should report violation for :latest tag."""
        repo = enforcer_setup["repo"]
        dockerfile = repo / "Dockerfile"
        dockerfile.write_text("""FROM python:latest
USER appuser
CMD ["python", "app.py"]
""")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_docker()

        assert result is False
        assert any("latest" in v["message"].lower() for v in enforcer.violations)

    def test_check_docker_valid(self, enforcer_setup):
        """Should pass for valid Dockerfile."""
        repo = enforcer_setup["repo"]
        dockerfile = repo / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11-slim
WORKDIR /app
COPY . .
USER appuser
HEALTHCHECK CMD python -c "print('ok')"
CMD ["python", "app.py"]
""")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_docker()

        assert result is True
        assert len(enforcer.violations) == 0

    def test_check_codeowners_missing_tier1(self, enforcer_setup):
        """Should report violation for missing CODEOWNERS in tier 1 repo."""
        repo = enforcer_setup["repo"]
        meta_dir = repo / ".meta"
        meta_dir.mkdir()
        (meta_dir / "repo.yaml").write_text(yaml.dump({
            "type": "library",
            "language": "python",
            "tier": 1
        }))

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        enforcer.check_metadata()
        result = enforcer.check_codeowners()

        assert result is False
        assert any("codeowners" in v["check"].lower() for v in enforcer.violations)

    def test_check_codeowners_valid(self, enforcer_setup):
        """Should pass for valid CODEOWNERS."""
        repo = enforcer_setup["repo"]
        github_dir = repo / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text("* @owner")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_codeowners()

        assert result is True

    def test_check_readme_missing(self, enforcer_setup):
        """Should report violation for missing README."""
        enforcer = PolicyEnforcer(
            enforcer_setup["repo"],
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_readme()

        assert result is False
        assert any("readme" in v["check"].lower() for v in enforcer.violations)

    def test_check_readme_valid(self, enforcer_setup):
        """Should pass for valid README."""
        repo = enforcer_setup["repo"]
        (repo / "README.md").write_text("# My Project\n\nThis is a description of my project.")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        result = enforcer.check_readme()

        assert result is True

    def test_check_all(self, enforcer_setup):
        """Should run all checks and return counts."""
        repo = enforcer_setup["repo"]

        # Create minimal valid structure
        (repo / "README.md").write_text("# Test\n\nDescription here.")
        meta_dir = repo / ".meta"
        meta_dir.mkdir()
        (meta_dir / "repo.yaml").write_text(yaml.dump({
            "type": "demo",
            "language": "python"
        }))

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        violations, warnings = enforcer.check_all()

        assert isinstance(violations, int)
        assert isinstance(warnings, int)

    def test_report_json(self, enforcer_setup):
        """Should generate valid JSON report."""
        repo = enforcer_setup["repo"]
        (repo / "README.md").write_text("# Test")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        enforcer.check_all()
        report = enforcer.report(fmt='json')

        data = json.loads(report)
        assert "repo" in data
        assert "summary" in data
        assert "violations" in data

    def test_report_text(self, enforcer_setup):
        """Should generate text report."""
        repo = enforcer_setup["repo"]
        (repo / "README.md").write_text("# Test")

        enforcer = PolicyEnforcer(
            repo,
            central_path=enforcer_setup["central"]
        )
        enforcer.check_all()
        report = enforcer.report(fmt='text')

        assert "Enforcement Report" in report
        assert "Summary" in report


class TestEnforceOrganization:
    """Tests for enforce_organization function."""

    def test_enforce_empty_org(self, tmp_path):
        """Should handle empty organization directory."""
        org = tmp_path / "empty-org"
        org.mkdir()

        central = tmp_path / "central"
        central.mkdir()
        (central / ".metaHub").mkdir()

        results = enforce_organization(org, central_path=central)

        assert results["organization"] == "empty-org"
        assert results["summary"]["total_repos"] == 0

    def test_enforce_org_with_repos(self, tmp_path):
        """Should enforce all repos in organization."""
        # Create central
        central = tmp_path / "central"
        central.mkdir()
        metahub = central / ".metaHub"
        metahub.mkdir()
        schemas = metahub / "schemas"
        schemas.mkdir()
        (schemas / "repo-schema.json").write_text(json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["type", "language"],
            "properties": {
                "type": {"type": "string"},
                "language": {"type": "string"}
            }
        }))

        # Create org with repos
        org = tmp_path / "test-org"
        org.mkdir()

        repo1 = org / "repo1"
        repo1.mkdir()
        (repo1 / "README.md").write_text("# Repo 1")

        repo2 = org / "repo2"
        repo2.mkdir()
        (repo2 / "README.md").write_text("# Repo 2")
        meta = repo2 / ".meta"
        meta.mkdir()
        (meta / "repo.yaml").write_text(yaml.dump({
            "type": "library",
            "language": "python"
        }))

        results = enforce_organization(org, central_path=central)

        assert results["summary"]["total_repos"] == 2
        assert results["summary"]["failed"] >= 1  # repo1 missing metadata
