"""
Test suite for .metaHub/scripts/enforce.py - Policy Enforcement.

Tests cover:
- Schema validation
- Docker security checks
- Repository structure validation
- CODEOWNERS validation
- Workflow checks
- Organization-level enforcement
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / ".metaHub" / "scripts"))

from enforce import PolicyEnforcer, enforce_organization


class TestPolicyEnforcer:
    """Tests for PolicyEnforcer class."""

    @pytest.fixture
    def mock_central_repo(self, tmp_path):
        """Create mock central repo with schema."""
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()
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
            "additionalProperties": True
        }
        (schemas / "repo-schema.json").write_text(json.dumps(schema))

        return tmp_path

    @pytest.fixture
    def valid_repo(self, tmp_path, mock_central_repo):
        """Create a valid repository structure."""
        repo = tmp_path / "test-repo"
        repo.mkdir()

        # .meta/repo.yaml
        (repo / ".meta").mkdir()
        (repo / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "library",
            "language": "python",
            "tier": 2,
            "status": "active"
        }))

        # README.md
        (repo / "README.md").write_text("# Test Repository\n\nThis is a test.")

        # .github/CODEOWNERS
        github = repo / ".github"
        github.mkdir()
        (github / "CODEOWNERS").write_text("* @test-org\n")

        # .github/workflows/ci.yml
        workflows = github / "workflows"
        workflows.mkdir()
        (workflows / "ci.yml").write_text(yaml.dump({
            "name": "CI",
            "on": {"push": {"branches": ["main"]}},
            "permissions": {"contents": "read"},
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}}
        }))

        # tests/
        (repo / "tests").mkdir()

        return repo, mock_central_repo

    def test_valid_repo_passes(self, valid_repo):
        """Valid repository should pass all checks."""
        repo_path, central_path = valid_repo
        enforcer = PolicyEnforcer(repo_path, central_path=central_path)
        violations, warnings = enforcer.check_all()

        assert violations == 0

    def test_missing_metadata_is_violation(self, tmp_path, mock_central_repo):
        """Missing .meta/repo.yaml should be a violation."""
        repo = tmp_path / "no-meta-repo"
        repo.mkdir()
        (repo / "README.md").write_text("# No Meta")

        enforcer = PolicyEnforcer(repo, central_path=mock_central_repo)
        violations, warnings = enforcer.check_all()

        assert violations > 0
        assert any("metadata" in v["check"] for v in enforcer.violations)

    def test_invalid_schema_is_violation(self, tmp_path, mock_central_repo):
        """Invalid schema should be a violation."""
        repo = tmp_path / "bad-schema-repo"
        repo.mkdir()
        (repo / ".meta").mkdir()
        # Missing required 'type' field
        (repo / ".meta" / "repo.yaml").write_text(yaml.dump({
            "language": "python"
        }))
        (repo / "README.md").write_text("# Bad Schema")

        enforcer = PolicyEnforcer(repo, central_path=mock_central_repo)
        violations, warnings = enforcer.check_all()

        assert violations > 0
        assert any("Schema validation failed" in v["message"] for v in enforcer.violations)

    def test_missing_readme_is_violation(self, tmp_path, mock_central_repo):
        """Missing README.md should be a violation."""
        repo = tmp_path / "no-readme-repo"
        repo.mkdir()
        (repo / ".meta").mkdir()
        (repo / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "tool",
            "language": "python",
            "tier": 4
        }))

        enforcer = PolicyEnforcer(repo, central_path=mock_central_repo)
        violations, warnings = enforcer.check_all()

        assert violations > 0
        assert any("readme" in v["check"] for v in enforcer.violations)


class TestDockerChecks:
    """Tests for Dockerfile validation."""

    @pytest.fixture
    def repo_with_dockerfile(self, tmp_path, mock_central_repo):
        """Create repo with Dockerfile."""
        repo = tmp_path / "docker-repo"
        repo.mkdir()
        (repo / ".meta").mkdir()
        (repo / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "tool",
            "language": "python",
            "tier": 4
        }))
        (repo / "README.md").write_text("# Docker Repo")

        return repo, mock_central_repo

    @pytest.fixture
    def mock_central_repo(self, tmp_path):
        """Create mock central repo."""
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()
        schemas = metahub / "schemas"
        schemas.mkdir()
        schema = {
            "type": "object",
            "required": ["type", "language"],
            "properties": {
                "type": {"type": "string"},
                "language": {"type": "string"},
                "tier": {"type": "integer"}
            },
            "additionalProperties": True
        }
        (schemas / "repo-schema.json").write_text(json.dumps(schema))
        return tmp_path

    def test_good_dockerfile_passes(self, repo_with_dockerfile):
        """Good Dockerfile should pass."""
        repo, central = repo_with_dockerfile
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir requirements.txt

USER appuser

HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "app.py"]
"""
        (repo / "Dockerfile").write_text(dockerfile_content)

        enforcer = PolicyEnforcer(repo, central_path=central)
        enforcer.check_docker()

        docker_violations = [v for v in enforcer.violations if v["check"] == "docker"]
        assert len(docker_violations) == 0

    def test_latest_tag_is_violation(self, repo_with_dockerfile):
        """Using :latest tag should be a violation."""
        repo, central = repo_with_dockerfile
        dockerfile_content = """
FROM python:latest

USER appuser

HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "app.py"]
"""
        (repo / "Dockerfile").write_text(dockerfile_content)

        enforcer = PolicyEnforcer(repo, central_path=central)
        enforcer.check_docker()

        assert any(":latest" in v["message"] for v in enforcer.violations)

    def test_missing_user_is_violation(self, repo_with_dockerfile):
        """Missing USER directive should be a violation."""
        repo, central = repo_with_dockerfile
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "app.py"]
"""
        (repo / "Dockerfile").write_text(dockerfile_content)

        enforcer = PolicyEnforcer(repo, central_path=central)
        enforcer.check_docker()

        assert any("non-root" in v["message"] for v in enforcer.violations)

    def test_secrets_in_env_is_violation(self, repo_with_dockerfile):
        """Secrets in ENV should be a violation."""
        repo, central = repo_with_dockerfile
        dockerfile_content = """
FROM python:3.11-slim

USER appuser

ENV DATABASE_PASSWORD=secret123

HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "app.py"]
"""
        (repo / "Dockerfile").write_text(dockerfile_content)

        enforcer = PolicyEnforcer(repo, central_path=central)
        enforcer.check_docker()

        assert any("secrets" in v["message"].lower() for v in enforcer.violations)


class TestWorkflowChecks:
    """Tests for workflow validation."""

    @pytest.fixture
    def repo_with_workflow(self, tmp_path, mock_central_repo):
        """Create repo with workflow."""
        repo = tmp_path / "workflow-repo"
        repo.mkdir()
        (repo / ".meta").mkdir()
        (repo / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "tool",
            "language": "python",
            "tier": 4
        }))
        (repo / "README.md").write_text("# Workflow Repo")
        (repo / ".github").mkdir()
        (repo / ".github" / "workflows").mkdir()

        return repo, mock_central_repo

    @pytest.fixture
    def mock_central_repo(self, tmp_path):
        """Create mock central repo."""
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()
        schemas = metahub / "schemas"
        schemas.mkdir()
        schema = {
            "type": "object",
            "required": ["type", "language"],
            "properties": {
                "type": {"type": "string"},
                "language": {"type": "string"},
                "tier": {"type": "integer"}
            },
            "additionalProperties": True
        }
        (schemas / "repo-schema.json").write_text(json.dumps(schema))
        return tmp_path

    def test_missing_permissions_is_warning(self, repo_with_workflow):
        """Missing permissions block should be a warning."""
        repo, central = repo_with_workflow
        workflow = {
            "name": "CI",
            "on": {"push": {"branches": ["main"]}},
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}}
        }
        (repo / ".github" / "workflows" / "ci.yml").write_text(yaml.dump(workflow))

        enforcer = PolicyEnforcer(repo, central_path=central)
        enforcer.check_workflows()

        assert any("permissions" in w["message"] for w in enforcer.warnings)

    def test_continue_on_error_is_warning(self, repo_with_workflow):
        """continue-on-error should be a warning."""
        repo, central = repo_with_workflow
        workflow = {
            "name": "CI",
            "on": {"push": {"branches": ["main"]}},
            "permissions": {"contents": "read"},
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"run": "echo test", "continue-on-error": True}]
                }
            }
        }
        (repo / ".github" / "workflows" / "ci.yml").write_text(yaml.dump(workflow))

        enforcer = PolicyEnforcer(repo, central_path=central)
        enforcer.check_workflows()

        assert any("continue-on-error" in w["message"] for w in enforcer.warnings)


class TestOrganizationEnforcement:
    """Tests for organization-level enforcement."""

    @pytest.fixture
    def mock_org(self, tmp_path):
        """Create mock organization."""
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()
        schemas = metahub / "schemas"
        schemas.mkdir()
        schema = {
            "type": "object",
            "required": ["type", "language"],
            "properties": {
                "type": {"type": "string"},
                "language": {"type": "string"},
                "tier": {"type": "integer"}
            },
            "additionalProperties": True
        }
        (schemas / "repo-schema.json").write_text(json.dumps(schema))

        org = tmp_path / "test-org"
        org.mkdir()

        # Valid repo
        repo1 = org / "valid-repo"
        repo1.mkdir()
        (repo1 / ".meta").mkdir()
        (repo1 / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "library",
            "language": "python",
            "tier": 4
        }))
        (repo1 / "README.md").write_text("# Valid Repo")

        # Invalid repo (no metadata)
        repo2 = org / "invalid-repo"
        repo2.mkdir()
        (repo2 / "README.md").write_text("# Invalid Repo")

        return org, tmp_path

    def test_enforces_all_repos(self, mock_org):
        """Should enforce all repos in organization."""
        org_path, central_path = mock_org
        results = enforce_organization(org_path, central_path=central_path)

        assert results["summary"]["total_repos"] == 2
        assert results["summary"]["passed"] >= 1
        assert results["summary"]["failed"] >= 1

    def test_aggregates_violations(self, mock_org):
        """Should aggregate violation counts."""
        org_path, central_path = mock_org
        results = enforce_organization(org_path, central_path=central_path)

        assert results["summary"]["total_violations"] >= 1


class TestReportGeneration:
    """Tests for report generation."""

    @pytest.fixture
    def enforcer_with_issues(self, tmp_path):
        """Create enforcer with some issues."""
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()
        schemas = metahub / "schemas"
        schemas.mkdir()
        schema = {
            "type": "object",
            "required": ["type", "language"],
            "properties": {"type": {"type": "string"}, "language": {"type": "string"}},
            "additionalProperties": True
        }
        (schemas / "repo-schema.json").write_text(json.dumps(schema))

        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / "README.md").write_text("# Short")  # Too short, will warn

        enforcer = PolicyEnforcer(repo, central_path=tmp_path)
        enforcer.check_all()

        return enforcer

    def test_text_report(self, enforcer_with_issues):
        """Should generate text report."""
        report = enforcer_with_issues.report(fmt='text')

        assert "Enforcement Report" in report
        assert "Violations:" in report

    def test_json_report(self, enforcer_with_issues):
        """Should generate JSON report."""
        report = enforcer_with_issues.report(fmt='json')
        parsed = json.loads(report)

        assert "violations" in parsed
        assert "warnings" in parsed
        assert "summary" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
