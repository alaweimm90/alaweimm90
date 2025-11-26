"""
Tests for .metaHub/scripts/meta.py - Meta Auditor & Promotion System.

Tests cover:
- MetaAuditor class
- ProjectPromoter class
- Gap detection and scoring
- Project promotion workflow
- CLI commands
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / ".metaHub" / "scripts"))

from meta import MetaAuditor, ProjectPromoter, GapSeverity, GovernanceGap, ProjectAudit


class TestMetaAuditor:
    """Tests for MetaAuditor class."""

    @pytest.fixture
    def mock_portfolio(self, tmp_path):
        """Create mock portfolio structure."""
        # Create central repo
        central = tmp_path
        metahub = central / ".metaHub"
        metahub.mkdir()

        # Create templates
        templates = metahub / "templates"
        templates.mkdir()
        precommit = templates / "pre-commit"
        precommit.mkdir()
        (precommit / "python.yaml").write_text("# Python config")
        (precommit / "generic.yaml").write_text("# Generic config")

        # Create organizations
        orgs = central / "organizations"
        orgs.mkdir()

        # Org 1 with compliant repo
        org1 = orgs / "test-org-1"
        org1.mkdir()

        compliant = org1 / "lib-compliant"
        compliant.mkdir()
        (compliant / "README.md").write_text("# Compliant Library\n\nThis is a well-documented library.")
        (compliant / "LICENSE").write_text("MIT License")
        (compliant / ".meta").mkdir()
        (compliant / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "library",
            "language": "python",
            "tier": 2
        }))
        (compliant / ".github").mkdir()
        (compliant / ".github" / "CODEOWNERS").write_text("* @owner")
        (compliant / ".github" / "workflows").mkdir()
        (compliant / ".github" / "workflows" / "ci.yml").write_text("name: CI")
        (compliant / "tests").mkdir()
        (compliant / "pyproject.toml").write_text("[project]\nname = 'lib'")

        # Non-compliant repo
        noncompliant = org1 / "demo-broken"
        noncompliant.mkdir()
        (noncompliant / "README.md").write_text("# Short")  # Too short

        return central

    def test_scan_all_projects(self, mock_portfolio):
        """Should scan all projects."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects()

        assert len(audits) == 2

    def test_detects_missing_metadata(self, mock_portfolio):
        """Should detect missing .meta/repo.yaml."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects()

        broken = next(a for a in audits if a.name == "demo-broken")
        p0_gaps = [g for g in broken.gaps if g.severity == GapSeverity.P0]

        assert len(p0_gaps) >= 1
        assert any("metadata" in g.category for g in p0_gaps)

    def test_detects_missing_license(self, mock_portfolio):
        """Should detect missing LICENSE file."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects()

        broken = next(a for a in audits if a.name == "demo-broken")
        license_gaps = [g for g in broken.gaps if "license" in g.message.lower()]

        assert len(license_gaps) >= 1

    def test_compliant_repo_high_score(self, mock_portfolio):
        """Compliant repo should have high score."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects()

        compliant = next(a for a in audits if a.name == "lib-compliant")

        assert compliant.compliance_score >= 80
        assert compliant.promotion_ready is True

    def test_noncompliant_repo_low_score(self, mock_portfolio):
        """Non-compliant repo should have low score."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects()

        broken = next(a for a in audits if a.name == "demo-broken")

        assert broken.compliance_score < 70
        assert broken.promotion_ready is False

    def test_filter_by_organization(self, mock_portfolio):
        """Should filter by organization."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects(org_filter="test-org-1")

        assert len(audits) == 2
        assert all(a.organization == "test-org-1" for a in audits)

    def test_filter_nonexistent_org(self, mock_portfolio):
        """Should return empty for nonexistent org."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects(org_filter="nonexistent")

        assert len(audits) == 0

    def test_language_detection(self, mock_portfolio):
        """Should detect language from project files."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects()

        compliant = next(a for a in audits if a.name == "lib-compliant")
        assert compliant.language == "python"

    def test_type_inference(self, mock_portfolio):
        """Should infer type from name prefix."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        audits = auditor.scan_all_projects()

        broken = next(a for a in audits if a.name == "demo-broken")
        assert broken.repo_type == "demo"

    def test_generate_json_report(self, mock_portfolio):
        """Should generate valid JSON report."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        auditor.scan_all_projects()
        report = auditor.generate_report(fmt='json')

        data = json.loads(report)
        assert "total_projects" in data
        assert "projects" in data
        assert "summary" in data

    def test_generate_markdown_report(self, mock_portfolio):
        """Should generate Markdown report."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        auditor.scan_all_projects()
        report = auditor.generate_report(fmt='markdown')

        assert "# Portfolio Compliance Audit Report" in report
        assert "test-org-1" in report

    def test_generate_text_report(self, mock_portfolio):
        """Should generate text report."""
        auditor = MetaAuditor(base_path=mock_portfolio)
        auditor.scan_all_projects()
        report = auditor.generate_report(fmt='text')

        assert "META AUDITOR" in report
        assert "Total Projects" in report


class TestProjectPromoter:
    """Tests for ProjectPromoter class."""

    @pytest.fixture
    def mock_project(self, tmp_path):
        """Create mock project for promotion."""
        central = tmp_path
        metahub = central / ".metaHub"
        metahub.mkdir()

        templates = metahub / "templates"
        templates.mkdir()
        precommit = templates / "pre-commit"
        precommit.mkdir()
        (precommit / "python.yaml").write_text("repos:\n  - repo: test")
        (precommit / "generic.yaml").write_text("repos: []")

        orgs = central / "organizations"
        orgs.mkdir()
        org = orgs / "test-org"
        org.mkdir()

        project = org / "my-project"
        project.mkdir()
        (project / "README.md").write_text("# My Project")
        (project / "pyproject.toml").write_text("[project]\nname = 'my-project'")

        return {
            "central": central,
            "project": project,
            "org": "test-org"
        }

    def test_promote_creates_metadata(self, mock_project):
        """Should create .meta/repo.yaml."""
        promoter = ProjectPromoter(base_path=mock_project["central"])
        result = promoter.promote_project(
            mock_project["project"],
            mock_project["org"]
        )

        assert result["success"] is True
        assert (mock_project["project"] / ".meta" / "repo.yaml").exists()

    def test_promote_creates_codeowners(self, mock_project):
        """Should create CODEOWNERS."""
        promoter = ProjectPromoter(base_path=mock_project["central"])
        result = promoter.promote_project(
            mock_project["project"],
            mock_project["org"]
        )

        assert (mock_project["project"] / ".github" / "CODEOWNERS").exists()

    def test_promote_creates_ci_workflow(self, mock_project):
        """Should create CI workflow."""
        promoter = ProjectPromoter(base_path=mock_project["central"])
        result = promoter.promote_project(
            mock_project["project"],
            mock_project["org"]
        )

        assert (mock_project["project"] / ".github" / "workflows" / "ci.yml").exists()

    def test_promote_creates_precommit(self, mock_project):
        """Should create pre-commit config."""
        promoter = ProjectPromoter(base_path=mock_project["central"])
        result = promoter.promote_project(
            mock_project["project"],
            mock_project["org"]
        )

        assert (mock_project["project"] / ".pre-commit-config.yaml").exists()

    def test_promote_dry_run(self, mock_project):
        """Dry run should not create files."""
        promoter = ProjectPromoter(base_path=mock_project["central"])
        result = promoter.promote_project(
            mock_project["project"],
            mock_project["org"],
            dry_run=True
        )

        assert result["success"] is True
        assert not (mock_project["project"] / ".meta" / "repo.yaml").exists()

    def test_promote_detects_language(self, mock_project):
        """Should detect Python from pyproject.toml."""
        promoter = ProjectPromoter(base_path=mock_project["central"])
        result = promoter.promote_project(
            mock_project["project"],
            mock_project["org"]
        )

        # Check metadata has correct language
        meta_file = mock_project["project"] / ".meta" / "repo.yaml"
        with open(meta_file) as f:
            metadata = yaml.safe_load(f)

        assert metadata["language"] == "python"

    def test_promote_nonexistent_project(self, mock_project):
        """Should fail for nonexistent project."""
        promoter = ProjectPromoter(base_path=mock_project["central"])
        result = promoter.promote_project(
            Path("/nonexistent/path"),
            mock_project["org"]
        )

        assert result["success"] is False
        assert "error" in result


class TestGapSeverity:
    """Tests for gap severity and scoring."""

    def test_p0_blocks_promotion(self):
        """P0 gaps should block promotion."""
        audit = ProjectAudit(
            name="test",
            path="/test",
            organization="org",
            language="python",
            repo_type="library",
            has_metadata=True,
            gaps=[GovernanceGap(
                severity=GapSeverity.P0,
                category="security",
                message="Critical issue"
            )],
            compliance_score=50.0
        )

        # P0 gaps mean not promotion ready
        assert audit.promotion_ready is False

    def test_score_calculation(self):
        """Score should decrease with gaps."""
        # No gaps = 100
        audit1 = ProjectAudit(
            name="test1",
            path="/test1",
            organization="org",
            language="python",
            repo_type="library",
            has_metadata=True,
            gaps=[],
            compliance_score=100.0
        )

        # With P1 gap = lower score
        audit2 = ProjectAudit(
            name="test2",
            path="/test2",
            organization="org",
            language="python",
            repo_type="library",
            has_metadata=True,
            gaps=[GovernanceGap(
                severity=GapSeverity.P1,
                category="structure",
                message="Missing file"
            )],
            compliance_score=85.0
        )

        assert audit1.compliance_score > audit2.compliance_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
