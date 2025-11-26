"""
Test suite for catalog.py - Service Catalog Generator.

Tests cover:
- Organization scanning
- Repository metadata extraction
- Catalog generation (JSON, Markdown, HTML)
- Summary statistics
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / ".metaHub" / "scripts"))

from catalog import CatalogBuilder


class TestCatalogBuilder:
    """Tests for CatalogBuilder class."""

    @pytest.fixture
    def mock_org_structure(self, tmp_path):
        """Create mock organization structure."""
        orgs_dir = tmp_path / "organizations"
        orgs_dir.mkdir()

        # Create org1 with 2 repos
        org1 = orgs_dir / "test-org-1"
        org1.mkdir()

        repo1 = org1 / "lib-utils"
        repo1.mkdir()
        (repo1 / ".meta").mkdir()
        (repo1 / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "library",
            "language": "python",
            "tier": 2,
            "status": "active"
        }))
        (repo1 / "README.md").write_text("# lib-utils")

        repo2 = org1 / "tool-cli"
        repo2.mkdir()
        (repo2 / ".meta").mkdir()
        (repo2 / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "tool",
            "language": "typescript",
            "tier": 2,
            "status": "active"
        }))
        (repo2 / "README.md").write_text("# tool-cli")
        (repo2 / ".github").mkdir()
        (repo2 / ".github" / "workflows").mkdir()

        # Create org2 with 1 repo
        org2 = orgs_dir / "test-org-2"
        org2.mkdir()

        repo3 = org2 / "demo-app"
        repo3.mkdir()
        (repo3 / "README.md").write_text("# demo-app")
        # No .meta/repo.yaml - tests inference

        return tmp_path

    def test_scan_organizations(self, mock_org_structure):
        """Should scan all organizations and repos."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        catalog = builder.scan_organizations()

        assert catalog["summary"]["total_organizations"] == 2
        assert catalog["summary"]["total_repositories"] == 3

    def test_extracts_metadata(self, mock_org_structure):
        """Should extract metadata from .meta/repo.yaml."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        catalog = builder.scan_organizations()

        # Find lib-utils in catalog
        lib_utils = None
        for org in catalog["organizations"]:
            for repo in org["repos"]:
                if repo["name"] == "lib-utils":
                    lib_utils = repo
                    break

        assert lib_utils is not None
        assert lib_utils["type"] == "library"
        assert lib_utils["language"] == "python"
        assert lib_utils["tier"] == 2
        assert lib_utils["has_metadata"] is True

    def test_infers_type_from_prefix(self, mock_org_structure):
        """Should infer repo type from name prefix."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        catalog = builder.scan_organizations()

        # Find demo-app (no metadata)
        demo_app = None
        for org in catalog["organizations"]:
            for repo in org["repos"]:
                if repo["name"] == "demo-app":
                    demo_app = repo
                    break

        assert demo_app is not None
        assert demo_app["type"] == "demo"
        assert demo_app["has_metadata"] is False

    def test_tracks_compliance(self, mock_org_structure):
        """Should track compliance indicators."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        catalog = builder.scan_organizations()

        # Find tool-cli (has workflows)
        tool_cli = None
        for org in catalog["organizations"]:
            for repo in org["repos"]:
                if repo["name"] == "tool-cli":
                    tool_cli = repo
                    break

        assert tool_cli is not None
        assert tool_cli["compliance"]["has_readme"] is True
        assert tool_cli["compliance"]["has_ci"] is True

    def test_summary_by_language(self, mock_org_structure):
        """Should aggregate by language."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        catalog = builder.scan_organizations()

        assert "python" in catalog["summary"]["by_language"]
        assert catalog["summary"]["by_language"]["python"] == 1
        assert "typescript" in catalog["summary"]["by_language"]

    def test_summary_by_tier(self, mock_org_structure):
        """Should aggregate by tier."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        catalog = builder.scan_organizations()

        assert catalog["summary"]["by_tier"][2] == 2  # lib-utils and tool-cli
        assert catalog["summary"]["by_tier"][4] >= 1  # demo-app (no metadata)

    def test_generate_json(self, mock_org_structure, tmp_path):
        """Should generate valid JSON output."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        builder.scan_organizations()

        output_file = tmp_path / "catalog.json"
        result = builder.generate_json(output_file)

        assert output_file.exists()
        parsed = json.loads(result)
        assert "organizations" in parsed
        assert "summary" in parsed

    def test_generate_markdown(self, mock_org_structure, tmp_path):
        """Should generate Markdown output."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        builder.scan_organizations()

        output_file = tmp_path / "catalog.md"
        result = builder.generate_markdown(output_file)

        assert output_file.exists()
        assert "# Service Catalog" in result
        assert "test-org-1" in result

    def test_generate_html(self, mock_org_structure, tmp_path):
        """Should generate HTML output."""
        builder = CatalogBuilder(
            base_path=mock_org_structure,
            org_path=str(mock_org_structure / "organizations")
        )
        builder.scan_organizations()

        output_file = tmp_path / "catalog.html"
        result = builder.generate_html(output_file)

        assert output_file.exists()
        assert "<html" in result
        assert "Service Catalog" in result


class TestCatalogEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_organizations_dir(self, tmp_path):
        """Should handle empty organizations directory."""
        orgs_dir = tmp_path / "organizations"
        orgs_dir.mkdir()

        builder = CatalogBuilder(
            base_path=tmp_path,
            org_path=str(orgs_dir)
        )
        catalog = builder.scan_organizations()

        assert catalog["summary"]["total_organizations"] == 0
        assert catalog["summary"]["total_repositories"] == 0

    def test_missing_organizations_dir(self, tmp_path):
        """Should raise error for missing directory."""
        builder = CatalogBuilder(base_path=tmp_path)

        with pytest.raises(RuntimeError, match="not found"):
            builder.scan_organizations()

    def test_invalid_yaml_metadata(self, tmp_path):
        """Should handle invalid YAML in repo.yaml."""
        orgs_dir = tmp_path / "organizations"
        orgs_dir.mkdir()
        org = orgs_dir / "test-org"
        org.mkdir()
        repo = org / "bad-repo"
        repo.mkdir()
        (repo / ".meta").mkdir()
        (repo / ".meta" / "repo.yaml").write_text("invalid: yaml: content:")
        (repo / "README.md").write_text("# Bad Repo")

        builder = CatalogBuilder(
            base_path=tmp_path,
            org_path=str(orgs_dir)
        )
        catalog = builder.scan_organizations()

        # Should still catalog the repo, just with inferred values
        assert catalog["summary"]["total_repositories"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
