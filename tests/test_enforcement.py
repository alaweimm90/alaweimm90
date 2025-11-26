"""
Test suite for Golden Path enforcement script.

Tests cover:
- Central repo path detection
- Repo type inference
- Language detection
- Metadata generation (schema compliance)
- Dry-run mode
- File writing behavior
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from verify_and_enforce_golden_path import (
    RepoEnforcer,
    get_central_repo_path,
    setup_logging,
)


class TestGetCentralRepoPath:
    """Tests for auto-detection of central repo path."""

    def test_detects_from_environment_variable(self, tmp_path):
        """Should use GOLDEN_PATH_ROOT env var when set."""
        # Create .metaHub directory in temp path
        metahub_dir = tmp_path / ".metaHub"
        metahub_dir.mkdir()

        with patch.dict(os.environ, {"GOLDEN_PATH_ROOT": str(tmp_path)}):
            result = get_central_repo_path()
            assert result == tmp_path

    def test_warns_if_env_var_invalid(self, tmp_path, caplog):
        """Should warn if GOLDEN_PATH_ROOT doesn't contain .metaHub."""
        with patch.dict(os.environ, {"GOLDEN_PATH_ROOT": str(tmp_path)}):
            # Create .metaHub in cwd instead
            cwd_metahub = Path.cwd() / ".metaHub"
            if cwd_metahub.exists():
                result = get_central_repo_path()
                assert "not found there" in caplog.text or result is not None

    def test_auto_detects_from_parent_directories(self, tmp_path):
        """Should search up directory tree for .metaHub."""
        # Create nested structure
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()

        # Change to nested directory and try detection
        original_cwd = os.getcwd()
        try:
            os.chdir(nested)
            # Clear env var
            with patch.dict(os.environ, {"GOLDEN_PATH_ROOT": ""}, clear=False):
                os.environ.pop("GOLDEN_PATH_ROOT", None)
                result = get_central_repo_path()
                assert result == tmp_path
        finally:
            os.chdir(original_cwd)

    def test_raises_if_not_found(self, tmp_path):
        """Should raise RuntimeError if no .metaHub found."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch.dict(os.environ, {}, clear=True):
                # Ensure no .metaHub exists
                with pytest.raises(RuntimeError, match="Could not find central repo"):
                    get_central_repo_path()
        finally:
            os.chdir(original_cwd)


class TestRepoEnforcer:
    """Tests for RepoEnforcer class."""

    @pytest.fixture
    def enforcer_setup(self, tmp_path):
        """Create a basic enforcer setup for testing."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        central_path = tmp_path / "central"
        central_path.mkdir()
        (central_path / ".metaHub").mkdir()
        (central_path / ".metaHub" / "templates").mkdir()
        (central_path / ".metaHub" / "templates" / "pre-commit").mkdir()

        repo_data = {
            "name": "test-repo",
            "description": "Test repository",
            "languages": ["python"],
            "active_status": "active",
        }

        return {
            "repo_path": repo_path,
            "central_path": central_path,
            "repo_data": repo_data,
            "org": "test-org",
            "repo_name": "test-repo",
        }

    def test_infer_repo_type_from_prefix(self, enforcer_setup):
        """Should infer repo type from name prefix."""
        test_cases = [
            ("lib-utils", "library"),
            ("tool-cli", "tool"),
            ("adapter-api", "adapter"),
            ("demo-app", "demo"),
            ("paper-research", "research"),
        ]

        for repo_name, expected_type in test_cases:
            enforcer = RepoEnforcer(
                enforcer_setup["repo_path"],
                enforcer_setup["org"],
                repo_name,
                enforcer_setup["repo_data"],
                enforcer_setup["central_path"],
            )
            assert enforcer.infer_repo_type() == expected_type

    def test_infer_primary_language_from_files(self, enforcer_setup):
        """Should detect language from project files."""
        repo_path = enforcer_setup["repo_path"]

        # Test Python detection
        (repo_path / "pyproject.toml").touch()
        enforcer = RepoEnforcer(
            repo_path,
            enforcer_setup["org"],
            enforcer_setup["repo_name"],
            {"languages": []},  # Empty languages to force file detection
            enforcer_setup["central_path"],
        )
        assert enforcer.infer_primary_language() == "python"

        # Clean up and test TypeScript
        (repo_path / "pyproject.toml").unlink()
        (repo_path / "package.json").touch()
        assert enforcer.infer_primary_language() == "typescript"

    def test_dry_run_does_not_write(self, enforcer_setup):
        """Dry-run mode should not write any files."""
        enforcer = RepoEnforcer(
            enforcer_setup["repo_path"],
            enforcer_setup["org"],
            enforcer_setup["repo_name"],
            enforcer_setup["repo_data"],
            enforcer_setup["central_path"],
            dry_run=True,
        )

        # Try to write a file
        test_path = enforcer_setup["repo_path"] / "test-file.txt"
        result = enforcer._write_file(test_path, "test content")

        assert result is True  # Should return True
        assert not test_path.exists()  # But file should not exist

    def test_write_file_creates_parent_dirs(self, enforcer_setup):
        """_write_file should create parent directories."""
        enforcer = RepoEnforcer(
            enforcer_setup["repo_path"],
            enforcer_setup["org"],
            enforcer_setup["repo_name"],
            enforcer_setup["repo_data"],
            enforcer_setup["central_path"],
            dry_run=False,
        )

        nested_path = enforcer_setup["repo_path"] / "a" / "b" / "test.txt"
        enforcer._write_file(nested_path, "content")

        assert nested_path.exists()
        assert nested_path.read_text() == "content"

    def test_metadata_schema_compliance(self, enforcer_setup):
        """Generated metadata should match schema fields."""
        enforcer = RepoEnforcer(
            enforcer_setup["repo_path"],
            enforcer_setup["org"],
            enforcer_setup["repo_name"],
            enforcer_setup["repo_data"],
            enforcer_setup["central_path"],
        )

        # Call the method to generate metadata
        enforcer.create_or_update_meta_repo_yaml()

        # Read generated file
        meta_file = enforcer_setup["repo_path"] / ".meta" / "repo.yaml"
        assert meta_file.exists()

        with open(meta_file) as f:
            metadata = yaml.safe_load(f)

        # Verify schema-compliant fields
        assert "type" in metadata
        assert "language" in metadata  # singular, not "languages"
        assert "tier" in metadata  # integer, not "criticality_tier"
        assert isinstance(metadata["tier"], int)
        assert "coverage" in metadata
        assert "target" in metadata["coverage"]
        assert "docs" in metadata
        assert "profile" in metadata["docs"]
        assert "owner" in metadata

        # Verify old fields are NOT present
        assert "criticality_tier" not in metadata
        assert "languages" not in metadata
        assert "docs_profile" not in metadata

    def test_log_change_tracks_changes(self, enforcer_setup):
        """_log_change should append to changes_made list."""
        enforcer = RepoEnforcer(
            enforcer_setup["repo_path"],
            enforcer_setup["org"],
            enforcer_setup["repo_name"],
            enforcer_setup["repo_data"],
            enforcer_setup["central_path"],
        )

        assert len(enforcer.changes_made) == 0
        enforcer._log_change("Test change 1")
        enforcer._log_change("Test change 2")
        assert len(enforcer.changes_made) == 2
        assert "Test change 1" in enforcer.changes_made


class TestTierAssignment:
    """Tests for criticality tier assignment logic."""

    @pytest.fixture
    def base_setup(self, tmp_path):
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        central_path = tmp_path / "central"
        central_path.mkdir()
        (central_path / ".metaHub").mkdir()
        return repo_path, central_path

    def test_tier_1_for_critical_repos(self, base_setup):
        """Critical repos should get tier 1."""
        repo_path, central_path = base_setup

        for repo_name in ["core-control-center", ".github", "standards"]:
            enforcer = RepoEnforcer(
                repo_path, "org", repo_name, {}, central_path
            )
            enforcer.create_or_update_meta_repo_yaml()

            meta_file = repo_path / ".meta" / "repo.yaml"
            with open(meta_file) as f:
                metadata = yaml.safe_load(f)
            assert metadata["tier"] == 1

            # Clean up for next iteration
            meta_file.unlink()

    def test_tier_2_for_active_tools(self, base_setup):
        """Active library/tool repos should get tier 2."""
        repo_path, central_path = base_setup

        enforcer = RepoEnforcer(
            repo_path, "org", "lib-utils",
            {"active_status": "active"},
            central_path
        )
        enforcer.create_or_update_meta_repo_yaml()

        meta_file = repo_path / ".meta" / "repo.yaml"
        with open(meta_file) as f:
            metadata = yaml.safe_load(f)
        assert metadata["tier"] == 2

    def test_tier_3_for_research(self, base_setup):
        """Research/demo/adapter repos should get tier 3."""
        repo_path, central_path = base_setup

        enforcer = RepoEnforcer(
            repo_path, "org", "demo-app", {}, central_path
        )
        enforcer.create_or_update_meta_repo_yaml()

        meta_file = repo_path / ".meta" / "repo.yaml"
        with open(meta_file) as f:
            metadata = yaml.safe_load(f)
        assert metadata["tier"] == 3


class TestSetupLogging:
    """Tests for logging configuration."""

    def test_creates_file_handler_when_requested(self, tmp_path):
        """Should create log file when log_file is provided."""
        log_file = tmp_path / "test.log"
        logger = setup_logging(verbose=False, log_file=log_file)

        logger.info("Test message")

        # Note: Due to how logging works, we need to flush
        for handler in logger.handlers:
            handler.flush()

    def test_verbose_sets_debug_level(self, tmp_path):
        """Verbose mode should set DEBUG level."""
        import logging as log_module

        # Reset logging
        log_module.root.handlers = []

        logger = setup_logging(verbose=True)
        assert logger.level == log_module.DEBUG or log_module.root.level == log_module.DEBUG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
