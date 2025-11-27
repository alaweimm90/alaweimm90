"""
Test suite for checkpoint.py - Drift Detection.

Tests cover:
- State snapshot generation
- Drift detection between checkpoints
- Report generation
- Compliance tracking
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / ".metaHub" / "scripts"))

from checkpoint import CheckpointManager


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def mock_repo_structure(self, tmp_path):
        """Create mock repository structure."""
        # Create .metaHub directory
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()
        (metahub / "checkpoints").mkdir()

        # Create organizations
        orgs_dir = tmp_path / "organizations"
        orgs_dir.mkdir()

        org = orgs_dir / "test-org"
        org.mkdir()

        # Repo 1: Compliant
        repo1 = org / "compliant-repo"
        repo1.mkdir()
        (repo1 / ".meta").mkdir()
        (repo1 / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "library",
            "language": "python",
            "tier": 3,
            "status": "active"
        }))
        (repo1 / "README.md").write_text("# Compliant Repo")

        # Repo 2: Non-compliant (missing readme)
        repo2 = org / "non-compliant-repo"
        repo2.mkdir()
        (repo2 / ".meta").mkdir()
        (repo2 / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "tool",
            "language": "python",
            "tier": 2
        }))

        return tmp_path

    def test_generate_current_state(self, mock_repo_structure):
        """Should generate current state snapshot."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        state = mgr.generate_current_state()

        assert "timestamp" in state
        assert "repositories" in state
        assert "summary" in state
        assert state["summary"]["total_repositories"] == 2

    def test_tracks_compliance(self, mock_repo_structure):
        """Should track compliance status."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        state = mgr.generate_current_state()

        # Compliant repo (tier 3 only needs metadata + readme)
        compliant = state["repositories"]["test-org/compliant-repo"]
        assert compliant["compliant"] is True

        # Non-compliant repo (tier 2 needs more)
        non_compliant = state["repositories"]["test-org/non-compliant-repo"]
        assert non_compliant["compliant"] is False

    def test_generates_checksum(self, mock_repo_structure):
        """Should generate state checksum."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        state = mgr.generate_current_state()

        assert "checksum" in state
        assert len(state["checksum"]) == 16  # SHA256 truncated to 16 chars

    def test_save_checkpoint(self, mock_repo_structure):
        """Should save checkpoint to file."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        mgr.generate_current_state()
        checkpoint_file = mgr.save_checkpoint()

        assert checkpoint_file.exists()
        assert "checkpoint-" in checkpoint_file.name

        # Should also create latest symlink
        latest = mock_repo_structure / ".metaHub" / "checkpoints" / "checkpoint-latest.json"
        assert latest.exists()

    def test_load_previous_checkpoint(self, mock_repo_structure):
        """Should load previous checkpoint."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        mgr.generate_current_state()
        mgr.save_checkpoint()

        # Create new manager and load
        mgr2 = CheckpointManager(base_path=mock_repo_structure)
        loaded = mgr2.load_previous_checkpoint()

        assert loaded is True
        assert mgr2.previous_state is not None

    def test_detect_drift_no_changes(self, mock_repo_structure):
        """Should detect no drift when state unchanged."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        mgr.generate_current_state()
        mgr.save_checkpoint()

        # Same state
        mgr.load_previous_checkpoint()
        mgr.generate_current_state()
        drift = mgr.detect_drift()

        assert drift["summary"]["changed_count"] == 0

    def test_detect_new_repos(self, mock_repo_structure):
        """Should detect new repositories."""
        mgr = CheckpointManager(base_path=mock_repo_structure)

        # Initial state with empty previous
        mgr.generate_current_state()
        drift = mgr.detect_drift()

        assert drift["has_drift"] is True
        assert len(drift["changes"]["new_repos"]) == 2

    def test_generate_text_report(self, mock_repo_structure):
        """Should generate text report."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        mgr.generate_current_state()
        mgr.detect_drift()

        report = mgr.generate_report(fmt='text')

        assert "DRIFT DETECTION REPORT" in report
        assert "Total Repositories:" in report

    def test_generate_markdown_report(self, mock_repo_structure):
        """Should generate markdown report."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        mgr.generate_current_state()
        mgr.detect_drift()

        report = mgr.generate_report(fmt='markdown')

        assert "# Drift Detection Report" in report
        assert "| Metric | Value |" in report

    def test_generate_json_report(self, mock_repo_structure):
        """Should generate JSON report."""
        mgr = CheckpointManager(base_path=mock_repo_structure)
        mgr.generate_current_state()
        mgr.detect_drift()

        report = mgr.generate_report(fmt='json')
        parsed = json.loads(report)

        assert "drift" in parsed
        assert "current_state" in parsed


class TestDriftDetection:
    """Tests for drift detection scenarios."""

    @pytest.fixture
    def two_state_setup(self, tmp_path):
        """Create setup for two-state comparison."""
        metahub = tmp_path / ".metaHub"
        metahub.mkdir()
        (metahub / "checkpoints").mkdir()

        orgs_dir = tmp_path / "organizations"
        orgs_dir.mkdir()
        org = orgs_dir / "test-org"
        org.mkdir()

        return tmp_path, org

    def test_detect_deleted_repos(self, two_state_setup):
        """Should detect deleted repositories."""
        tmp_path, org = two_state_setup

        # Create initial repo
        repo = org / "to-delete"
        repo.mkdir()
        (repo / "README.md").write_text("# Will be deleted")

        # Save initial state
        mgr = CheckpointManager(base_path=tmp_path)
        mgr.generate_current_state()
        mgr.save_checkpoint("checkpoint-prev.json")

        # Delete repo
        (repo / "README.md").unlink()
        repo.rmdir()

        # Detect drift
        mgr2 = CheckpointManager(base_path=tmp_path)
        mgr2.load_previous_checkpoint(tmp_path / ".metaHub" / "checkpoints" / "checkpoint-prev.json")
        mgr2.generate_current_state()
        drift = mgr2.detect_drift()

        assert "test-org/to-delete" in drift["changes"]["deleted_repos"]

    def test_detect_compliance_improvement(self, two_state_setup):
        """Should detect compliance improvement."""
        tmp_path, org = two_state_setup

        # Create non-compliant repo (tier 3 needs readme)
        repo = org / "improving-repo"
        repo.mkdir()
        (repo / ".meta").mkdir()
        (repo / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "demo",
            "language": "python",
            "tier": 3
        }))
        # No README initially

        # Save initial state
        mgr = CheckpointManager(base_path=tmp_path)
        mgr.generate_current_state()
        mgr.save_checkpoint("checkpoint-prev.json")

        # Add README (now compliant for tier 3)
        (repo / "README.md").write_text("# Now compliant!")

        # Detect drift
        mgr2 = CheckpointManager(base_path=tmp_path)
        mgr2.load_previous_checkpoint(tmp_path / ".metaHub" / "checkpoints" / "checkpoint-prev.json")
        mgr2.generate_current_state()
        drift = mgr2.detect_drift()

        assert "test-org/improving-repo" in drift["changes"]["compliance_improved"]

    def test_detect_compliance_degradation(self, two_state_setup):
        """Should detect compliance degradation."""
        tmp_path, org = two_state_setup

        # Create compliant repo
        repo = org / "degrading-repo"
        repo.mkdir()
        (repo / ".meta").mkdir()
        (repo / ".meta" / "repo.yaml").write_text(yaml.dump({
            "type": "demo",
            "language": "python",
            "tier": 3
        }))
        (repo / "README.md").write_text("# Compliant")

        # Save initial state
        mgr = CheckpointManager(base_path=tmp_path)
        mgr.generate_current_state()
        mgr.save_checkpoint("checkpoint-prev.json")

        # Remove README (now non-compliant)
        (repo / "README.md").unlink()

        # Detect drift
        mgr2 = CheckpointManager(base_path=tmp_path)
        mgr2.load_previous_checkpoint(tmp_path / ".metaHub" / "checkpoints" / "checkpoint-prev.json")
        mgr2.generate_current_state()
        drift = mgr2.detect_drift()

        assert "test-org/degrading-repo" in drift["changes"]["compliance_degraded"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
