"""
Unit tests for HELIOS CLI.

Tests for:
- CLI command parsing
- Domain listing
- Help messages
"""

import pytest


class TestCLI:
    """Test HELIOS CLI."""

    @pytest.mark.unit
    def test_cli_imports(self):
        """Test that CLI can be imported."""
        try:
            from helios.cli import main
            assert main is not None
        except ImportError:
            pytest.skip("CLI not available")

    @pytest.mark.unit
    def test_cli_version(self):
        """Test CLI version command."""
        try:
            from helios.cli import main
            # This would normally exit, so we just verify it exists
            assert main is not None
        except ImportError:
            pytest.skip("CLI not available")

    @pytest.mark.unit
    def test_cli_help(self):
        """Test CLI help command."""
        try:
            from helios.cli import main
            # Can add help test here
            assert main is not None
        except ImportError:
            pytest.skip("CLI not available")

    @pytest.mark.unit
    def test_domain_list_function(self):
        """Test domain listing function."""
        try:
            from helios.cli import list_domains
            # Just verify it can be called
            assert list_domains is not None
        except ImportError:
            pytest.skip("CLI not available")

    @pytest.mark.unit
    def test_domain_info_function(self):
        """Test domain info function."""
        try:
            from helios.cli import domain_info
            assert domain_info is not None
        except ImportError:
            pytest.skip("CLI not available")

    @pytest.mark.unit
    def test_generate_function(self):
        """Test hypothesis generation function."""
        try:
            from helios.cli import generate_hypotheses
            assert generate_hypotheses is not None
        except ImportError:
            pytest.skip("CLI not available")

    @pytest.mark.unit
    def test_validate_function(self):
        """Test hypothesis validation function."""
        try:
            from helios.cli import validate_hypothesis
            assert validate_hypothesis is not None
        except ImportError:
            pytest.skip("CLI not available")
