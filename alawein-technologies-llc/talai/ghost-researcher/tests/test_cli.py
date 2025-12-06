"""
Tests for CLI interface (main function)
"""
import pytest
import sys
import subprocess
from pathlib import Path
import json
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestCLIList:
    """Test the list command"""

    def test_list_command(self):
        """Test list command shows scientists"""
        result = subprocess.run(
            ['python3', '-m', 'ghost_researcher.main', 'list'],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={'PYTHONPATH': 'src'}
        )

        assert result.returncode == 0
        assert 'einstein' in result.stdout.lower()
        assert 'feynman' in result.stdout.lower()
        assert 'AVAILABLE SCIENTISTS' in result.stdout


class TestCLIInfo:
    """Test the info command"""

    def test_info_command_valid_scientist(self):
        """Test info command with valid scientist"""
        result = subprocess.run(
            ['python3', '-m', 'ghost_researcher.main', 'info', '--scientist', 'einstein'],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={'PYTHONPATH': 'src'}
        )

        assert result.returncode == 0
        # Name appears in uppercase in output
        assert ('Albert Einstein' in result.stdout or 'ALBERT EINSTEIN' in result.stdout)
        assert '1879-1955' in result.stdout
        assert 'KNOWN FOR' in result.stdout
        assert 'PERSONALITY' in result.stdout
        assert 'FAMOUS QUOTES' in result.stdout

    def test_info_command_invalid_scientist(self):
        """Test info command with invalid scientist"""
        result = subprocess.run(
            ['python3', '-m', 'ghost_researcher.main', 'info', '--scientist', 'invalid'],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={'PYTHONPATH': 'src'}
        )

        assert result.returncode != 0
        assert 'Unknown scientist' in result.stderr or 'Unknown scientist' in result.stdout


class TestCLIConsult:
    """Test the consult command"""

    def test_consult_command_creates_consultation(self):
        """Test consult command creates a consultation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'test_ghost.json'

            result = subprocess.run(
                ['python3', '-m', 'ghost_researcher.main', 'consult',
                 '--scientist', 'curie',
                 '--problem', 'fusion energy',
                 '--domain', 'physics'],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent),
                env={'PYTHONPATH': 'src'}
            )

            assert result.returncode == 0
            assert 'Marie Curie' in result.stdout or 'CURIE' in result.stdout
            assert 'fusion energy' in result.stdout.lower()
            assert 'INITIAL REACTION' in result.stdout
            assert 'KEY INSIGHTS' in result.stdout
            assert 'THEORETICAL FRAMEWORK' in result.stdout

    def test_consult_command_with_all_scientists(self):
        """Test consult works with different scientists"""
        scientists = ['darwin', 'turing', 'newton']

        for scientist in scientists:
            result = subprocess.run(
                ['python3', '-m', 'ghost_researcher.main', 'consult',
                 '--scientist', scientist,
                 '--problem', 'AI evolution',
                 '--domain', 'computer_science'],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent),
                env={'PYTHONPATH': 'src'},
                timeout=10
            )

            assert result.returncode == 0, f"Failed for {scientist}"
            assert len(result.stdout) > 100  # Should have substantial output


class TestCLINoCommand:
    """Test CLI with no command"""

    def test_no_command_shows_help(self):
        """Test running with no command"""
        result = subprocess.run(
            ['python3', '-m', 'ghost_researcher.main', '--help'],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={'PYTHONPATH': 'src'}
        )

        # Should show help or usage information
        output = result.stdout + result.stderr
        assert 'usage' in output.lower() or 'GhostResearcher' in output
