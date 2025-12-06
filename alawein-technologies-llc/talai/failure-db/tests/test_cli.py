"""Tests for CLI functionality."""

import pytest
import sys
import subprocess
import tempfile
import json
from pathlib import Path


class TestCLI:
    """Test command-line interface."""

    def test_cli_module_import(self):
        """Test that main module can be imported."""
        from failure_db.main import FailureDB, main
        assert FailureDB is not None
        assert main is not None
        assert callable(main)

    def test_cli_submit_failure(self):
        """Test submitting failure via CLI."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            db_file = f.name
            json.dump({"failures": {}, "markets": {}, "bets": []}, f)

        try:
            result = subprocess.run([
                sys.executable, '-m', 'failure_db.main',
                'submit-failure',
                '--title', 'Test Failure',
                '--domain', 'Testing',
                '--hypothesis', 'Test hypothesis',
                '--approach', 'Test approach',
                '--reason', 'Test reason',
                '--evidence', 'Test evidence',
                '--cost', '1000',
                '--time', '10',
                '--lessons', 'Lesson 1,Lesson 2',
                '--submitter', 'test_user'
            ], capture_output=True, text=True, cwd='/mnt/c/Users/mesha/Documents/TalAI/failure-db',
            env={'PYTHONPATH': '/mnt/c/Users/mesha/Documents/TalAI/failure-db/src'})

            assert result.returncode == 0
            assert 'Failure submitted successfully' in result.stdout
        finally:
            Path(db_file).unlink(missing_ok=True)

    def test_cli_search_failures(self):
        """Test searching failures via CLI."""
        result = subprocess.run([
            sys.executable, '-m', 'failure_db.main',
            'search'
        ], capture_output=True, text=True, cwd='/mnt/c/Users/mesha/Documents/TalAI/failure-db',
        env={'PYTHONPATH': '/mnt/c/Users/mesha/Documents/TalAI/failure-db/src'})

        # Should run without error even if no results
        assert result.returncode == 0

    def test_cli_create_market(self):
        """Test creating market via CLI."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            db_file = f.name
            json.dump({"failures": {}, "markets": {}, "bets": []}, f)

        try:
            result = subprocess.run([
                sys.executable, '-m', 'failure_db.main',
                'create-market',
                '--idea', 'Test Idea',
                '--description', 'Test Description',
                '--domain', 'Testing',
                '--creator', 'test_user',
                '--deadline', '2030-12-31'
            ], capture_output=True, text=True, cwd='/mnt/c/Users/mesha/Documents/TalAI/failure-db',
            env={'PYTHONPATH': '/mnt/c/Users/mesha/Documents/TalAI/failure-db/src'})

            assert result.returncode == 0
            assert 'Prediction market created' in result.stdout
        finally:
            Path(db_file).unlink(missing_ok=True)

    def test_cli_no_command_shows_help(self):
        """Test that no command shows help."""
        result = subprocess.run([
            sys.executable, '-m', 'failure_db.main'
        ], capture_output=True, text=True, cwd='/mnt/c/Users/mesha/Documents/TalAI/failure-db',
        env={'PYTHONPATH': '/mnt/c/Users/mesha/Documents/TalAI/failure-db/src'})

        # Should exit with code 1 and show usage
        assert result.returncode == 1

    def test_cli_markets_command(self):
        """Test viewing markets command."""
        result = subprocess.run([
            sys.executable, '-m', 'failure_db.main',
            'markets'
        ], capture_output=True, text=True, cwd='/mnt/c/Users/mesha/Documents/TalAI/failure-db',
        env={'PYTHONPATH': '/mnt/c/Users/mesha/Documents/TalAI/failure-db/src'})

        # Should run without error
        assert result.returncode == 0

    def test_cli_analytics_command(self):
        """Test analytics command."""
        result = subprocess.run([
            sys.executable, '-m', 'failure_db.main',
            'analytics'
        ], capture_output=True, text=True, cwd='/mnt/c/Users/mesha/Documents/TalAI/failure-db',
        env={'PYTHONPATH': '/mnt/c/Users/mesha/Documents/TalAI/failure-db/src'})

        # Should run without error
        assert result.returncode == 0
        assert 'FAILURE ANALYTICS' in result.stdout

    def test_cli_leaderboard_command(self):
        """Test leaderboard command."""
        result = subprocess.run([
            sys.executable, '-m', 'failure_db.main',
            'leaderboard'
        ], capture_output=True, text=True, cwd='/mnt/c/Users/mesha/Documents/TalAI/failure-db',
        env={'PYTHONPATH': '/mnt/c/Users/mesha/Documents/TalAI/failure-db/src'})

        # Should run without error
        assert result.returncode == 0
        assert 'TOP PREDICTORS' in result.stdout
