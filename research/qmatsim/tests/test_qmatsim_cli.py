# tests/test_qmatsim_cli.py

import subprocess
import sys
import os
from pathlib import Path

def test_cli_help():
    """Test that CLI help runs successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "qmatsim", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result.returncode == 0
    assert "QMatSim CLI" in result.stdout or "usage:" in result.stdout

def test_invalid_command():
    """Test that unknown command triggers help/exit."""
    result = subprocess.run(
        [sys.executable, "-m", "qmatsim", "unknown"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "usage:" in result.stdout
