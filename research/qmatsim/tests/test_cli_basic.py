import subprocess
import sys

import pytest


def test_help_menu():
    result = subprocess.run([sys.executable, "-m", "qmatsim", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "QMatSim CLI" in result.stdout


def test_simcore_metadata_example():
    """Optional hook into SimCore's shared SimulationMetadata type.

    This test is skipped if the `simcore` package is not available in
    the environment, but when it is installed it exercises the shared
    SimulationMetadata dataclass to ensure basic compatibility between
    QMatSim and SimCore.
    """

    simcore = pytest.importorskip("simcore")

    metadata = simcore.SimulationMetadata(run_id="qmatsim-demo")
    assert metadata.run_id == "qmatsim-demo"
