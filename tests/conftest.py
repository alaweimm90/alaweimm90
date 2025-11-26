"""
Pytest configuration and shared fixtures for Golden Path tests.
"""
import os
import sys
from pathlib import Path

import pytest

# Add scripts directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / ".metaHub" / "scripts"))


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    # Store original
    original = os.environ.get("GOLDEN_PATH_ROOT")

    yield

    # Restore
    if original is None:
        os.environ.pop("GOLDEN_PATH_ROOT", None)
    else:
        os.environ["GOLDEN_PATH_ROOT"] = original


@pytest.fixture
def mock_central_repo(tmp_path):
    """Create a mock central repo structure for testing."""
    central = tmp_path / "central"
    central.mkdir()

    # Create .metaHub structure
    metahub = central / ".metaHub"
    metahub.mkdir()

    # Create templates
    templates = metahub / "templates"
    templates.mkdir()

    # Pre-commit templates
    precommit = templates / "pre-commit"
    precommit.mkdir()
    (precommit / "python.yaml").write_text("# Python pre-commit config")
    (precommit / "typescript.yaml").write_text("# TypeScript pre-commit config")
    (precommit / "generic.yaml").write_text("# Generic pre-commit config")

    # Docker templates
    docker = templates / "docker"
    docker.mkdir()
    (docker / "python.Dockerfile").write_text("FROM python:3.11")
    (docker / "typescript.Dockerfile").write_text("FROM node:20")

    # README template
    (templates / "README.md.template").write_text("""# {{ REPO_NAME }}

{{ REPO_DESCRIPTION }}

Organization: {{ ORG_NAME }}
License: {{ LICENSE_TYPE }}
""")

    # Create schemas
    schemas = metahub / "schemas"
    schemas.mkdir()
    (schemas / "repo-schema.json").write_text("""{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["type", "language"],
  "properties": {
    "type": {"type": "string"},
    "language": {"type": "string"},
    "tier": {"type": "integer", "minimum": 1, "maximum": 4}
  }
}""")

    return central


@pytest.fixture
def mock_inventory(tmp_path):
    """Create a mock inventory.json for testing."""
    import json

    docs = tmp_path / "docs" / "migration-archive"
    docs.mkdir(parents=True)

    inventory = {
        "organizations": [
            {
                "name": "test-org",
                "repositories": [
                    {
                        "name": "test-repo",
                        "description": "Test repository",
                        "languages": ["python"],
                        "active_status": "active",
                    }
                ]
            }
        ]
    }

    inventory_path = docs / "inventory.json"
    with open(inventory_path, "w") as f:
        json.dump(inventory, f)

    return inventory_path
