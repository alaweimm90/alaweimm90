"""
Tests for TemplateManager class and related functionality.
"""

import pytest
import json
from PEDs_Playbook.main import TemplateManager, SupplementPlan


class TestTemplateManagerCreation:
    """Test template manager creation."""

    def test_create_empty_manager(self, template_manager):
        """Test creating an empty template manager."""
        assert len(template_manager.list_templates()) == 0

    def test_create_manager_with_templates(self, template_manager_with_templates):
        """Test creating template manager with templates."""
        assert len(template_manager_with_templates.list_templates()) == 1


class TestTemplateOperations:
    """Test template CRUD operations."""

    def test_create_template(self, template_manager, populated_supplement_plan):
        """Test creating a new template."""
        template_manager.create_template("Test Template", populated_supplement_plan)
        assert "Test Template" in template_manager.list_templates()

    def test_create_template_with_empty_name_raises_error(self, template_manager,
                                                          populated_supplement_plan):
        """Test that empty template name raises ValueError."""
        with pytest.raises(ValueError, match="Template name cannot be empty"):
            template_manager.create_template("", populated_supplement_plan)

    def test_create_template_with_whitespace_name_raises_error(self, template_manager,
                                                               populated_supplement_plan):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="Template name cannot be empty"):
            template_manager.create_template("   ", populated_supplement_plan)

    def test_create_template_with_invalid_type_raises_error(self, template_manager):
        """Test that invalid plan type raises TypeError."""
        with pytest.raises(TypeError, match="Must be a SupplementPlan instance"):
            template_manager.create_template("Test", "not a plan")

    def test_get_template(self, template_manager_with_templates):
        """Test retrieving a template by name."""
        template = template_manager_with_templates.get_template("Muscle Gain Basic")
        assert template is not None
        assert isinstance(template, SupplementPlan)

    def test_get_nonexistent_template_returns_none(self, template_manager):
        """Test that getting non-existent template returns None."""
        template = template_manager.get_template("Nonexistent")
        assert template is None

    def test_list_templates(self, template_manager_with_templates):
        """Test listing all template names."""
        templates = template_manager_with_templates.list_templates()
        assert isinstance(templates, list)
        assert "Muscle Gain Basic" in templates

    def test_delete_template(self, template_manager_with_templates):
        """Test deleting a template."""
        result = template_manager_with_templates.delete_template("Muscle Gain Basic")
        assert result is True
        assert "Muscle Gain Basic" not in template_manager_with_templates.list_templates()

    def test_delete_nonexistent_template_returns_false(self, template_manager):
        """Test that deleting non-existent template returns False."""
        result = template_manager.delete_template("Nonexistent")
        assert result is False


class TestTemplateImportExport:
    """Test template import/export functionality."""

    def test_export_template(self, template_manager_with_templates):
        """Test exporting a template to JSON."""
        json_data = template_manager_with_templates.export_template("Muscle Gain Basic")
        assert json_data is not None
        assert isinstance(json_data, str)
        # Verify it's valid JSON
        data = json.loads(json_data)
        assert 'plan_id' in data
        assert 'supplements' in data

    def test_export_nonexistent_template_returns_none(self, template_manager):
        """Test that exporting non-existent template returns None."""
        json_data = template_manager.export_template("Nonexistent")
        assert json_data is None

    def test_import_template(self, template_manager, populated_supplement_plan):
        """Test importing a template from JSON."""
        # First export a template
        json_data = json.dumps(populated_supplement_plan.to_dict())
        # Then import it
        result = template_manager.import_template("Imported Template", json_data)
        assert result is True
        assert "Imported Template" in template_manager.list_templates()

    def test_import_template_creates_valid_plan(self, template_manager,
                                               populated_supplement_plan):
        """Test that imported template creates a valid plan."""
        json_data = json.dumps(populated_supplement_plan.to_dict())
        template_manager.import_template("Valid Template", json_data)
        imported = template_manager.get_template("Valid Template")
        assert len(imported.supplements) == len(populated_supplement_plan.supplements)

    def test_import_invalid_json_returns_false(self, template_manager):
        """Test that importing invalid JSON returns False."""
        result = template_manager.import_template("Bad Template", "not valid json")
        assert result is False

    def test_import_incomplete_data_returns_false(self, template_manager):
        """Test that importing incomplete data returns False."""
        incomplete_json = json.dumps({"plan_id": "test"})  # Missing required fields
        result = template_manager.import_template("Incomplete", incomplete_json)
        assert result is False

    def test_export_import_roundtrip(self, template_manager_with_templates):
        """Test that export/import roundtrip preserves data."""
        # Export
        exported = template_manager_with_templates.export_template("Muscle Gain Basic")
        # Import with new name
        new_manager = TemplateManager()
        new_manager.import_template("Roundtrip Test", exported)
        # Verify
        original = template_manager_with_templates.get_template("Muscle Gain Basic")
        imported = new_manager.get_template("Roundtrip Test")
        assert len(original.supplements) == len(imported.supplements)
        assert original.total_cost == imported.total_cost
