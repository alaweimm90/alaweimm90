"""
Comprehensive test suite for ValidationFramework

Tests cover:
- Field validation (required, type checking, ranges)
- Pattern matching and regex validation
- Custom validators
- Batch validation
- Schema validation
- Data quality checks
- Edge cases and error handling
"""

import re
import pytest
from typing import Dict, Any, List

from validation_framework.main import (
    Validator,
    FieldRule,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationType,
    BatchValidator,
    SchemaValidator,
    DataQualityChecker,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_validator():
    """Create a basic validator with common rules"""
    validator = Validator()
    validator.add_rule(FieldRule(
        name="name",
        required=True,
        data_type=str,
        min_length=2,
        max_length=50
    ))
    validator.add_rule(FieldRule(
        name="age",
        required=True,
        data_type=int,
        min_value=0,
        max_value=150
    ))
    validator.add_rule(FieldRule(
        name="email",
        required=False,
        data_type=str,
        pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    ))
    return validator


@pytest.fixture
def sample_valid_data():
    """Sample valid data for testing"""
    return {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com"
    }


@pytest.fixture
def sample_invalid_data():
    """Sample invalid data for testing"""
    return {
        "name": "J",  # Too short
        "age": 200,  # Out of range
        "email": "invalid-email"  # Invalid pattern
    }


@pytest.fixture
def batch_data():
    """Sample batch data for testing"""
    return [
        {"name": "Alice", "age": 25, "email": "alice@example.com"},
        {"name": "Bob", "age": 30, "email": "bob@example.com"},
        {"name": "C", "age": 35, "email": "charlie@example.com"},  # Invalid name
        {"name": "Diana", "age": 200, "email": "diana@example.com"},  # Invalid age
        {"name": "Eve", "age": 28, "email": "eve-invalid"},  # Invalid email
    ]


@pytest.fixture
def completeness_data():
    """Data for completeness testing"""
    return [
        {"name": "Alice", "age": 25, "email": "alice@example.com", "phone": "123-456"},
        {"name": "Bob", "age": 30, "email": "", "phone": "234-567"},
        {"name": "", "age": 35, "email": "charlie@example.com", "phone": ""},
        {"name": "Diana", "age": None, "email": "diana@example.com", "phone": "456-789"},
    ]


# ============================================================================
# Basic Validation Tests
# ============================================================================


def test_validator_creation():
    """Test validator can be created and rules added"""
    validator = Validator()
    rule = FieldRule(name="test_field", required=True)
    validator.add_rule(rule)
    assert "test_field" in validator.rules
    assert validator.rules["test_field"].required is True


def test_valid_data_passes_validation(basic_validator, sample_valid_data):
    """Test that valid data passes validation"""
    result = basic_validator.validate(sample_valid_data)
    assert result.is_valid is True
    assert result.errors_count == 0
    assert len(result.issues) == 0


def test_required_field_missing():
    """Test validation fails when required field is missing"""
    validator = Validator()
    validator.add_rule(FieldRule(name="username", required=True))

    result = validator.validate({})
    assert result.is_valid is False
    assert result.errors_count == 1
    assert any(issue.field == "username" for issue in result.issues)
    assert any(issue.validation_type == ValidationType.REQUIRED for issue in result.issues)


def test_required_field_empty_string():
    """Test validation fails when required field is empty string"""
    validator = Validator()
    validator.add_rule(FieldRule(name="username", required=True))

    result = validator.validate({"username": ""})
    assert result.is_valid is False
    assert result.errors_count == 1


def test_optional_field_missing():
    """Test validation passes when optional field is missing"""
    validator = Validator()
    validator.add_rule(FieldRule(name="nickname", required=False, data_type=str))

    result = validator.validate({})
    assert result.is_valid is True
    assert result.errors_count == 0


# ============================================================================
# Type Checking Tests
# ============================================================================


def test_integer_type_validation():
    """Test integer type validation"""
    validator = Validator()
    validator.add_rule(FieldRule(name="count", data_type=int))

    # Valid integer
    result = validator.validate({"count": 42})
    assert result.is_valid is True

    # String that can be converted to int
    result = validator.validate({"count": "42"})
    assert result.is_valid is True

    # Invalid integer
    result = validator.validate({"count": "not_a_number"})
    assert result.is_valid is False


def test_float_type_validation():
    """Test float type validation"""
    validator = Validator()
    validator.add_rule(FieldRule(name="price", data_type=float))

    # Valid float
    result = validator.validate({"price": 19.99})
    assert result.is_valid is True

    # String that can be converted to float
    result = validator.validate({"price": "19.99"})
    assert result.is_valid is True

    # Invalid float
    result = validator.validate({"price": "not_a_number"})
    assert result.is_valid is False


def test_string_type_validation():
    """Test string type validation"""
    validator = Validator()
    validator.add_rule(FieldRule(name="description", data_type=str))

    # Valid string
    result = validator.validate({"description": "A description"})
    assert result.is_valid is True

    # Number (not a string)
    result = validator.validate({"description": 123})
    assert result.is_valid is False


# ============================================================================
# Range Validation Tests
# ============================================================================


def test_numeric_min_value():
    """Test minimum value validation for numbers"""
    validator = Validator()
    validator.add_rule(FieldRule(name="age", data_type=int, min_value=18))

    # Valid age
    result = validator.validate({"age": 25})
    assert result.is_valid is True

    # Edge case: exactly minimum
    result = validator.validate({"age": 18})
    assert result.is_valid is True

    # Below minimum
    result = validator.validate({"age": 17})
    assert result.is_valid is False
    assert result.errors_count == 1


def test_numeric_max_value():
    """Test maximum value validation for numbers"""
    validator = Validator()
    validator.add_rule(FieldRule(name="percentage", data_type=float, max_value=100.0))

    # Valid percentage
    result = validator.validate({"percentage": 75.5})
    assert result.is_valid is True

    # Edge case: exactly maximum
    result = validator.validate({"percentage": 100.0})
    assert result.is_valid is True

    # Above maximum
    result = validator.validate({"percentage": 101.0})
    assert result.is_valid is False


def test_numeric_range():
    """Test value must be within range"""
    validator = Validator()
    validator.add_rule(FieldRule(name="score", data_type=int, min_value=0, max_value=100))

    # Valid score
    result = validator.validate({"score": 85})
    assert result.is_valid is True

    # Below range
    result = validator.validate({"score": -10})
    assert result.is_valid is False

    # Above range
    result = validator.validate({"score": 150})
    assert result.is_valid is False


# ============================================================================
# String Length Validation Tests
# ============================================================================


def test_string_min_length():
    """Test minimum string length validation"""
    validator = Validator()
    validator.add_rule(FieldRule(name="username", data_type=str, min_length=3))

    # Valid length
    result = validator.validate({"username": "john"})
    assert result.is_valid is True

    # Edge case: exactly minimum
    result = validator.validate({"username": "abc"})
    assert result.is_valid is True

    # Too short
    result = validator.validate({"username": "ab"})
    assert result.is_valid is False


def test_string_max_length():
    """Test maximum string length validation"""
    validator = Validator()
    validator.add_rule(FieldRule(name="bio", data_type=str, max_length=100))

    # Valid length
    result = validator.validate({"bio": "A" * 50})
    assert result.is_valid is True

    # Edge case: exactly maximum
    result = validator.validate({"bio": "A" * 100})
    assert result.is_valid is True

    # Too long
    result = validator.validate({"bio": "A" * 101})
    assert result.is_valid is False


def test_string_length_range():
    """Test string length must be within range"""
    validator = Validator()
    validator.add_rule(FieldRule(name="password", data_type=str, min_length=8, max_length=20))

    # Valid length
    result = validator.validate({"password": "mypassword123"})
    assert result.is_valid is True

    # Too short
    result = validator.validate({"password": "pass"})
    assert result.is_valid is False

    # Too long
    result = validator.validate({"password": "a" * 25})
    assert result.is_valid is False


# ============================================================================
# Pattern Validation Tests
# ============================================================================


def test_email_pattern_validation():
    """Test email pattern validation"""
    validator = Validator()
    validator.add_rule(FieldRule(
        name="email",
        data_type=str,
        pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    ))

    # Valid email
    result = validator.validate({"email": "user@example.com"})
    assert result.is_valid is True

    # Invalid email - no @
    result = validator.validate({"email": "userexample.com"})
    assert result.is_valid is False

    # Invalid email - no domain
    result = validator.validate({"email": "user@"})
    assert result.is_valid is False


def test_phone_pattern_validation():
    """Test phone number pattern validation"""
    validator = Validator()
    validator.add_rule(FieldRule(
        name="phone",
        data_type=str,
        pattern=r'^\d{3}-\d{3}-\d{4}$'
    ))

    # Valid phone
    result = validator.validate({"phone": "123-456-7890"})
    assert result.is_valid is True

    # Invalid format
    result = validator.validate({"phone": "1234567890"})
    assert result.is_valid is False


# ============================================================================
# Allowed Values Tests
# ============================================================================


def test_allowed_values_validation():
    """Test allowed values constraint"""
    validator = Validator()
    validator.add_rule(FieldRule(
        name="status",
        allowed_values={"active", "inactive", "pending"}
    ))

    # Valid value
    result = validator.validate({"status": "active"})
    assert result.is_valid is True

    # Invalid value
    result = validator.validate({"status": "deleted"})
    assert result.is_valid is False


# ============================================================================
# Custom Validator Tests
# ============================================================================


def test_custom_validator_function():
    """Test custom validator function"""
    def is_even(value):
        return int(value) % 2 == 0

    validator = Validator()
    validator.add_rule(FieldRule(
        name="even_number",
        custom_validator=is_even,
        custom_error_message="Value must be even"
    ))

    # Valid even number
    result = validator.validate({"even_number": 42})
    assert result.is_valid is True

    # Invalid odd number
    result = validator.validate({"even_number": 43})
    assert result.is_valid is False
    assert any("even" in issue.message.lower() for issue in result.issues)


def test_custom_validator_exception_handling():
    """Test custom validator handles exceptions"""
    def buggy_validator(value):
        raise ValueError("Intentional error")

    validator = Validator()
    validator.add_rule(FieldRule(
        name="test_field",
        custom_validator=buggy_validator
    ))

    result = validator.validate({"test_field": "test"})
    assert result.is_valid is False
    assert any("Custom validator error" in issue.message for issue in result.issues)


# ============================================================================
# ValidationResult Tests
# ============================================================================


def test_validation_result_add_issue():
    """Test adding issues to validation result"""
    result = ValidationResult(is_valid=True)

    # Add error
    error = ValidationIssue(
        field="test",
        message="Error message",
        severity=ValidationSeverity.ERROR,
        validation_type=ValidationType.REQUIRED
    )
    result.add_issue(error)

    assert result.is_valid is False
    assert result.errors_count == 1
    assert len(result.issues) == 1
    assert "test" in result.validated_fields


def test_validation_result_severity_counts():
    """Test counting issues by severity"""
    result = ValidationResult(is_valid=True)

    result.add_issue(ValidationIssue(
        field="field1", message="Error", severity=ValidationSeverity.ERROR,
        validation_type=ValidationType.REQUIRED
    ))
    result.add_issue(ValidationIssue(
        field="field2", message="Warning", severity=ValidationSeverity.WARNING,
        validation_type=ValidationType.CUSTOM
    ))
    result.add_issue(ValidationIssue(
        field="field3", message="Info", severity=ValidationSeverity.INFO,
        validation_type=ValidationType.CUSTOM
    ))

    assert result.errors_count == 1
    assert result.warnings_count == 1
    assert result.info_count == 1


def test_get_issues_by_severity():
    """Test filtering issues by severity"""
    result = ValidationResult(is_valid=True)

    result.add_issue(ValidationIssue(
        field="field1", message="Error", severity=ValidationSeverity.ERROR,
        validation_type=ValidationType.REQUIRED
    ))
    result.add_issue(ValidationIssue(
        field="field2", message="Warning", severity=ValidationSeverity.WARNING,
        validation_type=ValidationType.CUSTOM
    ))

    errors = result.get_issues_by_severity(ValidationSeverity.ERROR)
    warnings = result.get_issues_by_severity(ValidationSeverity.WARNING)

    assert len(errors) == 1
    assert len(warnings) == 1
    assert errors[0].field == "field1"
    assert warnings[0].field == "field2"


def test_get_issues_by_field():
    """Test filtering issues by field"""
    result = ValidationResult(is_valid=True)

    result.add_issue(ValidationIssue(
        field="name", message="Error 1", severity=ValidationSeverity.ERROR,
        validation_type=ValidationType.REQUIRED
    ))
    result.add_issue(ValidationIssue(
        field="name", message="Error 2", severity=ValidationSeverity.ERROR,
        validation_type=ValidationType.PATTERN
    ))
    result.add_issue(ValidationIssue(
        field="age", message="Error 3", severity=ValidationSeverity.ERROR,
        validation_type=ValidationType.RANGE
    ))

    name_issues = result.get_issues_by_field("name")
    age_issues = result.get_issues_by_field("age")

    assert len(name_issues) == 2
    assert len(age_issues) == 1


def test_validation_result_to_dict():
    """Test converting validation result to dictionary"""
    result = ValidationResult(is_valid=False)
    result.add_issue(ValidationIssue(
        field="test", message="Test error", severity=ValidationSeverity.ERROR,
        validation_type=ValidationType.REQUIRED
    ))

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["is_valid"] is False
    assert result_dict["errors_count"] == 1
    assert len(result_dict["issues"]) == 1


# ============================================================================
# Batch Validation Tests
# ============================================================================


def test_batch_validation(basic_validator, batch_data):
    """Test batch validation of multiple records"""
    batch_validator = BatchValidator(basic_validator)
    result = batch_validator.validate_batch(batch_data)

    assert result["total_records"] == 5
    assert result["valid_records"] > 0
    assert result["invalid_records"] > 0
    assert len(result["results"]) == 5


def test_batch_validation_all_valid(basic_validator):
    """Test batch validation with all valid records"""
    valid_records = [
        {"name": "Alice", "age": 25, "email": "alice@example.com"},
        {"name": "Bob", "age": 30, "email": "bob@example.com"},
    ]

    batch_validator = BatchValidator(basic_validator)
    result = batch_validator.validate_batch(valid_records)

    assert result["valid_records"] == 2
    assert result["invalid_records"] == 0
    assert result["total_errors"] == 0


def test_batch_validation_empty_list(basic_validator):
    """Test batch validation with empty list"""
    batch_validator = BatchValidator(basic_validator)
    result = batch_validator.validate_batch([])

    assert result["total_records"] == 0
    assert result["valid_records"] == 0
    assert result["invalid_records"] == 0


# ============================================================================
# Schema Validation Tests
# ============================================================================


def test_schema_validation_required_fields():
    """Test schema validation for required fields"""
    schema = {
        "name": {"required": True},
        "email": {"required": True},
        "age": {"required": False},
    }

    validator = SchemaValidator(schema)

    # Missing required field
    result = validator.validate_schema({"name": "John"})
    assert result.is_valid is False
    assert result.errors_count == 1

    # All required fields present
    result = validator.validate_schema({"name": "John", "email": "john@example.com"})
    assert result.is_valid is True


def test_schema_validation_unknown_fields():
    """Test schema validation warns about unknown fields"""
    schema = {
        "name": {"required": True},
        "email": {"required": True},
    }

    validator = SchemaValidator(schema)
    result = validator.validate_schema({
        "name": "John",
        "email": "john@example.com",
        "unknown_field": "value"
    })

    # Should have a warning for unknown field
    assert result.warnings_count == 1


# ============================================================================
# Data Quality Tests
# ============================================================================


def test_completeness_check(completeness_data):
    """Test data completeness checking"""
    checker = DataQualityChecker()
    result = checker.check_completeness(completeness_data, ["name", "age", "email", "phone"])

    assert result["total_records"] == 4
    assert "field_completeness" in result
    assert "name" in result["field_completeness"]
    assert result["field_completeness"]["name"]["completeness_percentage"] == 75.0


def test_completeness_empty_dataset():
    """Test completeness check with empty dataset"""
    checker = DataQualityChecker()
    result = checker.check_completeness([], ["field1", "field2"])

    assert result["completeness_score"] == 0.0
    assert result["total_records"] == 0


def test_uniqueness_check():
    """Test uniqueness checking"""
    data = [
        {"id": 1, "email": "alice@example.com"},
        {"id": 2, "email": "bob@example.com"},
        {"id": 3, "email": "alice@example.com"},  # duplicate email
        {"id": 1, "email": "charlie@example.com"},  # duplicate id
    ]

    checker = DataQualityChecker()
    result = checker.check_uniqueness(data, ["id", "email"])

    assert result["id"]["total_values"] == 4
    assert result["id"]["unique_values"] == 3
    assert result["id"]["duplicate_count"] == 1

    assert result["email"]["total_values"] == 4
    assert result["email"]["unique_values"] == 3
    assert result["email"]["duplicate_count"] == 1


def test_uniqueness_all_unique():
    """Test uniqueness check when all values are unique"""
    data = [
        {"id": 1, "email": "alice@example.com"},
        {"id": 2, "email": "bob@example.com"},
        {"id": 3, "email": "charlie@example.com"},
    ]

    checker = DataQualityChecker()
    result = checker.check_uniqueness(data, ["id"])

    assert result["id"]["duplicate_count"] == 0
    assert result["id"]["uniqueness_percentage"] == 100.0


# ============================================================================
# Integration and Edge Case Tests
# ============================================================================


def test_multiple_validation_errors():
    """Test multiple validation errors on single record"""
    validator = Validator()
    validator.add_rule(FieldRule(name="name", required=True, min_length=3))
    validator.add_rule(FieldRule(name="age", required=True, data_type=int, min_value=0))
    validator.add_rule(FieldRule(name="email", required=True, pattern=r'^[^@]+@[^@]+\.[^@]+$'))

    # All fields invalid
    result = validator.validate({
        "name": "AB",  # Too short
        "age": -5,  # Below minimum
        "email": "invalid"  # Bad pattern
    })

    assert result.is_valid is False
    assert result.errors_count >= 3


def test_validation_with_none_values():
    """Test validation handles None values correctly"""
    validator = Validator()
    validator.add_rule(FieldRule(name="optional_field", required=False, data_type=str))

    result = validator.validate({"optional_field": None})
    assert result.is_valid is True


def test_complex_real_world_scenario():
    """Test complex real-world validation scenario"""
    # User registration validator
    validator = Validator()
    validator.add_rule(FieldRule(
        name="username",
        required=True,
        data_type=str,
        min_length=3,
        max_length=20,
        pattern=r'^[a-zA-Z0-9_]+$'
    ))
    validator.add_rule(FieldRule(
        name="email",
        required=True,
        data_type=str,
        pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    ))
    validator.add_rule(FieldRule(
        name="age",
        required=True,
        data_type=int,
        min_value=13,
        max_value=120
    ))
    validator.add_rule(FieldRule(
        name="country",
        required=True,
        allowed_values={"US", "UK", "CA", "AU"}
    ))

    # Valid user
    valid_user = {
        "username": "john_doe123",
        "email": "john@example.com",
        "age": 25,
        "country": "US"
    }
    result = validator.validate(valid_user)
    assert result.is_valid is True

    # Invalid user
    invalid_user = {
        "username": "jo",  # Too short
        "email": "invalid-email",  # Bad format
        "age": 12,  # Too young
        "country": "FR"  # Not in allowed list
    }
    result = validator.validate(invalid_user)
    assert result.is_valid is False
    assert result.errors_count >= 4
