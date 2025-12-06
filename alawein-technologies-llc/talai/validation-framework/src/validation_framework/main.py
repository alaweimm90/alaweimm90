"""
ValidationFramework - A comprehensive data validation framework

Provides validators for data quality, schema compliance, and business rules.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationType(Enum):
    """Types of validation checks"""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    UNIQUENESS = "uniqueness"
    RELATIONSHIP = "relationship"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    field: str
    message: str
    severity: ValidationSeverity
    validation_type: ValidationType
    value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "validation_type": self.validation_type.value,
            "value": str(self.value) if self.value is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0
    info_count: int = 0
    validated_fields: Set[str] = field(default_factory=set)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue and update counts"""
        self.issues.append(issue)
        self.validated_fields.add(issue.field)

        if issue.severity == ValidationSeverity.ERROR:
            self.errors_count += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings_count += 1
        else:
            self.info_count += 1

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_field(self, field: str) -> List[ValidationIssue]:
        """Get issues for a specific field"""
        return [issue for issue in self.issues if issue.field == field]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
            "info_count": self.info_count,
            "validated_fields": list(self.validated_fields),
            "issues": [issue.to_dict() for issue in self.issues],
            "timestamp": self.timestamp,
        }


@dataclass
class FieldRule:
    """Validation rule for a field"""
    name: str
    required: bool = False
    data_type: Optional[type] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[Set[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    custom_error_message: Optional[str] = None


class Validator:
    """Core validator class"""

    def __init__(self):
        self.rules: Dict[str, FieldRule] = {}

    def add_rule(self, rule: FieldRule) -> None:
        """Add a validation rule"""
        self.rules[rule.name] = rule

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against defined rules"""
        result = ValidationResult(is_valid=True)

        for field_name, rule in self.rules.items():
            value = data.get(field_name)

            # Check required fields
            if rule.required and (value is None or value == ""):
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message=f"Field '{field_name}' is required",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.REQUIRED,
                    value=value,
                ))
                continue

            # Skip further validation if field is empty and not required
            if value is None or value == "":
                continue

            # Type checking
            if rule.data_type is not None:
                if not self._check_type(value, rule.data_type):
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message=f"Field '{field_name}' must be of type {rule.data_type.__name__}",
                        severity=ValidationSeverity.ERROR,
                        validation_type=ValidationType.TYPE_CHECK,
                        value=value,
                    ))
                    continue

            # Range validation for numeric types
            if isinstance(value, (int, float)):
                if rule.min_value is not None and value < rule.min_value:
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message=f"Field '{field_name}' must be >= {rule.min_value}",
                        severity=ValidationSeverity.ERROR,
                        validation_type=ValidationType.RANGE,
                        value=value,
                    ))

                if rule.max_value is not None and value > rule.max_value:
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message=f"Field '{field_name}' must be <= {rule.max_value}",
                        severity=ValidationSeverity.ERROR,
                        validation_type=ValidationType.RANGE,
                        value=value,
                    ))

            # Length validation for strings
            if isinstance(value, str):
                if rule.min_length is not None and len(value) < rule.min_length:
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message=f"Field '{field_name}' must have at least {rule.min_length} characters",
                        severity=ValidationSeverity.ERROR,
                        validation_type=ValidationType.RANGE,
                        value=value,
                    ))

                if rule.max_length is not None and len(value) > rule.max_length:
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message=f"Field '{field_name}' must have at most {rule.max_length} characters",
                        severity=ValidationSeverity.ERROR,
                        validation_type=ValidationType.RANGE,
                        value=value,
                    ))

                # Pattern validation
                if rule.pattern is not None:
                    if not re.match(rule.pattern, value):
                        result.add_issue(ValidationIssue(
                            field=field_name,
                            message=f"Field '{field_name}' does not match required pattern",
                            severity=ValidationSeverity.ERROR,
                            validation_type=ValidationType.PATTERN,
                            value=value,
                        ))

            # Allowed values validation
            if rule.allowed_values is not None:
                if value not in rule.allowed_values:
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message=f"Field '{field_name}' must be one of {rule.allowed_values}",
                        severity=ValidationSeverity.ERROR,
                        validation_type=ValidationType.CUSTOM,
                        value=value,
                    ))

            # Custom validator
            if rule.custom_validator is not None:
                try:
                    if not rule.custom_validator(value):
                        message = rule.custom_error_message or f"Field '{field_name}' failed custom validation"
                        result.add_issue(ValidationIssue(
                            field=field_name,
                            message=message,
                            severity=ValidationSeverity.ERROR,
                            validation_type=ValidationType.CUSTOM,
                            value=value,
                        ))
                except Exception as e:
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message=f"Custom validator error: {str(e)}",
                        severity=ValidationSeverity.ERROR,
                        validation_type=ValidationType.CUSTOM,
                        value=value,
                    ))

        return result

    def _check_type(self, value: Any, expected_type: type) -> bool:
        """Check if value matches expected type"""
        if expected_type == int:
            try:
                int(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == float:
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == str:
            return isinstance(value, str)
        elif expected_type == bool:
            return isinstance(value, bool)
        else:
            return isinstance(value, expected_type)


class BatchValidator:
    """Validator for batch data validation"""

    def __init__(self, validator: Validator):
        self.validator = validator

    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate multiple records"""
        results = []
        total_errors = 0
        total_warnings = 0
        valid_count = 0
        invalid_count = 0

        for idx, record in enumerate(records):
            result = self.validator.validate(record)
            results.append({
                "record_index": idx,
                "is_valid": result.is_valid,
                "errors": result.errors_count,
                "warnings": result.warnings_count,
                "issues": [issue.to_dict() for issue in result.issues],
            })

            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            total_errors += result.errors_count
            total_warnings += result.warnings_count

        return {
            "total_records": len(records),
            "valid_records": valid_count,
            "invalid_records": invalid_count,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "results": results,
        }


class SchemaValidator:
    """Validate data against a schema"""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate_schema(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data structure matches schema"""
        result = ValidationResult(is_valid=True)

        # Check for required fields in schema
        for field, spec in self.schema.items():
            if spec.get("required", False) and field not in data:
                result.add_issue(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.REQUIRED,
                ))

        # Check for unknown fields
        for field in data:
            if field not in self.schema:
                result.add_issue(ValidationIssue(
                    field=field,
                    message=f"Unknown field '{field}' not in schema",
                    severity=ValidationSeverity.WARNING,
                    validation_type=ValidationType.CUSTOM,
                    value=data[field],
                ))

        return result


class DataQualityChecker:
    """Check data quality metrics"""

    def check_completeness(self, records: List[Dict[str, Any]], required_fields: List[str]) -> Dict[str, Any]:
        """Check data completeness"""
        if not records:
            return {
                "completeness_score": 0.0,
                "total_records": 0,
                "field_completeness": {},
            }

        field_completeness = {}
        for field in required_fields:
            non_empty = sum(1 for record in records if record.get(field) not in [None, "", []])
            completeness = (non_empty / len(records)) * 100
            field_completeness[field] = {
                "completeness_percentage": completeness,
                "non_empty_count": non_empty,
                "empty_count": len(records) - non_empty,
            }

        overall_completeness = sum(
            field_completeness[f]["completeness_percentage"] for f in required_fields
        ) / len(required_fields)

        return {
            "completeness_score": overall_completeness,
            "total_records": len(records),
            "field_completeness": field_completeness,
        }

    def check_uniqueness(self, records: List[Dict[str, Any]], fields: List[str]) -> Dict[str, Any]:
        """Check uniqueness of field values"""
        uniqueness_stats = {}

        for field in fields:
            values = [record.get(field) for record in records if record.get(field) is not None]
            unique_values = set(values)
            duplicate_count = len(values) - len(unique_values)

            uniqueness_stats[field] = {
                "total_values": len(values),
                "unique_values": len(unique_values),
                "duplicate_count": duplicate_count,
                "uniqueness_percentage": (len(unique_values) / len(values) * 100) if values else 0,
            }

        return uniqueness_stats


def main():
    """Main entry point for CLI usage"""
    print("ValidationFramework v0.1.0")
    print("Part of IDEAS research tools suite")
    print("\nUse as a library:")
    print("  from validation_framework.main import Validator, FieldRule")
    print("\nExample:")
    print("  validator = Validator()")
    print("  validator.add_rule(FieldRule(name='email', required=True, pattern=r'^[^@]+@[^@]+\\.[^@]+$'))")
    print("  result = validator.validate({'email': 'test@example.com'})")


if __name__ == "__main__":
    main()
