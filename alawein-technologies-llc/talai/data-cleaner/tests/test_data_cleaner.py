"""
Comprehensive test suite for DataCleaner

Tests cover:
- Data cleaning operations
- Validation logic
- Transformation functions
- Edge cases
- Integration workflows
"""

import csv
import json
import pytest
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Dict

from data_cleaner.main import (
    DataCleaner,
    ColumnProfile,
    DataQualityReport,
    CleaningResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cleaner():
    """Create a DataCleaner instance"""
    return DataCleaner()


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file with clean data"""
    csv_file = tmp_path / "sample.csv"
    data = [
        {"id": "1", "name": "Alice", "age": "25", "score": "85.5"},
        {"id": "2", "name": "Bob", "age": "30", "score": "92.0"},
        {"id": "3", "name": "Charlie", "age": "35", "score": "78.3"},
        {"id": "4", "name": "Diana", "age": "28", "score": "88.7"},
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "age", "score"])
        writer.writeheader()
        writer.writerows(data)

    return str(csv_file)


@pytest.fixture
def messy_csv_file(tmp_path):
    """Create a CSV file with quality issues"""
    csv_file = tmp_path / "messy.csv"
    data = [
        {"id": "1", "name": "  alice  ", "age": "25", "score": "85.5", "status": "active"},
        {"id": "2", "name": "bob", "age": "", "score": "92.0", "status": "active"},
        {"id": "3", "name": "Charlie", "age": "35", "score": "", "status": "inactive"},
        {"id": "1", "name": "  alice  ", "age": "25", "score": "85.5", "status": "active"},  # duplicate
        {"id": "4", "name": "diana", "age": "1000", "score": "88.7", "status": "active"},  # outlier
        {"id": "5", "name": "", "age": "30", "score": "75.0", "status": ""},  # missing values
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "age", "score", "status"])
        writer.writeheader()
        writer.writerows(data)

    return str(csv_file)


@pytest.fixture
def duplicate_csv_file(tmp_path):
    """Create a CSV file with duplicate rows"""
    csv_file = tmp_path / "duplicates.csv"
    data = [
        {"id": "1", "name": "Alice", "age": "25"},
        {"id": "2", "name": "Bob", "age": "30"},
        {"id": "1", "name": "Alice", "age": "25"},  # exact duplicate
        {"id": "3", "name": "Charlie", "age": "35"},
        {"id": "2", "name": "Bob", "age": "30"},  # exact duplicate
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "age"])
        writer.writeheader()
        writer.writerows(data)

    return str(csv_file)


@pytest.fixture
def numeric_outliers_csv(tmp_path):
    """Create a CSV file with numeric outliers"""
    csv_file = tmp_path / "outliers.csv"
    # Create tight clustering around 50.0 with clear outliers at 56 and 44
    # This will give mean=50, std~1.85, so z-score for 56/44 will be > 3
    data = []

    # Tight cluster around 50
    for i in range(1, 11):
        data.append({"id": str(i), "value": "50.0"})
    for i in range(11, 16):
        data.append({"id": str(i), "value": "50.1"})
    for i in range(16, 21):
        data.append({"id": str(i), "value": "49.9"})

    # Add outliers
    data.append({"id": "21", "value": "56.0"})  # outlier (z-score > 3)
    data.append({"id": "22", "value": "44.0"})  # outlier (z-score > 3)

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "value"])
        writer.writeheader()
        writer.writerows(data)

    return str(csv_file)


@pytest.fixture
def constant_column_csv(tmp_path):
    """Create a CSV file with a constant column"""
    csv_file = tmp_path / "constant.csv"
    data = [
        {"id": "1", "name": "Alice", "status": "active"},
        {"id": "2", "name": "Bob", "status": "active"},
        {"id": "3", "name": "Charlie", "status": "active"},
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "status"])
        writer.writeheader()
        writer.writerows(data)

    return str(csv_file)


@pytest.fixture
def high_missing_csv(tmp_path):
    """Create a CSV file with high missing values"""
    csv_file = tmp_path / "high_missing.csv"
    data = [
        {"id": "1", "name": "Alice", "optional": ""},
        {"id": "2", "name": "Bob", "optional": ""},
        {"id": "3", "name": "Charlie", "optional": ""},
        {"id": "4", "name": "Diana", "optional": "value"},
        {"id": "5", "name": "Eve", "optional": ""},
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "optional"])
        writer.writeheader()
        writer.writerows(data)

    return str(csv_file)


@pytest.fixture
def datetime_csv(tmp_path):
    """Create a CSV file with datetime values"""
    csv_file = tmp_path / "datetime.csv"
    data = [
        {"id": "1", "date": "2024-01-01", "name": "Event1"},
        {"id": "2", "date": "2024-02-15", "name": "Event2"},
        {"id": "3", "date": "2024-03-20", "name": "Event3"},
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "date", "name"])
        writer.writeheader()
        writer.writerows(data)

    return str(csv_file)


# ============================================================================
# Test DataCleaner Initialization
# ============================================================================


class TestDataCleanerInitialization:
    """Test DataCleaner initialization and configuration"""

    def test_cleaner_initialization(self, cleaner):
        """Test DataCleaner initializes with correct defaults"""
        assert cleaner.outlier_threshold == 3.0
        assert cleaner.max_missing_pct == 50.0
        assert cleaner.fuzzy_match_threshold == 0.9

    def test_cleaner_threshold_customization(self):
        """Test customizing cleaner thresholds"""
        cleaner = DataCleaner()
        cleaner.outlier_threshold = 2.5
        cleaner.max_missing_pct = 30.0

        assert cleaner.outlier_threshold == 2.5
        assert cleaner.max_missing_pct == 30.0


# ============================================================================
# Test CSV Loading and Saving
# ============================================================================


class TestCSVOperations:
    """Test CSV loading and saving operations"""

    def test_load_csv_valid_file(self, cleaner, sample_csv_file):
        """Test loading a valid CSV file"""
        data = cleaner._load_csv(sample_csv_file)

        assert len(data) == 4
        assert data[0]["name"] == "Alice"
        assert data[0]["age"] == "25"

    def test_load_csv_preserves_all_columns(self, cleaner, sample_csv_file):
        """Test that all columns are preserved when loading"""
        data = cleaner._load_csv(sample_csv_file)

        assert "id" in data[0]
        assert "name" in data[0]
        assert "age" in data[0]
        assert "score" in data[0]

    def test_save_csv_creates_file(self, cleaner, tmp_path):
        """Test saving data to CSV creates a file"""
        output_file = tmp_path / "output.csv"
        data = [
            {"id": "1", "name": "Test"},
            {"id": "2", "name": "Test2"},
        ]

        cleaner._save_csv(data, str(output_file))

        assert output_file.exists()

    def test_save_csv_preserves_data(self, cleaner, tmp_path):
        """Test saving and reloading preserves data"""
        output_file = tmp_path / "output.csv"
        original_data = [
            {"id": "1", "name": "Alice", "age": "25"},
            {"id": "2", "name": "Bob", "age": "30"},
        ]

        cleaner._save_csv(original_data, str(output_file))
        loaded_data = cleaner._load_csv(str(output_file))

        assert len(loaded_data) == len(original_data)
        assert loaded_data[0]["name"] == original_data[0]["name"]
        assert loaded_data[1]["age"] == original_data[1]["age"]

    def test_save_empty_data(self, cleaner, tmp_path):
        """Test saving empty data doesn't create file"""
        output_file = tmp_path / "empty.csv"
        cleaner._save_csv([], str(output_file))

        # Should not crash, empty list just returns
        assert not output_file.exists()


# ============================================================================
# Test Data Type Inference
# ============================================================================


class TestDataTypeInference:
    """Test data type detection and inference"""

    def test_infer_numeric_type(self, cleaner):
        """Test inferring numeric data type"""
        values = ["10", "20.5", "30", "40.7", "50"]
        data_type = cleaner._infer_data_type(values)

        assert data_type == "numeric"

    def test_infer_categorical_type(self, cleaner):
        """Test inferring categorical data type"""
        values = ["red", "blue", "red", "green", "blue", "red"] * 10
        data_type = cleaner._infer_data_type(values)

        assert data_type == "categorical"

    def test_infer_text_type(self, cleaner):
        """Test inferring text data type (high cardinality)"""
        values = [f"unique_text_{i}" for i in range(100)]
        data_type = cleaner._infer_data_type(values)

        assert data_type == "text"

    def test_infer_datetime_type(self, cleaner):
        """Test inferring datetime data type"""
        values = ["2024-01-01", "2024-02-15", "2024-03-20"] * 30
        data_type = cleaner._infer_data_type(values)

        assert data_type == "datetime"

    def test_infer_empty_values(self, cleaner):
        """Test inferring type from empty values"""
        values = []
        data_type = cleaner._infer_data_type(values)

        assert data_type == "unknown"

    def test_infer_mixed_numeric_categorical(self, cleaner):
        """Test type inference with mixed data (mostly non-numeric)"""
        values = ["10", "text", "20", "other", "text", "more"] * 10
        data_type = cleaner._infer_data_type(values)

        assert data_type in ["categorical", "text"]


# ============================================================================
# Test Column Profiling
# ============================================================================


class TestColumnProfiling:
    """Test individual column profiling"""

    def test_profile_numeric_column(self, cleaner, sample_csv_file):
        """Test profiling a numeric column"""
        data = cleaner._load_csv(sample_csv_file)
        profile = cleaner._profile_column("age", data)

        assert profile.name == "age"
        assert profile.data_type == "numeric"
        assert profile.total_rows == 4
        assert profile.missing_count == 0
        assert profile.mean is not None
        assert profile.median is not None

    def test_profile_categorical_column(self, cleaner, constant_column_csv):
        """Test profiling a categorical column"""
        data = cleaner._load_csv(constant_column_csv)
        profile = cleaner._profile_column("status", data)

        assert profile.name == "status"
        assert profile.data_type == "categorical"
        assert profile.cardinality == 1
        assert len(profile.top_values) > 0

    def test_profile_column_with_missing(self, cleaner, messy_csv_file):
        """Test profiling column with missing values"""
        data = cleaner._load_csv(messy_csv_file)
        profile = cleaner._profile_column("age", data)

        assert profile.missing_count > 0
        assert profile.missing_pct > 0

    def test_profile_column_unique_count(self, cleaner, sample_csv_file):
        """Test unique count calculation"""
        data = cleaner._load_csv(sample_csv_file)
        profile = cleaner._profile_column("id", data)

        assert profile.unique_count == 4

    def test_add_numeric_stats(self, cleaner):
        """Test adding numeric statistics to profile"""
        profile = ColumnProfile(
            name="test",
            total_rows=5,
            missing_count=0,
            missing_pct=0.0,
            unique_count=5,
            data_type="numeric",
            sample_values=["10", "20", "30", "40", "50"]
        )

        values = ["10", "20", "30", "40", "50"]
        updated_profile = cleaner._add_numeric_stats(profile, values)

        assert updated_profile.mean == 30.0
        assert updated_profile.median == 30.0
        assert updated_profile.min_val == 10.0
        assert updated_profile.max_val == 50.0
        assert updated_profile.std_dev is not None

    def test_add_categorical_stats(self, cleaner):
        """Test adding categorical statistics to profile"""
        profile = ColumnProfile(
            name="test",
            total_rows=6,
            missing_count=0,
            missing_pct=0.0,
            unique_count=3,
            data_type="categorical",
            sample_values=[]
        )

        values = ["red", "blue", "red", "green", "blue", "red"]
        updated_profile = cleaner._add_categorical_stats(profile, values)

        assert updated_profile.cardinality == 3
        assert len(updated_profile.top_values) == 3
        assert updated_profile.top_values[0][0] == "red"  # Most common
        assert updated_profile.top_values[0][1] == 3  # Count


# ============================================================================
# Test Issue Identification
# ============================================================================


class TestIssueIdentification:
    """Test identification of data quality issues"""

    def test_identify_high_missing_rate(self, cleaner, high_missing_csv):
        """Test identification of high missing rate"""
        data = cleaner._load_csv(high_missing_csv)
        profile = cleaner._profile_column("optional", data)

        assert any("missing rate" in issue.lower() for issue in profile.issues)

    def test_identify_constant_column(self, cleaner, constant_column_csv):
        """Test identification of constant column"""
        data = cleaner._load_csv(constant_column_csv)
        profile = cleaner._profile_column("status", data)

        assert any("constant" in issue.lower() for issue in profile.issues)

    def test_identify_outliers(self, cleaner, numeric_outliers_csv):
        """Test identification of outliers"""
        data = cleaner._load_csv(numeric_outliers_csv)
        profile = cleaner._profile_column("value", data)

        assert profile.outlier_count > 0
        assert any("outlier" in issue.lower() for issue in profile.issues)

    def test_no_issues_clean_data(self, cleaner, sample_csv_file):
        """Test no issues detected in clean data"""
        data = cleaner._load_csv(sample_csv_file)
        profile = cleaner._profile_column("age", data)

        # Should have minimal or no issues
        assert len(profile.issues) == 0


# ============================================================================
# Test Duplicate Detection and Removal
# ============================================================================


class TestDuplicateHandling:
    """Test duplicate detection and removal"""

    def test_count_duplicates(self, cleaner, duplicate_csv_file):
        """Test counting duplicate rows"""
        data = cleaner._load_csv(duplicate_csv_file)
        duplicate_count = cleaner._count_duplicates(data)

        assert duplicate_count == 2

    def test_count_no_duplicates(self, cleaner, sample_csv_file):
        """Test counting when no duplicates exist"""
        data = cleaner._load_csv(sample_csv_file)
        duplicate_count = cleaner._count_duplicates(data)

        assert duplicate_count == 0

    def test_remove_duplicates(self, cleaner, duplicate_csv_file):
        """Test removing duplicate rows"""
        data = cleaner._load_csv(duplicate_csv_file)
        unique_data = cleaner._remove_duplicates(data)

        assert len(unique_data) == 3
        assert cleaner._count_duplicates(unique_data) == 0

    def test_remove_duplicates_preserves_order(self, cleaner, duplicate_csv_file):
        """Test that removing duplicates preserves first occurrence"""
        data = cleaner._load_csv(duplicate_csv_file)
        unique_data = cleaner._remove_duplicates(data)

        # First occurrence should be kept
        assert unique_data[0]["id"] == "1"
        assert unique_data[1]["id"] == "2"
        assert unique_data[2]["id"] == "3"


# ============================================================================
# Test Missing Value Handling
# ============================================================================


class TestMissingValueHandling:
    """Test missing value imputation and handling"""

    def test_handle_missing_numeric_median(self, cleaner):
        """Test numeric missing value imputation with median"""
        data = [
            {"id": "1", "value": "10"},
            {"id": "2", "value": "20"},
            {"id": "3", "value": ""},
            {"id": "4", "value": "30"},
        ]

        cleaned_data, changes = cleaner._handle_missing_values(data)

        # Should impute with median (20.0)
        assert cleaned_data[2]["value"] == "20.0"
        assert changes["missing_imputed_median"] == 1

    def test_handle_missing_categorical_mode(self, cleaner):
        """Test categorical missing value imputation with mode"""
        data = [
            {"id": "1", "status": "active"},
            {"id": "2", "status": "active"},
            {"id": "3", "status": ""},
            {"id": "4", "status": "inactive"},
        ]

        cleaned_data, changes = cleaner._handle_missing_values(data)

        # Should impute with mode (active)
        assert cleaned_data[2]["status"] == "active"
        assert changes["missing_imputed_mode"] == 1

    def test_handle_no_missing_values(self, cleaner, sample_csv_file):
        """Test handling when no missing values exist"""
        data = cleaner._load_csv(sample_csv_file)
        cleaned_data, changes = cleaner._handle_missing_values(data)

        assert len(changes) == 0
        assert len(cleaned_data) == len(data)

    def test_handle_empty_dataset(self, cleaner):
        """Test handling missing values on empty dataset"""
        data = []
        cleaned_data, changes = cleaner._handle_missing_values(data)

        assert len(cleaned_data) == 0
        assert len(changes) == 0


# ============================================================================
# Test Outlier Detection and Removal
# ============================================================================


class TestOutlierHandling:
    """Test outlier detection and removal"""

    def test_remove_outliers(self, cleaner, numeric_outliers_csv):
        """Test removing outlier rows"""
        data = cleaner._load_csv(numeric_outliers_csv)
        cleaned_data, changes = cleaner._remove_outliers(data)

        # Should remove rows with extreme values
        assert len(cleaned_data) < len(data)
        assert changes["outliers_removed"] > 0

    def test_outliers_not_removed_clean_data(self, cleaner, sample_csv_file):
        """Test no outliers removed from clean data"""
        data = cleaner._load_csv(sample_csv_file)
        cleaned_data, changes = cleaner._remove_outliers(data)

        assert len(cleaned_data) == len(data)
        assert changes.get("outliers_removed", 0) == 0

    def test_outlier_threshold_sensitivity(self, cleaner, numeric_outliers_csv):
        """Test outlier detection with different thresholds"""
        data = cleaner._load_csv(numeric_outliers_csv)

        # Strict threshold
        cleaner.outlier_threshold = 2.0
        strict_data, strict_changes = cleaner._remove_outliers(data)

        # Lenient threshold
        cleaner.outlier_threshold = 5.0
        lenient_data, lenient_changes = cleaner._remove_outliers(data)

        # Strict threshold should remove more rows
        assert len(strict_data) <= len(lenient_data)


# ============================================================================
# Test Format Standardization
# ============================================================================


class TestFormatStandardization:
    """Test text format standardization"""

    def test_standardize_whitespace(self, cleaner):
        """Test trimming whitespace"""
        data = [
            {"id": "1", "name": "  Alice  "},
            {"id": "2", "name": "Bob   "},
        ]

        cleaned_data, changes = cleaner._standardize_formats(data)

        assert cleaned_data[0]["name"] == "Alice"
        assert cleaned_data[1]["name"] == "Bob"
        assert changes["formats_standardized"] >= 2

    def test_standardize_case(self, cleaner):
        """Test standardizing text case"""
        data = [
            {"id": "1", "name": "alice"},
            {"id": "2", "name": "BOB"},
        ]

        cleaned_data, changes = cleaner._standardize_formats(data)

        # Should be Title Case
        assert cleaned_data[0]["name"] == "Alice"
        assert cleaned_data[1]["name"] == "Bob"

    def test_standardize_preserves_long_text(self, cleaner):
        """Test that long text is not title-cased"""
        long_text = "a" * 60
        data = [
            {"id": "1", "description": long_text},
        ]

        cleaned_data, changes = cleaner._standardize_formats(data)

        # Long text should not be title cased
        assert cleaned_data[0]["description"] == long_text


# ============================================================================
# Test Quality Score Calculation
# ============================================================================


class TestQualityScoreCalculation:
    """Test overall quality score calculation"""

    def test_calculate_quality_score_perfect_data(self, cleaner, sample_csv_file):
        """Test quality score for perfect data"""
        data = cleaner._load_csv(sample_csv_file)
        profiles = [cleaner._profile_column(col, data) for col in data[0].keys()]
        duplicates = cleaner._count_duplicates(data)

        score = cleaner._calculate_quality_score(profiles, duplicates, len(data))

        # Perfect data should have high score
        assert score >= 90.0

    def test_calculate_quality_score_with_issues(self, cleaner, messy_csv_file):
        """Test quality score with data quality issues"""
        data = cleaner._load_csv(messy_csv_file)
        profiles = [cleaner._profile_column(col, data) for col in data[0].keys()]
        duplicates = cleaner._count_duplicates(data)

        score = cleaner._calculate_quality_score(profiles, duplicates, len(data))

        # Messy data should have lower score
        assert score < 90.0

    def test_quality_score_range(self, cleaner, messy_csv_file):
        """Test quality score is within valid range"""
        data = cleaner._load_csv(messy_csv_file)
        profiles = [cleaner._profile_column(col, data) for col in data[0].keys()]
        duplicates = cleaner._count_duplicates(data)

        score = cleaner._calculate_quality_score(profiles, duplicates, len(data))

        assert 0.0 <= score <= 100.0


# ============================================================================
# Test Profile Data (Integration)
# ============================================================================


class TestProfileData:
    """Test complete data profiling workflow"""

    def test_profile_data_basic(self, cleaner, sample_csv_file):
        """Test basic data profiling"""
        report = cleaner.profile_data(sample_csv_file)

        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 4
        assert report.total_columns == 4
        assert len(report.column_profiles) == 4

    def test_profile_data_empty_raises_error(self, cleaner, tmp_path):
        """Test profiling empty dataset raises error"""
        empty_file = tmp_path / "empty.csv"
        with open(empty_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name"])
            writer.writeheader()

        with pytest.raises(ValueError, match="Empty dataset"):
            cleaner.profile_data(str(empty_file))

    def test_profile_data_detects_duplicates(self, cleaner, duplicate_csv_file):
        """Test profiling detects duplicates"""
        report = cleaner.profile_data(duplicate_csv_file)

        assert report.duplicate_rows == 2

    def test_profile_data_generates_recommendations(self, cleaner, messy_csv_file):
        """Test profiling generates recommendations"""
        report = cleaner.profile_data(messy_csv_file)

        assert len(report.recommendations) > 0

    def test_profile_data_identifies_critical_issues(self, cleaner, messy_csv_file):
        """Test profiling identifies critical issues"""
        report = cleaner.profile_data(messy_csv_file)

        # Should identify duplicates
        assert any("duplicate" in issue.lower() for issue in report.critical_issues)

    def test_profile_data_timestamp(self, cleaner, sample_csv_file):
        """Test report includes timestamp"""
        report = cleaner.profile_data(sample_csv_file)

        assert report.timestamp is not None
        assert len(report.timestamp) > 0


# ============================================================================
# Test Clean Data (Integration)
# ============================================================================


class TestCleanData:
    """Test complete data cleaning workflow"""

    def test_clean_data_basic(self, cleaner, messy_csv_file, tmp_path):
        """Test basic data cleaning"""
        output_file = tmp_path / "cleaned.csv"
        result = cleaner.clean_data(messy_csv_file, str(output_file))

        assert isinstance(result, CleaningResult)
        assert result.original_rows > 0
        assert result.cleaned_rows > 0
        assert output_file.exists()

    def test_clean_data_removes_duplicates_by_default(self, cleaner, duplicate_csv_file, tmp_path):
        """Test cleaning removes duplicates by default"""
        output_file = tmp_path / "cleaned.csv"
        result = cleaner.clean_data(duplicate_csv_file, str(output_file))

        assert result.original_rows == 5
        assert result.cleaned_rows == 3
        assert "duplicates_removed" in result.changes_made

    def test_clean_data_with_config_disable_duplicates(self, cleaner, duplicate_csv_file, tmp_path):
        """Test cleaning with duplicate removal disabled"""
        output_file = tmp_path / "cleaned.csv"
        config = {"remove_duplicates": False}
        result = cleaner.clean_data(duplicate_csv_file, str(output_file), config)

        assert result.original_rows == result.cleaned_rows
        assert "duplicates_removed" not in result.changes_made

    def test_clean_data_handles_missing_values(self, cleaner, messy_csv_file, tmp_path):
        """Test cleaning handles missing values"""
        output_file = tmp_path / "cleaned.csv"
        result = cleaner.clean_data(messy_csv_file, str(output_file))

        # Should have imputed some missing values
        assert any("missing_imputed" in key for key in result.changes_made.keys())

    def test_clean_data_standardizes_formats(self, cleaner, messy_csv_file, tmp_path):
        """Test cleaning standardizes formats"""
        output_file = tmp_path / "cleaned.csv"
        result = cleaner.clean_data(messy_csv_file, str(output_file))

        # Should have standardized formats
        assert "formats_standardized" in result.changes_made or len(result.changes_made) > 0

    def test_clean_data_with_outlier_removal(self, cleaner, numeric_outliers_csv, tmp_path):
        """Test cleaning with outlier removal enabled"""
        output_file = tmp_path / "cleaned.csv"
        config = {"remove_outliers": True}
        result = cleaner.clean_data(numeric_outliers_csv, str(output_file), config)

        assert result.rows_removed > 0
        assert "outliers_removed" in result.changes_made

    def test_clean_data_result_timestamp(self, cleaner, sample_csv_file, tmp_path):
        """Test cleaning result includes timestamp"""
        output_file = tmp_path / "cleaned.csv"
        result = cleaner.clean_data(sample_csv_file, str(output_file))

        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_clean_data_quality_improvement(self, cleaner, sample_csv_file, tmp_path):
        """Test cleaning result includes quality improvement"""
        output_file = tmp_path / "cleaned.csv"
        result = cleaner.clean_data(sample_csv_file, str(output_file))

        assert result.quality_improvement >= 0


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_row_dataset(self, cleaner, tmp_path):
        """Test handling dataset with single row"""
        csv_file = tmp_path / "single.csv"
        data = [{"id": "1", "name": "Alice"}]

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name"])
            writer.writeheader()
            writer.writerows(data)

        report = cleaner.profile_data(str(csv_file))

        assert report.total_rows == 1
        assert report.duplicate_rows == 0

    def test_single_column_dataset(self, cleaner, tmp_path):
        """Test handling dataset with single column"""
        csv_file = tmp_path / "single_col.csv"
        data = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id"])
            writer.writeheader()
            writer.writerows(data)

        report = cleaner.profile_data(str(csv_file))

        assert report.total_columns == 1
        assert len(report.column_profiles) == 1

    def test_all_missing_column(self, cleaner, tmp_path):
        """Test handling column with all missing values"""
        csv_file = tmp_path / "all_missing.csv"
        data = [
            {"id": "1", "empty": ""},
            {"id": "2", "empty": ""},
            {"id": "3", "empty": ""},
        ]

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "empty"])
            writer.writeheader()
            writer.writerows(data)

        report = cleaner.profile_data(str(csv_file))

        # Find the empty column profile
        empty_profile = next(p for p in report.column_profiles if p.name == "empty")
        assert empty_profile.missing_pct == 100.0

    def test_numeric_with_single_value(self, cleaner):
        """Test numeric stats with single value"""
        profile = ColumnProfile(
            name="test",
            total_rows=1,
            missing_count=0,
            missing_pct=0.0,
            unique_count=1,
            data_type="numeric",
            sample_values=["10"]
        )

        values = ["10"]
        updated_profile = cleaner._add_numeric_stats(profile, values)

        # Should handle single value gracefully
        assert updated_profile.mean == 10.0
        assert updated_profile.std_dev == 0

    def test_recommendation_generation_no_issues(self, cleaner, sample_csv_file):
        """Test recommendation generation with clean data"""
        data = cleaner._load_csv(sample_csv_file)
        profiles = [cleaner._profile_column(col, data) for col in data[0].keys()]

        critical_issues, recommendations = cleaner._generate_recommendations(profiles, 0)

        # Clean data should have minimal issues
        assert len(critical_issues) == 0
