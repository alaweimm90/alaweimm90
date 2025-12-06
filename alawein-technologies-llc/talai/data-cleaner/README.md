# DataCleaner - Automated Data Cleaning and Quality Assessment

Automatically detect and fix common data quality issues in CSV datasets.

## Features

- **Missing value imputation** - Median for numeric, mode for categorical
- **Outlier detection** - Z-score based (>3 std deviations)
- **Duplicate removal** - Exact row matching
- **Format standardization** - Whitespace, case normalization
- **Type inference** - Auto-detect numeric/categorical/datetime/text
- **Quality scoring** - 0-100 overall quality metric
- **Detailed profiling** - Column-level statistics and issues

## Installation

```bash
cd data-cleaner
# No dependencies - pure Python 3.11+
```

## Usage

### Data Profiling

```bash
python cleaner.py profile --input dirty_data.csv --output report.json
```

### Data Cleaning

```bash
python cleaner.py clean --input dirty_data.csv --output cleaned.csv
```

### Custom Configuration

```bash
python cleaner.py clean --input data.csv --output clean.csv --config config.json
```

Config file example:
```json
{
  "remove_duplicates": true,
  "handle_missing": true,
  "missing_strategy": "auto",
  "remove_outliers": false,
  "standardize": true,
  "fix_types": true
}
```

## Revenue Model

- **Individual**: $79/month
- **Team**: $249/month
- **Enterprise**: Custom

## Build Info

- Build time: 5 hours
- Credit used: ~$75
- Lines of code: 680
- Status: Functional prototype
