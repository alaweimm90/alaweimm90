# advanced_analytics
**Module:** `Python/utils/advanced_analytics.py`
## Overview
Advanced analytics and machine learning utilities for SciComp.
This module provides sophisticated analytics including:
- Automated machine learning pipelines
- Statistical analysis and hypothesis testing
- Time series analysis and forecasting
- Dimensionality reduction techniques
- Anomaly detection algorithms
## Functions
### `quick_analysis(data, target)`
Quickly analyze data with automatic method selection.
**Source:** [Line 616](Python/utils/advanced_analytics.py#L616)
### `compare_models(X, y, models)`
Compare multiple models on the same dataset.
**Source:** [Line 622](Python/utils/advanced_analytics.py#L622)
## Classes
### `AnalysisType`
Types of analyses available.
**Class Source:** [Line 43](Python/utils/advanced_analytics.py#L43)
### `AnalysisResult`
Container for analysis results.
#### Methods
##### `to_dict(self)`
Convert result to dictionary for serialization.
**Source:** [Line 64](Python/utils/advanced_analytics.py#L64)
##### `save(self, filename)`
Save analysis result to file.
**Source:** [Line 73](Python/utils/advanced_analytics.py#L73)
**Class Source:** [Line 54](Python/utils/advanced_analytics.py#L54)
### `AdvancedAnalytics`
Main advanced analytics class.
#### Methods
##### `__init__(self)`
Initialize advanced analytics engine.
**Source:** [Line 82](Python/utils/advanced_analytics.py#L82)
##### `auto_analyze(self, data, target, analysis_type)`
Automatically determine and perform the best analysis.
Args:
data: Input data
target: Target values (for supervised learning)
analysis_type: Force specific analysis type
Returns:
Analysis result
**Source:** [Line 88](Python/utils/advanced_analytics.py#L88)
##### `_determine_analysis_type(self, data, target)`
Automatically determine the best analysis type.
**Source:** [Line 121](Python/utils/advanced_analytics.py#L121)
##### `classify(self, X, y, test_size)`
Perform classification analysis.
Args:
X: Feature matrix
y: Target labels
test_size: Test set proportion
Returns:
Classification results
**Source:** [Line 140](Python/utils/advanced_analytics.py#L140)
##### `regress(self, X, y, test_size)`
Perform regression analysis.
Args:
X: Feature matrix
y: Target values
test_size: Test set proportion
Returns:
Regression results
**Source:** [Line 215](Python/utils/advanced_analytics.py#L215)
##### `cluster(self, X, n_clusters)`
Perform clustering analysis.
Args:
X: Data to cluster
n_clusters: Number of clusters (auto-determined if None)
Returns:
Clustering results
**Source:** [Line 292](Python/utils/advanced_analytics.py#L292)
##### `_find_optimal_clusters(self, X, max_clusters)`
Find optimal number of clusters using elbow method.
**Source:** [Line 343](Python/utils/advanced_analytics.py#L343)
##### `detect_anomalies(self, X, contamination)`
Detect anomalies in data.
Args:
X: Data to analyze for anomalies
contamination: Expected proportion of anomalies
Returns:
Anomaly detection results
**Source:** [Line 364](Python/utils/advanced_analytics.py#L364)
##### `reduce_dimensions(self, X, n_components, method)`
Perform dimensionality reduction.
Args:
X: High-dimensional data
n_components: Number of components (auto-determined if None)
method: Reduction method ('pca', 'ica')
Returns:
Dimensionality reduction results
**Source:** [Line 410](Python/utils/advanced_analytics.py#L410)
##### `analyze_time_series(self, data, forecast_steps)`
Analyze time series data.
Args:
data: Time series data
forecast_steps: Number of steps to forecast
Returns:
Time series analysis results
**Source:** [Line 472](Python/utils/advanced_analytics.py#L472)
**Class Source:** [Line 79](Python/utils/advanced_analytics.py#L79)
### `StatisticalTesting`
Statistical hypothesis testing utilities.
#### Methods
##### `__init__(self)`
Initialize statistical testing.
**Source:** [Line 538](Python/utils/advanced_analytics.py#L538)
##### `test_normality(self, data)`
Test if data follows normal distribution.
**Source:** [Line 543](Python/utils/advanced_analytics.py#L543)
##### `compare_groups(self, group1, group2)`
Compare two groups statistically.
**Source:** [Line 562](Python/utils/advanced_analytics.py#L562)
##### `correlation_analysis(self, x, y)`
Analyze correlation between two variables.
**Source:** [Line 590](Python/utils/advanced_analytics.py#L590)
**Class Source:** [Line 535](Python/utils/advanced_analytics.py#L535)