
---
**Notation Standards**: See [NOTATION_STANDARDS.md](NOTATION_STANDARDS.md) for consistent mathematical notation across all Librex.QAP documentation.
---

Source: C:\Users\mesha\Pictures\random\notebooks\Librex.QAP_method_comparison_tutorial_documentation.md
Imported: 2025-11-17T14:08:47.214403

# Notebook Documentation

*Converted from: Librex.QAP_method_comparison_tutorial.ipynb*
*Date: 2025-11-17*

# Librex.QAP Method Comparison Tutorial

**A Comprehensive Guide to Comparing Optimization Methods for the Quadratic Assignment Problem**

This tutorial demonstrates how to use Librex.QAP's extensive method collection (48+ algorithms) for systematic performance comparison, analysis, and research.

## Table of Contents
1. [Setup and Installation](#setup)
2. [Method Discovery and Selection](#discovery)
3. [Performance Benchmarking](#benchmarking)
4. [Visualization and Analysis](#visualization)
5. [Statistical Analysis](#statistics)
6. [Research Applications](#research)
7. [Advanced Usage Patterns](#advanced)

---


<a id="setup"></a>
## 1. Setup and Installation

First, let's install Librex.QAP and import the necessary modules.


**Install Librex.QAP (if not already installed)**
**!pip install Librex.QAP**

**Import core modules**
**Set plotting style**
**Import Librex.QAP**


<a id="discovery"></a>
## 2. Method Discovery and Selection

Librex.QAP contains 48+ optimization methods across 8 categories. Let's explore the method landscape.


**Get all available methods**
`all_methods = get_available_methods()`
**Get source statistics**
`source_stats = get_source_statistics()`
**Explore methods by category**
**Select representative methods for comparison**
`comparison_methods = [`
**# Add quantum/ML methods when available**


<a id="benchmarking"></a>
## 3. Performance Benchmarking

Let's benchmark our selected methods on representative QAP instances.


**Select test instances**
**Initialize results storage**
`benchmark_results = {}`
**Run benchmarks**
**# Load instance**
**# Get method**
`method = get_method(method_name)`
**# Time the execution**
`start_time = time()`
**# Calculate gap to optimal**
`gap = calculate_gap(result['objective'], instance_name)`


<a id="visualization"></a>
## 4. Visualization and Analysis

Now let's visualize the benchmarking results using Librex.QAP's visualization tools.


**Create performance heatmap**
**Prepare data for heatmap**
`gap_data = {}`
**Create and display heatmap**
`fig = plt.figure(figsize=(12, 8))`
**Create convergence plots for best methods**
**Plot convergence for a representative instance**
`test_instance = 'had12'`
**Create method comparison plot**


<a id="statistics"></a>
## 5. Statistical Analysis

Let's perform statistical analysis of the benchmarking results.


**Convert results to DataFrame for analysis**
`results_data = []`
**Statistical significance testing**
**# Best performing method overall**
**# Most consistent method**
**# Fastest method**
**Create statistical plots**
**Gap distribution by method**
`sns.boxplot(data=df, x='method', y='gap_percent', ax=ax1)`
**Time distribution by method**
`sns.boxplot(data=df, x='method', y='solve_time', ax=ax2)`
**Gap vs Time scatter**
**Convergence rates**


<a id="research"></a>
## 6. Research Applications

Librex.QAP supports various research applications. Let's demonstrate a few.


**Research Application 1: Method Portfolio Analysis**
**Analyze method diversity**
`source_analysis = SourceAttributionDashboard()`
**Research Application 2: Complexity Analysis**
**Research Application 3: Scalability Study**
**Test methods on different sized instances**
`size_ranges = [`
**# Test a fast method**
`method = get_method('fft_laplace')`
**Plot scalability results**
`fig, ax = plt.subplots(figsize=(10, 6))`


<a id="advanced"></a>
## 7. Advanced Usage Patterns

Advanced patterns for research and production use.


**Advanced Pattern 1: Custom Method Selection**
**# Get instance info**
**# Method selection logic**
`if n <= 20:  # Small instances`
**Test method selection**
`test_instances = ['had12', 'chr15a', 'tai30a']`
**Advanced Pattern 2: Ensemble Methods**
**# Simple ensemble: best objective**
**Test ensemble**
**Advanced Pattern 3: Automated Benchmarking**
**# This would create a detailed HTML/PDF report**
**# For now, just run the benchmark**

**Run automated benchmark**


## Summary

This tutorial demonstrated the comprehensive capabilities of Librex.QAP for method comparison and analysis:

### Key Features Covered:
1. **Method Discovery**: 48+ methods across 8 categories
2. **Performance Benchmarking**: Systematic evaluation on QAPLIB instances
3. **Visualization**: Heatmaps, convergence plots, and statistical analysis
4. **Research Applications**: Portfolio analysis, complexity studies, scalability testing
5. **Advanced Patterns**: Intelligent selection, ensembles, automated benchmarking

### Research-Ready Features:
- **Comprehensive Method Library**: From classical to cutting-edge algorithms
- **Source Attribution**: Complete provenance tracking
- **Statistical Analysis**: Rigorous performance evaluation
- **Publication-Ready Visualizations**: Academic-quality plots and reports
- **Extensible Framework**: Easy addition of new methods and analyses

### Getting Started:
```bash
pip install Librex.QAP
```

```python
from Librex.QAP.methods import get_available_methods, get_method
from Librex.QAP.visualization.performance_heatmap import PerformanceHeatmap

# Start exploring the 48+ optimization methods!
```

### Further Reading:
- [Librex.QAP Documentation](https://Librex.QAP.readthedocs.io/)
- [Method Taxonomy](METHOD_TAXONOMY.md)
- [Integration Guide](INTEGRATION_GUIDE.md)
- [Research Examples](examples/)

---

**Librex.QAP**: Advancing the state-of-the-art in quadratic assignment problem optimization.

