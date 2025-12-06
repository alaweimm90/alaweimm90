"""
Berkeley SciComp - Plotting Package
==================================
Professional scientific plotting and visualization toolkit with Berkeley branding.
Provides consistent plotting capabilities across the SciComp framework.
Features:
- Scientific publication-ready plots
- Berkeley color scheme integration
- Interactive plotting capabilities
- 2D and 3D visualization
- Statistical plots
- Time series analysis
- Field visualization
- Animation support
Author: Berkeley SciComp Team
Date: 2024
"""
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
BERKELEY_LIGHT_GOLD = '#FDB515'
BERKELEY_GRAY = '#6C6B6C'
BERKELEY_WHITE = '#FFFFFF'
# Extended palette for multi-series plots
BERKELEY_PALETTE = [
    BERKELEY_BLUE,
    CALIFORNIA_GOLD,
    BERKELEY_LIGHT_BLUE,
    '#C3282C',  # Berkeley Red
    '#7B2142',  # Berkeley Maroon
    '#007C7C',  # Berkeley Teal
    '#A4A4A4',  # Berkeley Medium Gray
    '#46535C'   # Berkeley Dark Gray
]
# Import main modules
from .scientific_plots import *
from .time_series import *
from .statistical_plots import *
from .field_visualization import *
from .interactive_plots import *
from .publication_plots import *
# Version info
__version__ = '1.0.0'
__author__ = 'Berkeley SciComp Team'
def demo():
    """
    Run a comprehensive demonstration of the plotting capabilities.
    This function showcases various plot types, styling options,
    and features of the Berkeley SciComp plotting package.
    """
    print("Berkeley SciComp - Plotting Package Demo")
    print("=" * 50)
    print()
    # Import required modules
    import numpy as np
    import matplotlib.pyplot as plt
    from .scientific_plots import ScientificPlot
    from .time_series import TimeSeriesPlot
    from .statistical_plots import StatisticalPlot
    # Set up data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) * np.exp(-x/5)
    y2 = np.cos(x) * np.exp(-x/5)
    y3 = np.sin(2*x) * np.exp(-x/3)
    print("1. Scientific Publication Plot")
    print("-" * 30)
    # Create scientific plot
    sci_plot = ScientificPlot()
    fig, ax = sci_plot.create_figure(figsize=(10, 6))
    sci_plot.plot(x, y1, label='Damped Sine', color=BERKELEY_BLUE)
    sci_plot.plot(x, y2, label='Damped Cosine', color=CALIFORNIA_GOLD)
    sci_plot.plot(x, y3, label='Double Frequency', color=BERKELEY_LIGHT_BLUE)
    sci_plot.set_labels('Time (s)', 'Amplitude', 'Damped Oscillations')
    sci_plot.add_legend()
    sci_plot.add_grid()
    plt.show()
    print("Scientific plot created successfully!")
    print()
    print("2. Time Series Analysis")
    print("-" * 22)
    # Generate time series data
    t = np.linspace(0, 365, 365)  # One year
    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, 1, len(t))
    ts_data = trend + seasonal + noise
    # Create time series plot
    ts_plot = TimeSeriesPlot()
    fig, axes = ts_plot.create_subplots(2, 1, figsize=(12, 8))
    ts_plot.plot_time_series(t, ts_data, ax=axes[0],
                            title='Synthetic Time Series with Trend and Seasonality')
    # Decomposition
    ts_plot.plot_decomposition(t, ts_data, ax=axes[1])
    plt.tight_layout()
    plt.show()
    print("Time series plot created successfully!")
    print()
    print("3. Statistical Visualization")
    print("-" * 27)
    # Generate statistical data
    np.random.seed(42)
    data1 = np.random.normal(100, 15, 1000)
    data2 = np.random.normal(110, 20, 800)
    data3 = np.random.normal(95, 12, 1200)
    # Create statistical plot
    stat_plot = StatisticalPlot()
    fig, axes = stat_plot.create_subplots(2, 2, figsize=(12, 10))
    # Histogram
    stat_plot.histogram([data1, data2, data3],
                       labels=['Group A', 'Group B', 'Group C'],
                       ax=axes[0, 0], title='Distribution Comparison')
    # Box plot
    stat_plot.boxplot([data1, data2, data3],
                     labels=['Group A', 'Group B', 'Group C'],
                     ax=axes[0, 1], title='Box Plot Comparison')
    # Scatter plot with regression
    x_scatter = np.random.normal(50, 10, 200)
    y_scatter = 2 * x_scatter + np.random.normal(0, 5, 200)
    stat_plot.scatter_with_regression(x_scatter, y_scatter, ax=axes[1, 0],
                                    title='Scatter Plot with Regression')
    # Correlation heatmap
    corr_data = np.random.rand(5, 5)
    corr_data = np.corrcoef(np.random.rand(5, 100))
    stat_plot.correlation_heatmap(corr_data,
                                 labels=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'],
                                 ax=axes[1, 1], title='Correlation Matrix')
    plt.tight_layout()
    plt.show()
    print("Statistical plots created successfully!")
    print()
    print("=" * 50)
    print("Berkeley SciComp Plotting Demo Complete!")
    print()
    print("Features demonstrated:")
    print("• Professional scientific plots with Berkeley branding")
    print("• Time series analysis and decomposition")
    print("• Statistical visualizations (histograms, box plots, etc.)")
    print("• Consistent color schemes and styling")
    print("• Publication-ready formatting")
    print()
    print("Additional features available:")
    print("• 3D field visualization")
    print("• Interactive plots")
    print("• Animation capabilities")
    print("• Custom Berkeley themes")
    print("• Export to various formats")
if __name__ == "__main__":
    demo()