# berkeley_style
**Module:** `Python/visualization/berkeley_style.py`
## Overview
Berkeley Visual Identity - Scientific Plotting Style
Official UC Berkeley color scheme and styling for publication-quality figures.
Implements university branding guidelines for academic and research publications.
Colors follow UC Berkeley's official brand guidelines:
- Primary: Berkeley Blue (#003262), California Gold (#FDB515)
- Secondary palette for diverse visualizations
- High-contrast accessibility compliance
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
## Constants
- **`BERKELEY_COLORS`**
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_SEQUENCE`**
- **`BERKELEY_STYLE`**
## Functions
### `apply_berkeley_style()`
Apply Berkeley visual identity to matplotlib.
**Source:** [Line 129](Python/visualization/berkeley_style.py#L129)
### `_register_berkeley_colormaps()`
Register Berkeley-themed colormaps.
**Source:** [Line 137](Python/visualization/berkeley_style.py#L137)
### `publication_figure(figsize)`
Create publication-ready Berkeley-styled figure.
**Source:** [Line 538](Python/visualization/berkeley_style.py#L538)
### `presentation_figure(figsize)`
Create presentation-ready Berkeley-styled figure.
**Source:** [Line 542](Python/visualization/berkeley_style.py#L542)
### `poster_figure(figsize)`
Create poster-ready Berkeley-styled figure.
**Source:** [Line 546](Python/visualization/berkeley_style.py#L546)
### `save_publication_figure(fig, filename, dpi, format)`
Save publication-quality figure with proper settings.
Parameters
----------
fig : Figure
Matplotlib figure to save
filename : str or Path
Output filename
dpi : int, default 300
Resolution
format : str, default 'png'
Output format
**Source:** [Line 550](Python/visualization/berkeley_style.py#L550)
## Classes
### `BerkeleyPlot`
Berkeley-styled plotting class for scientific figures.
Provides convenient methods for creating publication-quality plots
following UC Berkeley's visual identity guidelines.
#### Methods
##### `__init__(self, figsize, style)`
Initialize Berkeley-styled plot.
Parameters
----------
figsize : tuple, default (10, 6)
Figure size in inches
style : str, default 'publication'
Plot style ('publication', 'presentation', 'poster')
**Source:** [Line 169](Python/visualization/berkeley_style.py#L169)
##### `create_figure(self, nrows, ncols, subplot_kw, gridspec_kw)`
Create Berkeley-styled figure and axes.
Parameters
----------
nrows, ncols : int, default 1
Number of subplot rows and columns
subplot_kw : dict, optional
Additional subplot keywords
gridspec_kw : dict, optional
GridSpec keywords
Returns
-------
fig : Figure
Matplotlib figure
axes : Axes or array of Axes
Subplot axes
**Source:** [Line 202](Python/visualization/berkeley_style.py#L202)
##### `line_plot(self, x, y, labels, colors, linestyles, markers, title, xlabel, ylabel, legend, ax)`
Create Berkeley-styled line plot.
Parameters
----------
x : ndarray
X-axis data
y : ndarray or list of ndarray
Y-axis data (single array or list for multiple lines)
labels : list of str, optional
Line labels for legend
colors : list of str, optional
Line colors (uses Berkeley sequence if None)
linestyles : list of str, optional
Line styles
markers : list of str, optional
Marker styles
title : str, optional
Plot title
xlabel, ylabel : str, optional
Axis labels
legend : bool, default True
Whether to show legend
ax : Axes, optional
Matplotlib axes to plot on
Returns
-------
Axes
Matplotlib axes object
**Source:** [Line 229](Python/visualization/berkeley_style.py#L229)
##### `scatter_plot(self, x, y, c, s, alpha, cmap, title, xlabel, ylabel, colorbar, ax)`
Create Berkeley-styled scatter plot.
Parameters
----------
x, y : ndarray
Scatter plot coordinates
c : ndarray, optional
Color values for points
s : float or ndarray, optional
Point sizes
alpha : float, default 0.7
Point transparency
cmap : str, default 'berkeley_spectrum'
Colormap name
title : str, optional
Plot title
xlabel, ylabel : str, optional
Axis labels
colorbar : bool, default True
Whether to show colorbar
ax : Axes, optional
Matplotlib axes to plot on
Returns
-------
Axes
Matplotlib axes object
**Source:** [Line 315](Python/visualization/berkeley_style.py#L315)
##### `heatmap(self, data, x_labels, y_labels, cmap, title, xlabel, ylabel, colorbar, ax)`
Create Berkeley-styled heatmap.
Parameters
----------
data : ndarray
2D data array
x_labels, y_labels : list of str, optional
Axis tick labels
cmap : str, default 'berkeley_blues'
Colormap name
title : str, optional
Plot title
xlabel, ylabel : str, optional
Axis labels
colorbar : bool, default True
Whether to show colorbar
ax : Axes, optional
Matplotlib axes to plot on
Returns
-------
Axes
Matplotlib axes object
**Source:** [Line 386](Python/visualization/berkeley_style.py#L386)
##### `wavefunction(self, x, psi, title, show_probability, ax)`
Plot quantum wavefunction with Berkeley styling.
Parameters
----------
x : ndarray
Position coordinates
psi : ndarray
Wavefunction values (can be complex)
title : str, optional
Plot title
show_probability : bool, default True
Whether to show probability density
ax : Axes, optional
Matplotlib axes to plot on
Returns
-------
Axes
Matplotlib axes object
**Source:** [Line 451](Python/visualization/berkeley_style.py#L451)
##### `save_figure(self, filename, dpi, format, bbox_inches, pad_inches, transparent)`
Save figure with Berkeley styling.
Parameters
----------
filename : str or Path
Output filename
dpi : int, default 300
Resolution in dots per inch
format : str, default 'png'
Output format
bbox_inches : str, default 'tight'
Bounding box setting
pad_inches : float, default 0.1
Padding around figure
transparent : bool, default False
Whether background should be transparent
**Source:** [Line 506](Python/visualization/berkeley_style.py#L506)
**Class Source:** [Line 161](Python/visualization/berkeley_style.py#L161)