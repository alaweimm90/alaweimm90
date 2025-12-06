# scientific_plots
**Module:** `Python/Plotting/scientific_plots.py`
## Overview
Scientific Plotting Module
=========================
Professional scientific plotting with Berkeley branding and publication-ready formatting.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
- **`BERKELEY_PALETTE`**
## Functions
### `demo()`
Demonstrate scientific plotting capabilities.
**Source:** [Line 496](Python/Plotting/scientific_plots.py#L496)
## Classes
### `ScientificPlot`
Professional scientific plotting class with Berkeley branding.
Provides publication-ready plots with consistent styling,
proper error handling, and scientific formatting conventions.
#### Methods
##### `__init__(self, style, dpi)`
Initialize scientific plot with Berkeley styling.
Args:
style: Plot style ('berkeley', 'minimal', 'presentation')
dpi: Resolution for saved figures
**Source:** [Line 42](Python/Plotting/scientific_plots.py#L42)
##### `_setup_style(self)`
Configure matplotlib parameters for scientific plotting.
**Source:** [Line 58](Python/Plotting/scientific_plots.py#L58)
##### `create_figure(self, figsize, nrows, ncols)`
Create a new figure with Berkeley styling.
Args:
figsize: Figure size in inches
nrows: Number of subplot rows
ncols: Number of subplot columns
Returns:
Tuple of (figure, axes)
**Source:** [Line 94](Python/Plotting/scientific_plots.py#L94)
##### `_add_berkeley_watermark(self, fig, ax)`
Add subtle Berkeley branding to the plot.
**Source:** [Line 119](Python/Plotting/scientific_plots.py#L119)
##### `plot(self, x, y, ax, label, color, style, marker)`
Create a line plot with Berkeley styling.
Args:
x: X-axis data
y: Y-axis data
ax: Axes to plot on (current axes if None)
label: Data series label
color: Line color (auto-select if None)
style: Line style
marker: Marker style
**kwargs: Additional plot parameters
Returns:
Line2D object
**Source:** [Line 126](Python/Plotting/scientific_plots.py#L126)
##### `scatter(self, x, y, ax, label, color, size, alpha)`
Create a scatter plot with Berkeley styling.
Args:
x: X-axis data
y: Y-axis data
ax: Axes to plot on
label: Data series label
color: Point color
size: Point size(s)
alpha: Transparency
**kwargs: Additional scatter parameters
Returns:
PathCollection object
**Source:** [Line 160](Python/Plotting/scientific_plots.py#L160)
##### `errorbar(self, x, y, yerr, xerr, ax, label, color)`
Create an error bar plot.
Args:
x: X-axis data
y: Y-axis data
yerr: Y-axis error bars
xerr: X-axis error bars
ax: Axes to plot on
label: Data series label
color: Plot color
**kwargs: Additional errorbar parameters
Returns:
ErrorbarContainer object
**Source:** [Line 194](Python/Plotting/scientific_plots.py#L194)
##### `fill_between(self, x, y1, y2, ax, label, color, alpha)`
Fill area between curves.
Args:
x: X-axis data
y1: Upper boundary
y2: Lower boundary
ax: Axes to plot on
label: Fill label
color: Fill color
alpha: Transparency
**kwargs: Additional fill parameters
Returns:
PolyCollection object
**Source:** [Line 229](Python/Plotting/scientific_plots.py#L229)
##### `set_labels(self, xlabel, ylabel, title, ax)`
Set axis labels and title with Berkeley styling.
Args:
xlabel: X-axis label
ylabel: Y-axis label
title: Plot title
ax: Axes to modify
**Source:** [Line 262](Python/Plotting/scientific_plots.py#L262)
##### `add_legend(self, ax, location)`
Add a professionally styled legend.
Args:
ax: Axes to add legend to
location: Legend location
**kwargs: Additional legend parameters
**Source:** [Line 282](Python/Plotting/scientific_plots.py#L282)
##### `add_grid(self, ax, alpha, which)`
Add a professional grid.
Args:
ax: Axes to add grid to
alpha: Grid transparency
which: Grid type ('major', 'minor', 'both')
**Source:** [Line 305](Python/Plotting/scientific_plots.py#L305)
##### `add_inset(self, ax, bounds, xlim, ylim)`
Add an inset plot.
Args:
ax: Parent axes
bounds: Inset bounds [x, y, width, height] in parent coordinates
xlim: Inset x-axis limits
ylim: Inset y-axis limits
Returns:
Inset axes
**Source:** [Line 321](Python/Plotting/scientific_plots.py#L321)
##### `add_annotation(self, text, xy, xytext, ax)`
Add an annotation with arrow.
Args:
text: Annotation text
xy: Point to annotate
xytext: Text position
ax: Axes to annotate
**kwargs: Additional annotation parameters
**Source:** [Line 348](Python/Plotting/scientific_plots.py#L348)
##### `save_figure(self, filename, fig)`
Save figure with high quality settings.
Args:
filename: Output filename
fig: Figure to save (current figure if None)
**kwargs: Additional savefig parameters
**Source:** [Line 378](Python/Plotting/scientific_plots.py#L378)
##### `_get_next_color(self)`
Get the next color in the Berkeley palette.
**Source:** [Line 402](Python/Plotting/scientific_plots.py#L402)
##### `reset_colors(self)`
Reset color cycling.
**Source:** [Line 408](Python/Plotting/scientific_plots.py#L408)
**Class Source:** [Line 34](Python/Plotting/scientific_plots.py#L34)
### `Plot3D`
3D scientific plotting with Berkeley styling.
#### Methods
##### `__init__(self, style)`
Initialize 3D plot with Berkeley styling.
**Source:** [Line 417](Python/Plotting/scientific_plots.py#L417)
##### `create_figure(self, figsize)`
Create a 3D figure.
Args:
figsize: Figure size
Returns:
Tuple of (figure, 3D axes)
**Source:** [Line 423](Python/Plotting/scientific_plots.py#L423)
##### `surface(self, X, Y, Z, ax, colormap, alpha)`
Create a 3D surface plot.
Args:
X: X-axis meshgrid
Y: Y-axis meshgrid
Z: Surface heights
ax: 3D axes
colormap: Color map name
alpha: Surface transparency
**kwargs: Additional surface parameters
**Source:** [Line 449](Python/Plotting/scientific_plots.py#L449)
##### `scatter3d(self, x, y, z, ax, color, size)`
Create a 3D scatter plot.
Args:
x: X-axis data
y: Y-axis data
z: Z-axis data
ax: 3D axes
color: Point color
size: Point size
**kwargs: Additional scatter parameters
**Source:** [Line 469](Python/Plotting/scientific_plots.py#L469)
##### `_get_next_color(self)`
Get the next color in the Berkeley palette.
**Source:** [Line 490](Python/Plotting/scientific_plots.py#L490)
**Class Source:** [Line 412](Python/Plotting/scientific_plots.py#L412)