# post_processing
**Module:** `Python/FEM/core/post_processing.py`
## Overview
Post-Processing and Visualization for Finite Element Analysis
Comprehensive post-processing tools for FEM results including stress visualization,
deformed shape plotting, contour analysis, and result extraction.
## Classes
### `FEMPostProcessor`
Post-processing system for finite element results.
Features:
- Deformed shape visualization
- Stress contour plots
- Result extraction and analysis
- Animation capabilities
#### Methods
##### `__init__(self, assembly, displacement)`
Initialize post-processor.
Parameters:
assembly: Global assembly system
displacement: Global displacement vector
**Source:** [Line 31](Python/FEM/core/post_processing.py#L31)
##### `plot_deformed_shape(self, scale_factor, show_undeformed, figsize)`
Plot deformed shape of the structure.
Parameters:
scale_factor: Displacement scaling factor for visualization
show_undeformed: Show undeformed mesh overlay
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 53](Python/FEM/core/post_processing.py#L53)
##### `_plot_deformed_2d(self, ax, scale_factor, show_undeformed)`
Plot 2D deformed shape.
**Source:** [Line 91](Python/FEM/core/post_processing.py#L91)
##### `_plot_deformed_1d(self, ax, scale_factor, show_undeformed)`
Plot 1D deformed shape.
**Source:** [Line 129](Python/FEM/core/post_processing.py#L129)
##### `_plot_deformed_3d(self, ax, scale_factor, show_undeformed)`
Plot 3D deformed shape (simplified surface plot).
**Source:** [Line 161](Python/FEM/core/post_processing.py#L161)
##### `plot_stress_contours(self, stress_component, figsize)`
Plot stress contours.
Parameters:
stress_component: Stress component to plot ('von_mises', 'xx', 'yy', 'xy')
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 167](Python/FEM/core/post_processing.py#L167)
##### `_extrapolate_stresses_to_nodes(self, stress_component)`
Extrapolate element stresses to nodes.
**Source:** [Line 230](Python/FEM/core/post_processing.py#L230)
##### `_plot_mesh_overlay(self, ax, alpha)`
Plot mesh overlay on contour plot.
**Source:** [Line 272](Python/FEM/core/post_processing.py#L272)
##### `plot_displacement_vectors(self, scale_factor, figsize)`
Plot displacement vectors.
Parameters:
scale_factor: Vector scaling factor
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 282](Python/FEM/core/post_processing.py#L282)
##### `generate_results_summary(self)`
Generate comprehensive results summary.
Returns:
Results summary dictionary
**Source:** [Line 334](Python/FEM/core/post_processing.py#L334)
##### `_compute_stress_statistics(self)`
Compute stress statistics.
**Source:** [Line 359](Python/FEM/core/post_processing.py#L359)
##### `_summarize_reaction_forces(self)`
Summarize reaction forces.
**Source:** [Line 386](Python/FEM/core/post_processing.py#L386)
##### `export_results(self, filename, format)`
Export results to file.
Parameters:
filename: Output filename
format: Export format ('vtk', 'csv', 'json')
**Source:** [Line 410](Python/FEM/core/post_processing.py#L410)
##### `_export_vtk(self, filename)`
Export results in VTK format.
**Source:** [Line 427](Python/FEM/core/post_processing.py#L427)
##### `_export_csv(self, filename)`
Export results in CSV format.
**Source:** [Line 431](Python/FEM/core/post_processing.py#L431)
##### `_export_json(self, filename)`
Export results in JSON format.
**Source:** [Line 456](Python/FEM/core/post_processing.py#L456)
**Class Source:** [Line 20](Python/FEM/core/post_processing.py#L20)