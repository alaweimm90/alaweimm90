"""
Post-Processing and Visualization for Finite Element Analysis
Comprehensive post-processing tools for FEM results including stress visualization,
deformed shape plotting, contour analysis, and result extraction.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import List, Dict, Tuple, Optional, Union
import warnings
from .assembly import GlobalAssembly
from .mesh_generation import Mesh
class FEMPostProcessor:
    """
    Post-processing system for finite element results.
    Features:
    - Deformed shape visualization
    - Stress contour plots
    - Result extraction and analysis
    - Animation capabilities
    """
    def __init__(self, assembly: GlobalAssembly, displacement: np.ndarray):
        """
        Initialize post-processor.
        Parameters:
            assembly: Global assembly system
            displacement: Global displacement vector
        """
        self.assembly = assembly
        self.displacement = displacement
        self.mesh = assembly.mesh
        # Compute derived quantities
        self.element_stresses = assembly.compute_element_stresses(displacement)
        self.reaction_forces = assembly.compute_reaction_forces(displacement)
        self.energies = assembly.compute_system_energy(displacement)
        # Berkeley color scheme
        self.berkeley_blue = '#003262'
        self.california_gold = '#FDB515'
        self.berkeley_colors = [self.berkeley_blue, self.california_gold, '#859438', '#D9661F']
    def plot_deformed_shape(self, scale_factor: float = 1.0, show_undeformed: bool = True,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot deformed shape of the structure.
        Parameters:
            scale_factor: Displacement scaling factor for visualization
            show_undeformed: Show undeformed mesh overlay
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        if self.mesh.dimension == 2:
            self._plot_deformed_2d(ax, scale_factor, show_undeformed)
        elif self.mesh.dimension == 3:
            self._plot_deformed_3d(ax, scale_factor, show_undeformed)
        else:
            self._plot_deformed_1d(ax, scale_factor, show_undeformed)
        ax.set_title(f'Deformed Shape (Scale Factor: {scale_factor})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        # Add legend
        legend_elements = []
        if show_undeformed:
            legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--',
                                            label='Undeformed'))
        legend_elements.append(plt.Line2D([0], [0], color=self.berkeley_blue,
                                        label='Deformed'))
        ax.legend(handles=legend_elements)
        return fig
    def _plot_deformed_2d(self, ax: plt.Axes, scale_factor: float, show_undeformed: bool):
        """Plot 2D deformed shape."""
        # Get node coordinates
        coords = self.mesh.get_node_coordinates()
        # Calculate deformed coordinates
        deformed_coords = coords.copy()
        for node_id, node in self.mesh.nodes.items():
            node_displacement = self.assembly.get_displacement_at_node(self.displacement, node_id)
            deformed_coords[node_id] = node.coordinates[:2] + scale_factor * node_displacement[:2]
        # Plot undeformed mesh
        if show_undeformed:
            for element in self.mesh.elements.values():
                if element.element_type in ['triangle2d', 'quad2d']:
                    element_coords = np.array([coords[nid][:2] for nid in element.node_ids])
                    element_coords = np.vstack([element_coords, element_coords[0]])  # Close polygon
                    ax.plot(element_coords[:, 0], element_coords[:, 1],
                           'k--', alpha=0.5, linewidth=1, label='Undeformed' if element.id == 0 else "")
        # Plot deformed mesh
        for element in self.mesh.elements.values():
            if element.element_type in ['triangle2d', 'quad2d']:
                element_coords = np.array([deformed_coords[nid] for nid in element.node_ids])
                element_coords = np.vstack([element_coords, element_coords[0]])  # Close polygon
                ax.plot(element_coords[:, 0], element_coords[:, 1],
                       color=self.berkeley_blue, linewidth=2)
                ax.fill(element_coords[:, 0], element_coords[:, 1],
                       color=self.berkeley_blue, alpha=0.1)
        # Plot nodes
        deformed_node_coords = np.array(list(deformed_coords.values()))
        ax.scatter(deformed_node_coords[:, 0], deformed_node_coords[:, 1],
                  c=self.california_gold, s=30, zorder=5, edgecolors='black')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    def _plot_deformed_1d(self, ax: plt.Axes, scale_factor: float, show_undeformed: bool):
        """Plot 1D deformed shape."""
        # Get nodal coordinates and displacements
        node_ids = sorted(self.mesh.nodes.keys())
        x_coords = []
        y_undeformed = []
        y_deformed = []
        for node_id in node_ids:
            node = self.mesh.nodes[node_id]
            x_coords.append(node.coordinates[0])
            y_undeformed.append(0.0)
            displacement = self.assembly.get_displacement_at_node(self.displacement, node_id)
            y_deformed.append(scale_factor * displacement[0])
        x_coords = np.array(x_coords)
        y_undeformed = np.array(y_undeformed)
        y_deformed = np.array(y_deformed)
        # Plot undeformed
        if show_undeformed:
            ax.plot(x_coords, y_undeformed, 'k--', linewidth=2, label='Undeformed')
            ax.scatter(x_coords, y_undeformed, c='gray', s=50, zorder=5)
        # Plot deformed
        ax.plot(x_coords, y_deformed, color=self.berkeley_blue, linewidth=3, label='Deformed')
        ax.scatter(x_coords, y_deformed, c=self.california_gold, s=80, zorder=5, edgecolors='black')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Displacement')
    def _plot_deformed_3d(self, ax: plt.Axes, scale_factor: float, show_undeformed: bool):
        """Plot 3D deformed shape (simplified surface plot)."""
        warnings.warn("3D deformed shape plotting not fully implemented")
        ax.text(0.5, 0.5, '3D Plotting Not Implemented',
               transform=ax.transAxes, ha='center', va='center', fontsize=16)
    def plot_stress_contours(self, stress_component: str = 'von_mises',
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot stress contours.
        Parameters:
            stress_component: Stress component to plot ('von_mises', 'xx', 'yy', 'xy')
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        if self.mesh.dimension != 2:
            raise NotImplementedError("Stress contours only implemented for 2D problems")
        fig, ax = plt.subplots(figsize=figsize)
        # Extract stress values at nodes
        nodal_stresses = self._extrapolate_stresses_to_nodes(stress_component)
        # Create triangulation for contour plotting
        coords = self.mesh.get_node_coordinates()
        x = coords[:, 0]
        y = coords[:, 1]
        # Get triangular connectivity
        triangles = []
        for element in self.mesh.elements.values():
            if element.element_type == 'triangle2d':
                triangles.append(element.node_ids)
            elif element.element_type == 'quad2d':
                # Split quad into two triangles
                nodes = element.node_ids
                triangles.append([nodes[0], nodes[1], nodes[2]])
                triangles.append([nodes[0], nodes[2], nodes[3]])
        if triangles:
            tri = Triangulation(x, y, triangles)
            # Create contour plot
            levels = 20
            contour = ax.tricontourf(tri, nodal_stresses, levels=levels, cmap='viridis')
            # Add contour lines
            ax.tricontour(tri, nodal_stresses, levels=levels, colors='black',
                         alpha=0.3, linewidths=0.5)
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label(f'{stress_component.replace("_", " ").title()} Stress (Pa)')
        # Overlay mesh
        self._plot_mesh_overlay(ax)
        ax.set_title(f'{stress_component.replace("_", " ").title()} Stress Distribution',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return fig
    def _extrapolate_stresses_to_nodes(self, stress_component: str) -> np.ndarray:
        """Extrapolate element stresses to nodes."""
        num_nodes = len(self.mesh.nodes)
        nodal_stresses = np.zeros(num_nodes)
        nodal_counts = np.zeros(num_nodes)
        for element in self.mesh.elements.values():
            if element.id in self.element_stresses:
                element_stress = self.element_stresses[element.id]
                # Extract stress component
                if stress_component == 'von_mises':
                    if len(element_stress) == 3:  # 2D case: [σxx, σyy, τxy]
                        sxx, syy, sxy = element_stress
                        stress_value = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2)
                    elif len(element_stress) == 6:  # 3D case
                        sxx, syy, szz, syz, sxz, sxy = element_stress
                        stress_value = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 +
                                                     6*(sxy**2 + syz**2 + sxz**2)))
                    else:
                        stress_value = np.abs(element_stress[0])  # 1D case
                elif stress_component == 'xx':
                    stress_value = element_stress[0]
                elif stress_component == 'yy':
                    stress_value = element_stress[1] if len(element_stress) > 1 else 0
                elif stress_component == 'xy':
                    stress_value = element_stress[2] if len(element_stress) > 2 else 0
                else:
                    stress_value = element_stress[0]
                # Add to all nodes of element
                for node_id in element.node_ids:
                    nodal_stresses[node_id] += stress_value
                    nodal_counts[node_id] += 1
        # Average stresses at nodes
        nonzero_counts = nodal_counts > 0
        nodal_stresses[nonzero_counts] /= nodal_counts[nonzero_counts]
        return nodal_stresses
    def _plot_mesh_overlay(self, ax: plt.Axes, alpha: float = 0.3):
        """Plot mesh overlay on contour plot."""
        for element in self.mesh.elements.values():
            if element.element_type in ['triangle2d', 'quad2d']:
                coords = self.mesh.get_node_coordinates()
                element_coords = np.array([coords[nid][:2] for nid in element.node_ids])
                element_coords = np.vstack([element_coords, element_coords[0]])
                ax.plot(element_coords[:, 0], element_coords[:, 1],
                       'k-', alpha=alpha, linewidth=0.5)
    def plot_displacement_vectors(self, scale_factor: float = 1.0,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot displacement vectors.
        Parameters:
            scale_factor: Vector scaling factor
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        if self.mesh.dimension != 2:
            raise NotImplementedError("Displacement vectors only implemented for 2D problems")
        fig, ax = plt.subplots(figsize=figsize)
        # Get coordinates and displacements
        coords = self.mesh.get_node_coordinates()
        x = coords[:, 0]
        y = coords[:, 1]
        u = np.zeros(len(self.mesh.nodes))
        v = np.zeros(len(self.mesh.nodes))
        for node_id, node in self.mesh.nodes.items():
            displacement = self.assembly.get_displacement_at_node(self.displacement, node_id)
            u[node_id] = displacement[0]
            v[node_id] = displacement[1] if len(displacement) > 1 else 0
        # Plot displacement vectors
        magnitude = np.sqrt(u**2 + v**2)
        quiver = ax.quiver(x, y, scale_factor * u, scale_factor * v,
                          magnitude, cmap='viridis', scale=1, scale_units='xy',
                          angles='xy', width=0.003)
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label('Displacement Magnitude')
        # Plot mesh
        self._plot_mesh_overlay(ax)
        ax.set_title(f'Displacement Vectors (Scale Factor: {scale_factor})',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return fig
    def generate_results_summary(self) -> Dict[str, Union[float, Dict]]:
        """
        Generate comprehensive results summary.
        Returns:
            Results summary dictionary
        """
        summary = {
            'mesh_info': {
                'num_nodes': len(self.mesh.nodes),
                'num_elements': len(self.mesh.elements),
                'dimension': self.mesh.dimension
            },
            'displacement_stats': {
                'max_displacement': float(np.max(np.abs(self.displacement))),
                'max_displacement_node': int(np.argmax(np.abs(self.displacement))),
                'rms_displacement': float(np.sqrt(np.mean(self.displacement**2)))
            },
            'stress_stats': self._compute_stress_statistics(),
            'energy_info': self.energies,
            'reaction_forces': self._summarize_reaction_forces()
        }
        return summary
    def _compute_stress_statistics(self) -> Dict[str, float]:
        """Compute stress statistics."""
        all_von_mises = []
        for element_stress in self.element_stresses.values():
            if len(element_stress) == 3:  # 2D
                sxx, syy, sxy = element_stress
                von_mises = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2)
            elif len(element_stress) == 6:  # 3D
                sxx, syy, szz, syz, sxz, sxy = element_stress
                von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 +
                                          6*(sxy**2 + syz**2 + sxz**2)))
            else:  # 1D
                von_mises = abs(element_stress[0])
            all_von_mises.append(von_mises)
        if all_von_mises:
            return {
                'max_von_mises': float(np.max(all_von_mises)),
                'min_von_mises': float(np.min(all_von_mises)),
                'avg_von_mises': float(np.mean(all_von_mises)),
                'std_von_mises': float(np.std(all_von_mises))
            }
        else:
            return {}
    def _summarize_reaction_forces(self) -> Dict[str, Union[float, int]]:
        """Summarize reaction forces."""
        if not self.reaction_forces:
            return {}
        total_reaction = 0.0
        max_reaction = 0.0
        max_reaction_node = -1
        for node_id, reaction_vector in self.reaction_forces.items():
            reaction_magnitude = np.linalg.norm(reaction_vector)
            total_reaction += reaction_magnitude
            if reaction_magnitude > max_reaction:
                max_reaction = reaction_magnitude
                max_reaction_node = node_id
        return {
            'total_reaction_magnitude': float(total_reaction),
            'max_reaction_magnitude': float(max_reaction),
            'max_reaction_node': int(max_reaction_node),
            'num_reaction_points': len(self.reaction_forces)
        }
    def export_results(self, filename: str, format: str = 'vtk'):
        """
        Export results to file.
        Parameters:
            filename: Output filename
            format: Export format ('vtk', 'csv', 'json')
        """
        if format.lower() == 'vtk':
            self._export_vtk(filename)
        elif format.lower() == 'csv':
            self._export_csv(filename)
        elif format.lower() == 'json':
            self._export_json(filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    def _export_vtk(self, filename: str):
        """Export results in VTK format."""
        warnings.warn("VTK export not implemented yet")
    def _export_csv(self, filename: str):
        """Export results in CSV format."""
        import csv
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            if self.mesh.dimension == 2:
                writer.writerow(['Node_ID', 'X', 'Y', 'U_X', 'U_Y', 'Displacement_Magnitude'])
            else:
                writer.writerow(['Node_ID', 'X', 'U_X', 'Displacement_Magnitude'])
            # Write data
            for node_id, node in self.mesh.nodes.items():
                displacement = self.assembly.get_displacement_at_node(self.displacement, node_id)
                displacement_magnitude = np.linalg.norm(displacement)
                if self.mesh.dimension == 2:
                    writer.writerow([node_id, node.coordinates[0], node.coordinates[1],
                                   displacement[0], displacement[1], displacement_magnitude])
                else:
                    writer.writerow([node_id, node.coordinates[0], displacement[0],
                                   displacement_magnitude])
    def _export_json(self, filename: str):
        """Export results in JSON format."""
        import json
        results = self.generate_results_summary()
        # Add detailed node data
        results['nodes'] = {}
        for node_id, node in self.mesh.nodes.items():
            displacement = self.assembly.get_displacement_at_node(self.displacement, node_id)
            results['nodes'][str(node_id)] = {
                'coordinates': node.coordinates.tolist(),
                'displacement': displacement.tolist()
            }
        # Add element data
        results['elements'] = {}
        for element_id, stress in self.element_stresses.items():
            results['elements'][str(element_id)] = {
                'stress': stress.tolist()
            }
        with open(filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=2)