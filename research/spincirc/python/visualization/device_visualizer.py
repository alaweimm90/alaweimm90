#!/usr/bin/env python3
"""
3D Device Visualization for SpinCirc

Advanced 3D visualization of spintronic devices including geometry,
magnetic fields, current flow, and material properties.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

class DeviceVisualizer:
    """3D visualization of spintronic devices"""
    
    def __init__(self, berkeley_colors: bool = True):
        """
        Initialize device visualizer
        
        Args:
            berkeley_colors: Use Berkeley color scheme
        """
        self.berkeley_colors = berkeley_colors
        
        if berkeley_colors:
            from .berkeley_plots import BerkeleyColors
            self.colors = BerkeleyColors()
        else:
            # Default colors
            self.colors = type('Colors', (), {
                'BERKELEY_BLUE': '#003262',
                'CALIFORNIA_GOLD': '#FDB515',
                'GREEN_DARK': '#00553A',
                'ROSE_DARK': '#77072F',
                'SEQUENCE': ['#003262', '#FDB515', '#00553A', '#77072F']
            })()
    
    def visualize_mtj_stack(self, layer_thicknesses: List[float],
                           layer_materials: List[str],
                           layer_colors: Optional[List[str]] = None,
                           title: str = 'MTJ Stack Structure',
                           save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize MTJ layer stack in 3D"""
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Default colors if not provided
        if layer_colors is None:
            color_cycle = [self.colors.BERKELEY_BLUE, self.colors.CALIFORNIA_GOLD,
                          self.colors.GREEN_DARK, self.colors.ROSE_DARK]
            layer_colors = [color_cycle[i % len(color_cycle)] for i in range(len(layer_materials))]
        
        # Device dimensions
        width, length = 5, 5  # Lateral dimensions
        
        # Build stack from bottom to top
        z_bottom = 0
        
        for i, (thickness, material, color) in enumerate(zip(layer_thicknesses, layer_materials, layer_colors)):
            z_top = z_bottom + thickness
            
            # Create layer as a 3D box
            self._draw_3d_box(ax, 0, 0, z_bottom, width, length, thickness, color, alpha=0.7)
            
            # Add material label
            ax.text(width/2, length + 1, (z_bottom + z_top)/2, material,
                   fontsize=12, ha='center', va='center')
            
            # Add thickness annotation
            ax.text(-1, length/2, (z_bottom + z_top)/2, f'{thickness:.1f} nm',
                   fontsize=10, ha='center', va='center', rotation=90)
            
            z_bottom = z_top
        
        # Add magnetization arrows for magnetic layers
        magnetic_layers = ['CoFeB', 'Co', 'Fe', 'Permalloy', 'NiFe']
        z_bottom = 0
        
        for thickness, material in zip(layer_thicknesses, layer_materials):
            if any(mag_mat in material for mag_mat in magnetic_layers):
                z_center = z_bottom + thickness/2
                
                # Arrow showing magnetization direction
                arrow_props = dict(arrowstyle='->', color='red', lw=3)
                ax.annotate('', xyz=(width*0.8, length/2, z_center),
                           xytext=(width*0.2, length/2, z_center),
                           arrowprops=arrow_props)
            
            z_bottom += thickness
        
        # Styling
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Set equal aspect ratio
        max_dim = max(width, length, sum(layer_thicknesses))
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def visualize_spin_valve_geometry(self, device_params: Dict[str, float],
                                    show_current_flow: bool = True,
                                    title: str = 'Spin Valve Geometry',
                                    save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize spin valve device geometry"""
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract parameters
        length = device_params.get('length', 200e-9) * 1e9  # Convert to nm
        width = device_params.get('width', 100e-9) * 1e9
        thickness = device_params.get('thickness', 2e-9) * 1e9
        
        # Bottom ferromagnet
        fm1_thickness = 3
        self._draw_3d_box(ax, 0, 0, 0, length, width, fm1_thickness, 
                         self.colors.BERKELEY_BLUE, alpha=0.8)
        ax.text(length/2, width/2, fm1_thickness/2, 'Bottom FM',
               ha='center', va='center', fontsize=12, color='white', weight='bold')
        
        # Spacer layer
        spacer_thickness = thickness
        self._draw_3d_box(ax, 0, 0, fm1_thickness, length, width, spacer_thickness,
                         self.colors.CALIFORNIA_GOLD, alpha=0.6)
        ax.text(length/2, width/2, fm1_thickness + spacer_thickness/2, 'Spacer',
               ha='center', va='center', fontsize=12, weight='bold')
        
        # Top ferromagnet
        fm2_thickness = 3
        self._draw_3d_box(ax, 0, 0, fm1_thickness + spacer_thickness, 
                         length, width, fm2_thickness,
                         self.colors.GREEN_DARK, alpha=0.8)
        ax.text(length/2, width/2, fm1_thickness + spacer_thickness + fm2_thickness/2,
               'Top FM', ha='center', va='center', fontsize=12, color='white', weight='bold')
        
        # Contacts
        contact_thickness = 2
        contact_width = width * 0.3
        
        # Left contact
        self._draw_3d_box(ax, -20, (width-contact_width)/2, fm1_thickness,
                         20, contact_width, spacer_thickness,
                         self.colors.SEQUENCE[3], alpha=0.9)
        
        # Right contact
        self._draw_3d_box(ax, length, (width-contact_width)/2, fm1_thickness,
                         20, contact_width, spacer_thickness,
                         self.colors.SEQUENCE[3], alpha=0.9)
        
        # Current flow visualization
        if show_current_flow:
            # Current streamlines
            y_positions = np.linspace(width*0.2, width*0.8, 5)
            z_center = fm1_thickness + spacer_thickness/2
            
            for y_pos in y_positions:
                # Current path
                x_path = np.linspace(-15, length + 15, 100)
                y_path = np.full_like(x_path, y_pos)
                z_path = np.full_like(x_path, z_center)
                
                ax.plot(x_path, y_path, z_path, color='blue', linewidth=2, alpha=0.7)
                
                # Arrow at end
                ax.quiver(length + 10, y_pos, z_center, 5, 0, 0,
                         color='blue', arrow_length_ratio=0.3, linewidth=2)
        
        # Magnetization directions
        # Bottom FM - left pointing
        ax.quiver(length*0.2, width/2, fm1_thickness/2, -length*0.3, 0, 0,
                 color='red', arrow_length_ratio=0.1, linewidth=4)
        ax.text(length*0.1, width/2, fm1_thickness/2 + 5, 'M₁',
               fontsize=14, color='red', weight='bold')
        
        # Top FM - right pointing (antiparallel)
        ax.quiver(length*0.2, width/2, fm1_thickness + spacer_thickness + fm2_thickness/2,
                 length*0.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=4)
        ax.text(length*0.35, width/2, fm1_thickness + spacer_thickness + fm2_thickness/2 + 5,
               'M₂', fontsize=14, color='red', weight='bold')
        
        # Labels and dimensions
        ax.text(-30, width/2, -10, f'L = {length:.0f} nm', fontsize=12)
        ax.text(length/2, -20, -10, f'W = {width:.0f} nm', fontsize=12)
        ax.text(length + 30, width/2, spacer_thickness, f't = {thickness:.1f} nm',
               fontsize=12, rotation=90)
        
        # Styling
        ax.set_xlabel('Length (nm)')
        ax.set_ylabel('Width (nm)')
        ax.set_zlabel('Height (nm)')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Set limits
        total_height = fm1_thickness + spacer_thickness + fm2_thickness
        ax.set_xlim(-40, length + 40)
        ax.set_ylim(-30, width + 30)
        ax.set_zlim(-5, total_height + 10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def visualize_magnetic_field(self, x_range: Tuple[float, float],
                               y_range: Tuple[float, float],
                               z_range: Tuple[float, float],
                               field_function: callable,
                               title: str = 'Magnetic Field Visualization',
                               save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize 3D magnetic field distribution"""
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], 15)
        y = np.linspace(y_range[0], y_range[1], 15)
        z = np.linspace(z_range[0], z_range[1], 10)
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Calculate field
        Hx, Hy, Hz = field_function(X, Y, Z)
        
        # Field magnitude
        H_mag = np.sqrt(Hx**2 + Hy**2 + Hz**2)
        
        # Normalize for display
        H_max = np.max(H_mag)
        if H_max > 0:
            Hx_norm = Hx / H_max
            Hy_norm = Hy / H_max
            Hz_norm = Hz / H_max
        else:
            Hx_norm, Hy_norm, Hz_norm = Hx, Hy, Hz
        
        # Subsample for cleaner visualization
        skip = 2
        X_sub = X[::skip, ::skip, ::skip]
        Y_sub = Y[::skip, ::skip, ::skip]
        Z_sub = Z[::skip, ::skip, ::skip]
        Hx_sub = Hx_norm[::skip, ::skip, ::skip]
        Hy_sub = Hy_norm[::skip, ::skip, ::skip]
        Hz_sub = Hz_norm[::skip, ::skip, ::skip]
        H_mag_sub = H_mag[::skip, ::skip, ::skip]
        
        # 3D quiver plot
        quiver = ax.quiver(X_sub, Y_sub, Z_sub, Hx_sub, Hy_sub, Hz_sub,
                          length=0.1, normalize=True, cmap='viridis',
                          alpha=0.7, linewidth=1.5)
        
        # Color by magnitude
        quiver.set_array(H_mag_sub.flatten())
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Field Magnitude (T)', fontsize=12)
        
        # Field lines (simplified - show a few selected ones)
        self._add_field_lines(ax, field_function, x_range, y_range, z_range)
        
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def visualize_current_density(self, x_grid: np.ndarray, y_grid: np.ndarray,
                                current_density: np.ndarray,
                                title: str = 'Current Density Distribution',
                                save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize 2D current density distribution"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Contour plot
        J_mag = np.sqrt(current_density[0]**2 + current_density[1]**2)
        
        contour = ax1.contourf(x_grid * 1e9, y_grid * 1e9, J_mag,
                              levels=20, cmap='viridis', alpha=0.8)
        
        # Add streamlines
        ax1.streamplot(x_grid * 1e9, y_grid * 1e9,
                      current_density[0], current_density[1],
                      color='white', linewidth=1.5, density=2, alpha=0.7)
        
        ax1.set_xlabel('X (nm)')
        ax1.set_ylabel('Y (nm)')
        ax1.set_title('Current Density Magnitude')
        ax1.set_aspect('equal')
        
        cbar1 = plt.colorbar(contour, ax=ax1)
        cbar1.set_label('|J| (A/m²)', fontsize=12)
        
        # Vector field plot
        skip = slice(None, None, 3)  # Subsample for cleaner arrows
        
        ax2.quiver(x_grid[skip, skip] * 1e9, y_grid[skip, skip] * 1e9,
                  current_density[0][skip, skip], current_density[1][skip, skip],
                  J_mag[skip, skip], cmap='plasma', scale=None, alpha=0.8)
        
        ax2.set_xlabel('X (nm)')
        ax2.set_ylabel('Y (nm)')
        ax2.set_title('Current Density Vectors')
        ax2.set_aspect('equal')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def visualize_device_cross_section(self, device_type: str,
                                     cross_section: str = 'xz',
                                     position: float = 0.5,
                                     show_fields: bool = True,
                                     title: Optional[str] = None,
                                     save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize device cross-section with fields"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        title = title or f'{device_type.upper()} Cross-Section ({cross_section.upper()})'
        
        if device_type.lower() == 'mtj':
            self._draw_mtj_cross_section(ax, cross_section, show_fields)
        elif device_type.lower() == 'spin_valve':
            self._draw_spin_valve_cross_section(ax, cross_section, show_fields)
        elif device_type.lower() == 'asl':
            self._draw_asl_cross_section(ax, cross_section, show_fields)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def _draw_3d_box(self, ax, x, y, z, dx, dy, dz, color, alpha=1.0):
        """Draw a 3D box"""
        
        # Define vertices
        vertices = [
            [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],  # Bottom
            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]  # Top
        ]
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
        ]
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create and add 3D polygons
        collection = Poly3DCollection(faces, facecolor=color, alpha=alpha,
                                     edgecolor='black', linewidth=0.5)
        ax.add_collection3d(collection)
    
    def _add_field_lines(self, ax, field_function, x_range, y_range, z_range):
        """Add magnetic field lines to 3D plot"""
        
        # Start points for field lines
        start_points = [
            [x_range[0], y_range[0], z_range[0]],
            [x_range[1], y_range[0], z_range[0]],
            [x_range[0], y_range[1], z_range[0]],
            [x_range[1], y_range[1], z_range[0]]
        ]
        
        for start in start_points:
            line_x, line_y, line_z = self._trace_field_line(field_function, start,
                                                           x_range, y_range, z_range)
            ax.plot(line_x, line_y, line_z, color='red', linewidth=2, alpha=0.8)
    
    def _trace_field_line(self, field_function, start_point, x_range, y_range, z_range,
                         max_steps=100, step_size=0.01):
        """Trace a single field line"""
        
        x, y, z = start_point
        line_x, line_y, line_z = [x], [y], [z]
        
        for _ in range(max_steps):
            # Get field at current point
            hx, hy, hz = field_function(np.array([x]), np.array([y]), np.array([z]))
            hx, hy, hz = hx[0], hy[0], hz[0]
            
            # Normalize
            h_mag = np.sqrt(hx**2 + hy**2 + hz**2)
            if h_mag == 0:
                break
            
            hx, hy, hz = hx/h_mag, hy/h_mag, hz/h_mag
            
            # Step forward
            x += hx * step_size
            y += hy * step_size
            z += hz * step_size
            
            # Check bounds
            if (x < x_range[0] or x > x_range[1] or
                y < y_range[0] or y > y_range[1] or
                z < z_range[0] or z > z_range[1]):
                break
            
            line_x.append(x)
            line_y.append(y)
            line_z.append(z)
        
        return line_x, line_y, line_z
    
    def _draw_mtj_cross_section(self, ax, cross_section, show_fields):
        """Draw MTJ cross-section"""
        
        # Layer structure
        y_positions = [0, 1, 1.2, 2.2]  # Bottom contact, bottom FM, barrier, top FM
        colors = [self.colors.SEQUENCE[3], self.colors.BERKELEY_BLUE,
                 self.colors.CALIFORNIA_GOLD, self.colors.GREEN_DARK]
        labels = ['Contact', 'Free Layer', 'MgO', 'Reference Layer']
        
        for i in range(len(y_positions)-1):
            rect = patches.Rectangle((0, y_positions[i]), 10, 
                                   y_positions[i+1] - y_positions[i],
                                   facecolor=colors[i], alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            
            # Add label
            ax.text(5, (y_positions[i] + y_positions[i+1])/2, labels[i],
                   ha='center', va='center', fontsize=12, weight='bold')
        
        # Add top contact
        rect = patches.Rectangle((0, y_positions[-1]), 10, 1,
                               facecolor=self.colors.SEQUENCE[3], alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(5, y_positions[-1] + 0.5, 'Contact', ha='center', va='center',
               fontsize=12, weight='bold')
        
        if show_fields:
            # Magnetization arrows
            ax.arrow(2, 0.6, 2, 0, head_width=0.1, head_length=0.2,
                    fc='red', ec='red', linewidth=3)
            ax.text(3, 0.8, 'M₁', fontsize=14, color='red', weight='bold')
            
            ax.arrow(8, 1.7, -2, 0, head_width=0.1, head_length=0.2,
                    fc='red', ec='red', linewidth=3)
            ax.text(7, 1.9, 'M₂', fontsize=14, color='red', weight='bold')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3.5)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Z (nm)')
    
    def _draw_spin_valve_cross_section(self, ax, cross_section, show_fields):
        """Draw spin valve cross-section"""
        
        # Device structure
        layers = [
            (0, 0.5, self.colors.SEQUENCE[3], 'Bottom Contact'),
            (0.5, 1.5, self.colors.BERKELEY_BLUE, 'Bottom FM'),
            (1.5, 3.5, self.colors.CALIFORNIA_GOLD, 'Spacer'),
            (3.5, 4.5, self.colors.GREEN_DARK, 'Top FM'),
            (4.5, 5, self.colors.SEQUENCE[3], 'Top Contact')
        ]
        
        for y_start, y_end, color, label in layers:
            rect = patches.Rectangle((0, y_start), 20, y_end - y_start,
                                   facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            ax.text(10, (y_start + y_end)/2, label, ha='center', va='center',
                   fontsize=12, weight='bold')
        
        if show_fields:
            # Current flow
            for y in np.linspace(0.5, 4.5, 5):
                ax.arrow(0.5, y, 19, 0, head_width=0.1, head_length=0.5,
                        fc='blue', ec='blue', alpha=0.6, linewidth=2)
            
            ax.text(1, 2.5, 'Current Flow', fontsize=12, color='blue', weight='bold')
            
            # Magnetizations
            ax.arrow(15, 1, -3, 0, head_width=0.2, head_length=0.5,
                    fc='red', ec='red', linewidth=4)
            ax.text(13, 1.3, 'M₁', fontsize=14, color='red', weight='bold')
            
            ax.arrow(5, 4, 3, 0, head_width=0.2, head_length=0.5,
                    fc='red', ec='red', linewidth=4)
            ax.text(7, 4.3, 'M₂', fontsize=14, color='red', weight='bold')
        
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 5.5)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Z (nm)')
    
    def _draw_asl_cross_section(self, ax, cross_section, show_fields):
        """Draw ASL cross-section"""
        
        # Channel
        channel = patches.Rectangle((2, 1), 16, 2, facecolor=self.colors.CALIFORNIA_GOLD,
                                  alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(channel)
        ax.text(10, 2, 'Spin Channel', ha='center', va='center',
               fontsize=14, weight='bold')
        
        # Input magnet
        input_mag = patches.Ellipse((1, 2), 1.5, 1, facecolor=self.colors.BERKELEY_BLUE,
                                  alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(input_mag)
        ax.text(1, 2, 'Input', ha='center', va='center', fontsize=12,
               color='white', weight='bold')
        
        # Output magnet
        output_mag = patches.Ellipse((19, 2), 1.5, 1, facecolor=self.colors.GREEN_DARK,
                                   alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(output_mag)
        ax.text(19, 2, 'Output', ha='center', va='center', fontsize=12,
               color='white', weight='bold')
        
        if show_fields:
            # Spin current
            ax.arrow(2.5, 2, 14, 0, head_width=0.3, head_length=0.5,
                    fc='red', ec='red', linewidth=4, alpha=0.8)
            ax.text(10, 2.8, 'Spin Current', ha='center', va='center',
                   fontsize=12, color='red', weight='bold')
            
            # Magnetization directions
            ax.arrow(0.3, 2, 0.4, 0, head_width=0.2, head_length=0.1,
                    fc='purple', ec='purple', linewidth=3)
            ax.arrow(19.3, 2, 0.4, 0, head_width=0.2, head_length=0.1,
                    fc='purple', ec='purple', linewidth=3)
        
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 4)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
    
    def create_device_animation(self, device_type: str, time_data: np.ndarray,
                              magnetization_data: np.ndarray,
                              save_path: Optional[str] = None) -> animation.FuncAnimation:
        """Create animated visualization of device dynamics"""
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            
            # Get current magnetization
            m_current = magnetization_data[:, frame]
            
            if device_type.lower() == 'mtj':
                # Draw MTJ with current magnetization
                self._animate_mtj_frame(ax, m_current, frame, len(time_data))
            
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_zlim(0, 5)
            ax.set_title(f'{device_type.upper()} Dynamics - t = {time_data[frame]:.2e} s')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(time_data),
                                     interval=50, blit=False)
        
        if save_path:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=20, metadata=dict(artist='SpinCirc'), bitrate=1800)
            anim.save(save_path, writer=writer)
            logger.info(f"Animation saved to {save_path}")
        
        return anim
    
    def _animate_mtj_frame(self, ax, magnetization, frame, total_frames):
        """Draw single frame of MTJ animation"""
        
        # Device structure (simplified)
        self._draw_3d_box(ax, 2, 2, 1, 6, 6, 0.5, self.colors.BERKELEY_BLUE, 0.7)
        self._draw_3d_box(ax, 2, 2, 1.7, 6, 6, 0.5, self.colors.GREEN_DARK, 0.7)
        
        # Magnetization arrow for free layer
        mx, my, mz = magnetization[0], magnetization[1], magnetization[2]
        
        # Scale arrow
        scale = 2
        ax.quiver(5, 5, 1.25, mx*scale, my*scale, mz*scale,
                 color='red', arrow_length_ratio=0.2, linewidth=4)
        
        # Fixed reference layer magnetization
        ax.quiver(5, 5, 1.95, 2, 0, 0, color='blue',
                 arrow_length_ratio=0.2, linewidth=4)
        
        # Progress indicator
        progress = frame / total_frames
        ax.text2D(0.02, 0.95, f'Progress: {progress:.1%}', transform=ax.transAxes,
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))