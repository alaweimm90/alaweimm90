#!/usr/bin/env python3
"""
Animation Tools for SpinCirc

Create animated visualizations of spintronic device dynamics including
magnetization precession, current flow, and parameter sweeps.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AnimationTools:
    """Tools for creating animated visualizations of spintronic dynamics"""
    
    def __init__(self, berkeley_colors: bool = True, fps: int = 30):
        """
        Initialize animation tools
        
        Args:
            berkeley_colors: Use Berkeley color scheme
            fps: Frames per second for animations
        """
        self.fps = fps
        self.berkeley_colors = berkeley_colors
        
        if berkeley_colors:
            self.colors = {
                'berkeley_blue': '#003262',
                'california_gold': '#FDB515',
                'green_dark': '#00553A',
                'rose_dark': '#77072F',
                'purple_dark': '#431170',
                'sequence': ['#003262', '#FDB515', '#00553A', '#77072F', '#431170']
            }
        else:
            self.colors = {
                'berkeley_blue': 'blue',
                'california_gold': 'orange',
                'green_dark': 'green',
                'rose_dark': 'red',
                'purple_dark': 'purple',
                'sequence': ['blue', 'orange', 'green', 'red', 'purple']
            }
    
    def animate_magnetization_dynamics(self, time: np.ndarray, 
                                     magnetization: np.ndarray,
                                     title: str = 'Magnetization Dynamics',
                                     save_path: Optional[str] = None,
                                     show_trajectory: bool = True) -> animation.FuncAnimation:
        """
        Animate magnetization vector dynamics on unit sphere
        
        Args:
            time: Time array
            magnetization: Magnetization data (3 x N)
            title: Animation title
            save_path: Path to save animation
            show_trajectory: Show trajectory trail
            
        Returns:
            Matplotlib animation object
        """
        
        fig = plt.figure(figsize=(12, 10))
        
        # Main 3D plot
        ax_3d = fig.add_subplot(221, projection='3d')
        
        # Component plots
        ax_mx = fig.add_subplot(222)
        ax_my = fig.add_subplot(223)
        ax_mz = fig.add_subplot(224)
        
        # Extract components
        mx, my, mz = magnetization[0, :], magnetization[1, :], magnetization[2, :]
        
        # Set up 3D plot
        self._setup_unit_sphere(ax_3d)
        
        # Initialize plot elements
        vector_line, = ax_3d.plot([], [], [], color=self.colors['berkeley_blue'], 
                                 linewidth=4, alpha=0.8)
        vector_head = ax_3d.scatter([], [], [], color=self.colors['california_gold'],
                                   s=100, alpha=0.9)
        
        if show_trajectory:
            trajectory_line, = ax_3d.plot([], [], [], color=self.colors['green_dark'],
                                        linewidth=2, alpha=0.6)
        
        # Component plots setup
        component_plots = [
            (ax_mx, mx, 'mx', self.colors['berkeley_blue']),
            (ax_my, my, 'my', self.colors['california_gold']),
            (ax_mz, mz, 'mz', self.colors['green_dark'])
        ]
        
        component_lines = []
        current_points = []
        
        for ax, data, label, color in component_plots:
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            
            # Full trajectory (light)
            ax.plot(time, data, color=color, alpha=0.3, linewidth=1)
            
            # Current trajectory (bold)
            line, = ax.plot([], [], color=color, linewidth=3, alpha=0.8)
            point, = ax.plot([], [], 'o', color=color, markersize=8, alpha=0.9)
            
            component_lines.append(line)
            current_points.append(point)
        
        def animate(frame):
            """Animation function"""
            current_time = time[frame]
            current_mx, current_my, current_mz = mx[frame], my[frame], mz[frame]
            
            # Update 3D vector
            vector_line.set_data([0, current_mx], [0, current_my])
            vector_line.set_3d_properties([0, current_mz])
            
            vector_head._offsets3d = ([current_mx], [current_my], [current_mz])
            
            # Update trajectory
            if show_trajectory and frame > 0:
                trajectory_line.set_data(mx[:frame+1], my[:frame+1])
                trajectory_line.set_3d_properties(mz[:frame+1])
            
            # Update component plots
            for i, (line, point) in enumerate(zip(component_lines, current_points)):
                data = [mx, my, mz][i]
                
                # Current trajectory up to this frame
                line.set_data(time[:frame+1], data[:frame+1])
                
                # Current point
                point.set_data([current_time], [data[frame]])
            
            # Update title with current time
            fig.suptitle(f'{title} - t = {current_time:.2e} s', fontsize=16, fontweight='bold')
            
            return [vector_line, vector_head] + component_lines + current_points
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(time),
                                     interval=1000/self.fps, blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            self._save_animation(anim, save_path)
        
        return anim
    
    def animate_current_flow(self, geometry: Dict[str, float],
                           current_data: np.ndarray,
                           time: np.ndarray,
                           title: str = 'Current Flow Animation',
                           save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Animate current flow through device
        
        Args:
            geometry: Device geometry parameters
            current_data: Current density data over time
            time: Time array
            title: Animation title
            save_path: Save path for animation
            
        Returns:
            Animation object
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract geometry
        length = geometry.get('length', 200e-9) * 1e9  # Convert to nm
        width = geometry.get('width', 100e-9) * 1e9
        
        # Create spatial grid
        x = np.linspace(0, length, current_data.shape[1])
        y = np.linspace(0, width, current_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Set up plots
        ax1.set_xlim(0, length)
        ax1.set_ylim(0, width)
        ax1.set_xlabel('Length (nm)')
        ax1.set_ylabel('Width (nm)')
        ax1.set_title('Current Density Magnitude')
        ax1.set_aspect('equal')
        
        ax2.set_xlim(0, length)
        ax2.set_ylim(0, width)
        ax2.set_xlabel('Length (nm)')
        ax2.set_ylabel('Width (nm)')
        ax2.set_title('Current Streamlines')
        ax2.set_aspect('equal')
        
        # Initialize plots
        current_magnitude = np.abs(current_data[0])
        
        # Contour plot
        contour = ax1.contourf(X, Y, current_magnitude, levels=20,
                              cmap='viridis', alpha=0.8)
        cbar = plt.colorbar(contour, ax=ax1)
        cbar.set_label('Current Density (A/m²)')
        
        # Streamline plot (simplified - using random field for demo)
        def get_current_field(frame):
            # This would normally extract Jx, Jy from current_data
            # For demo, create reasonable-looking field
            Jx = np.ones_like(X) * (1 + 0.1 * np.sin(2*np.pi*frame/len(time)))
            Jy = np.zeros_like(Y)
            return Jx, Jy
        
        # Initial streamlines
        Jx_init, Jy_init = get_current_field(0)
        streamlines = ax2.streamplot(X, Y, Jx_init, Jy_init, color='white',
                                    linewidth=2, density=2, alpha=0.8)
        
        def animate(frame):
            """Animation function"""
            
            # Clear previous streamlines
            ax2.clear()
            ax2.set_xlim(0, length)
            ax2.set_ylim(0, width)
            ax2.set_xlabel('Length (nm)')
            ax2.set_ylabel('Width (nm)')
            ax2.set_title('Current Streamlines')
            ax2.set_aspect('equal')
            
            # Update current magnitude
            if frame < current_data.shape[2]:
                current_mag_frame = np.abs(current_data[frame])
            else:
                current_mag_frame = current_magnitude
            
            # Update contour plot
            ax1.clear()
            ax1.set_xlim(0, length)
            ax1.set_ylim(0, width)
            ax1.set_xlabel('Length (nm)')
            ax1.set_ylabel('Width (nm)')
            ax1.set_title('Current Density Magnitude')
            ax1.set_aspect('equal')
            
            contour_new = ax1.contourf(X, Y, current_mag_frame, levels=20,
                                     cmap='viridis', alpha=0.8)
            
            # Update streamlines
            Jx_frame, Jy_frame = get_current_field(frame)
            ax2.streamplot(X, Y, Jx_frame, Jy_frame, color=self.colors['berkeley_blue'],
                          linewidth=2, density=2, alpha=0.8)
            
            # Add device outline
            device_outline = plt.Rectangle((0, 0), length, width, fill=False,
                                         edgecolor='black', linewidth=3)
            ax1.add_patch(device_outline)
            ax2.add_patch(plt.Rectangle((0, 0), length, width, fill=False,
                                       edgecolor='black', linewidth=3))
            
            # Update title with time
            current_time = time[frame % len(time)]
            fig.suptitle(f'{title} - t = {current_time:.2e} s', fontsize=16, fontweight='bold')
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=max(len(time), current_data.shape[2]),
                                     interval=1000/self.fps, blit=False, repeat=True)
        
        if save_path:
            self._save_animation(anim, save_path)
        
        return anim
    
    def animate_parameter_sweep(self, sweep_parameter: str,
                              sweep_values: np.ndarray,
                              results: Dict[str, np.ndarray],
                              title: str = 'Parameter Sweep Animation',
                              save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Animate parameter sweep results
        
        Args:
            sweep_parameter: Name of swept parameter
            sweep_values: Array of parameter values
            results: Dictionary of result arrays
            title: Animation title
            save_path: Save path
            
        Returns:
            Animation object
        """
        
        n_outputs = len(results)
        fig, axes = plt.subplots(1, min(n_outputs, 3), figsize=(15, 5))
        
        if n_outputs == 1:
            axes = [axes]
        elif n_outputs > 3:
            axes = axes[:3]  # Limit to 3 plots
        
        output_names = list(results.keys())[:len(axes)]
        colors = self.colors['sequence'][:len(axes)]
        
        # Set up plots
        for i, (ax, output_name, color) in enumerate(zip(axes, output_names, colors)):
            output_data = results[output_name]
            
            ax.set_xlim(sweep_values[0], sweep_values[-1])
            ax.set_ylim(np.min(output_data) * 0.9, np.max(output_data) * 1.1)
            ax.set_xlabel(sweep_parameter)
            ax.set_ylabel(output_name)
            ax.grid(True, alpha=0.3)
            
            # Full curve (light)
            ax.plot(sweep_values, output_data, color=color, alpha=0.3, linewidth=2)
        
        # Initialize animated elements
        animated_lines = []
        current_points = []
        
        for i, (ax, color) in enumerate(zip(axes, colors)):
            line, = ax.plot([], [], color=color, linewidth=4, alpha=0.8)
            point, = ax.plot([], [], 'o', color=color, markersize=10, alpha=0.9)
            
            animated_lines.append(line)
            current_points.append(point)
        
        def animate(frame):
            """Animation function"""
            current_param = sweep_values[frame]
            
            # Update each plot
            for i, (line, point, output_name) in enumerate(zip(animated_lines, current_points, output_names)):
                output_data = results[output_name]
                
                # Current sweep up to this frame
                line.set_data(sweep_values[:frame+1], output_data[:frame+1])
                
                # Current point
                point.set_data([current_param], [output_data[frame]])
            
            # Update title
            fig.suptitle(f'{title} - {sweep_parameter} = {current_param:.3f}',
                        fontsize=16, fontweight='bold')
            
            return animated_lines + current_points
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(sweep_values),
                                     interval=1000/self.fps, blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            self._save_animation(anim, save_path)
        
        return anim
    
    def create_plotly_animation(self, data: Dict[str, Any],
                              animation_type: str = 'magnetization',
                              title: str = 'SpinCirc Animation') -> go.Figure:
        """
        Create interactive Plotly animation
        
        Args:
            data: Animation data
            animation_type: Type of animation
            title: Animation title
            
        Returns:
            Plotly figure with animation
        """
        
        if animation_type == 'magnetization':
            return self._create_magnetization_plotly_animation(data, title)
        elif animation_type == 'sweep':
            return self._create_sweep_plotly_animation(data, title)
        else:
            raise ValueError(f"Unknown animation type: {animation_type}")
    
    def _create_magnetization_plotly_animation(self, data: Dict[str, Any],
                                             title: str) -> go.Figure:
        """Create Plotly magnetization animation"""
        
        time = data['time']
        magnetization = data['magnetization']
        
        mx, my, mz = magnetization[0, :], magnetization[1, :], magnetization[2, :]
        
        # Create frames
        frames = []
        
        for i in range(len(time)):
            frame_data = [
                go.Scatter3d(
                    x=[0, mx[i]], y=[0, my[i]], z=[0, mz[i]],
                    mode='lines+markers',
                    line=dict(color=self.colors['berkeley_blue'], width=8),
                    marker=dict(size=[5, 10], color=[self.colors['berkeley_blue'], self.colors['california_gold']]),
                    name='Magnetization Vector'
                ),
                go.Scatter3d(
                    x=mx[:i+1], y=my[:i+1], z=mz[:i+1],
                    mode='lines',
                    line=dict(color=self.colors['green_dark'], width=4),
                    name='Trajectory'
                )
            ]
            
            frames.append(go.Frame(data=frame_data, name=f'frame_{i}'))
        
        # Create unit sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Initial data
        initial_data = [
            go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
                showscale=False,
                name='Unit Sphere'
            ),
            go.Scatter3d(
                x=[0, mx[0]], y=[0, my[0]], z=[0, mz[0]],
                mode='lines+markers',
                line=dict(color=self.colors['berkeley_blue'], width=8),
                marker=dict(size=[5, 10], color=[self.colors['berkeley_blue'], self.colors['california_gold']]),
                name='Magnetization Vector'
            )
        ]
        
        # Create figure
        fig = go.Figure(data=initial_data, frames=frames)
        
        # Animation controls
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='mx',
                yaxis_title='my',
                zaxis_title='mz',
                aspectmode='cube'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                       'transition': {'duration': 50}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                         'mode': 'immediate', 'transition': {'duration': 0}}]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f'frame_{i}'], {'frame': {'duration': 0, 'redraw': True},
                                                'mode': 'immediate'}],
                        'label': f'{time[i]:.2e}',
                        'method': 'animate'
                    } for i in range(len(time))
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left'
            }]
        )
        
        return fig
    
    def _create_sweep_plotly_animation(self, data: Dict[str, Any],
                                     title: str) -> go.Figure:
        """Create Plotly parameter sweep animation"""
        
        sweep_values = data['sweep_values']
        results = data['results']
        sweep_parameter = data.get('sweep_parameter', 'Parameter')
        
        output_names = list(results.keys())
        
        # Create subplots
        fig = make_subplots(rows=1, cols=len(output_names),
                           subplot_titles=output_names)
        
        # Create frames
        frames = []
        
        for i in range(len(sweep_values)):
            frame_data = []
            
            for j, output_name in enumerate(output_names):
                output_data = results[output_name]
                
                trace = go.Scatter(
                    x=sweep_values[:i+1],
                    y=output_data[:i+1],
                    mode='lines+markers',
                    line=dict(color=self.colors['sequence'][j % len(self.colors['sequence'])]),
                    name=output_name
                )
                
                frame_data.append(trace)
            
            frames.append(go.Frame(data=frame_data, name=f'frame_{i}'))
        
        # Initial data
        initial_data = []
        for j, output_name in enumerate(output_names):
            trace = go.Scatter(
                x=[sweep_values[0]],
                y=[results[output_name][0]],
                mode='lines+markers',
                line=dict(color=self.colors['sequence'][j % len(self.colors['sequence'])]),
                name=output_name
            )
            initial_data.append(trace)
        
        fig = go.Figure(data=initial_data, frames=frames)
        
        # Update layout with animation controls
        fig.update_layout(
            title=title,
            xaxis_title=sweep_parameter,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                    }
                ]
            }]
        )
        
        return fig
    
    def _setup_unit_sphere(self, ax):
        """Set up unit sphere visualization"""
        
        # Create unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightgray')
        
        # Coordinate axes
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='red', alpha=0.6, arrow_length_ratio=0.05)
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='green', alpha=0.6, arrow_length_ratio=0.05)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='blue', alpha=0.6, arrow_length_ratio=0.05)
        
        ax.text(1.3, 0, 0, 'X', fontsize=12)
        ax.text(0, 1.3, 0, 'Y', fontsize=12)
        ax.text(0, 0, 1.3, 'Z', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel('mx')
        ax.set_ylabel('my')
        ax.set_zlabel('mz')
        ax.set_title('Magnetization Vector')
    
    def _save_animation(self, anim: animation.FuncAnimation, filepath: str):
        """Save matplotlib animation"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        if filepath.suffix.lower() in ['.mp4', '.avi']:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=self.fps, metadata=dict(artist='SpinCirc'), bitrate=1800)
        elif filepath.suffix.lower() == '.gif':
            Writer = animation.writers['pillow']
            writer = Writer(fps=self.fps)
        else:
            # Default to MP4
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=self.fps, metadata=dict(artist='SpinCirc'), bitrate=1800)
            filepath = filepath.with_suffix('.mp4')
        
        try:
            anim.save(str(filepath), writer=writer)
            logger.info(f"Animation saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            # Try alternative format
            try:
                filepath_gif = filepath.with_suffix('.gif')
                Writer = animation.writers['pillow']
                writer = Writer(fps=min(self.fps, 10))  # Reduce fps for GIF
                anim.save(str(filepath_gif), writer=writer)
                logger.info(f"Animation saved as GIF to {filepath_gif}")
            except Exception as e2:
                logger.error(f"Failed to save as GIF: {e2}")
    
    def create_comparison_animation(self, data_sets: Dict[str, Dict[str, Any]],
                                  animation_type: str = 'magnetization',
                                  title: str = 'Comparison Animation',
                                  save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Create side-by-side comparison animation
        
        Args:
            data_sets: Dictionary of data sets to compare
            animation_type: Type of animation
            title: Animation title
            save_path: Save path
            
        Returns:
            Animation object
        """
        
        n_datasets = len(data_sets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 6),
                               subplot_kw={'projection': '3d'})
        
        if n_datasets == 1:
            axes = [axes]
        
        dataset_names = list(data_sets.keys())
        
        # Set up each subplot
        animated_elements = []
        
        for i, (ax, dataset_name) in enumerate(zip(axes, dataset_names)):
            data = data_sets[dataset_name]
            
            if animation_type == 'magnetization':
                self._setup_unit_sphere(ax)
                ax.set_title(dataset_name)
                
                # Initialize animated elements for this dataset
                time = data['time']
                magnetization = data['magnetization']
                mx, my, mz = magnetization[0, :], magnetization[1, :], magnetization[2, :]
                
                vector_line, = ax.plot([], [], [], color=self.colors['berkeley_blue'],
                                     linewidth=4, alpha=0.8)
                vector_head = ax.scatter([], [], [], color=self.colors['california_gold'],
                                       s=100, alpha=0.9)
                trajectory_line, = ax.plot([], [], [], color=self.colors['green_dark'],
                                         linewidth=2, alpha=0.6)
                
                animated_elements.append((vector_line, vector_head, trajectory_line, mx, my, mz))
        
        def animate(frame):
            """Animation function"""
            
            for i, (vector_line, vector_head, trajectory_line, mx, my, mz) in enumerate(animated_elements):
                
                if frame < len(mx):
                    current_mx, current_my, current_mz = mx[frame], my[frame], mz[frame]
                    
                    # Update vector
                    vector_line.set_data([0, current_mx], [0, current_my])
                    vector_line.set_3d_properties([0, current_mz])
                    
                    vector_head._offsets3d = ([current_mx], [current_my], [current_mz])
                    
                    # Update trajectory
                    if frame > 0:
                        trajectory_line.set_data(mx[:frame+1], my[:frame+1])
                        trajectory_line.set_3d_properties(mz[:frame+1])
            
            # Get current time from first dataset
            first_data = list(data_sets.values())[0]
            current_time = first_data['time'][frame % len(first_data['time'])]
            fig.suptitle(f'{title} - t = {current_time:.2e} s', fontsize=16, fontweight='bold')
            
            return [elem for sublist in animated_elements for elem in sublist[:2]]
        
        # Determine total frames
        max_frames = max(len(data['time']) for data in data_sets.values())
        
        anim = animation.FuncAnimation(fig, animate, frames=max_frames,
                                     interval=1000/self.fps, blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            self._save_animation(anim, save_path)
        
        return anim