#!/usr/bin/env python3
"""
Interactive Plotting for SpinCirc

Interactive Plotly-based visualizations for exploring spintronic device data
with widgets, sliders, and real-time parameter adjustment.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import ipywidgets as widgets
from IPython.display import display
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging

logger = logging.getLogger(__name__)

class InteractivePlotter:
    """Interactive plotting with Plotly and IPython widgets"""
    
    def __init__(self, berkeley_theme: bool = True):
        """
        Initialize interactive plotter
        
        Args:
            berkeley_theme: Use Berkeley color scheme
        """
        self.berkeley_theme = berkeley_theme
        
        if berkeley_theme:
            self.colors = {
                'berkeley_blue': '#003262',
                'california_gold': '#FDB515',
                'green_dark': '#00553A',
                'rose_dark': '#77072F',
                'purple_dark': '#431170',
                'gray_dark': '#666666'
            }
            
            # Create Berkeley color scale
            self.berkeley_colorscale = [
                [0.0, self.colors['berkeley_blue']],
                [0.5, '#FFFFFF'],
                [1.0, self.colors['california_gold']]
            ]
        else:
            self.colors = px.colors.qualitative.Set1
            self.berkeley_colorscale = px.colors.sequential.Viridis
    
    def interactive_iv_explorer(self, voltage_range: Tuple[float, float] = (-1, 1),
                               device_params: Dict[str, Tuple[float, float, float]] = None,
                               iv_function: Callable = None) -> widgets.VBox:
        """
        Create interactive I-V curve explorer
        
        Args:
            voltage_range: Voltage range for sweep
            device_params: Parameter ranges {name: (min, max, default)}
            iv_function: Function that calculates I-V given parameters
            
        Returns:
            IPython widget container
        """
        
        # Default parameters if not provided
        if device_params is None:
            device_params = {
                'TMR': (50, 500, 200),
                'RA_product': (1, 100, 10),
                'thickness': (0.8, 3.0, 1.2)
            }
        
        # Default I-V function
        if iv_function is None:
            def default_iv_function(V, params):
                tmr = params['TMR']
                ra = params['RA_product']
                t = params['thickness']
                
                # Simplified MTJ I-V characteristic
                R_p = ra * 1e-6  # Convert to Ohm
                R_ap = R_p * (1 + tmr/100)
                
                # Bias-dependent TMR
                tmr_bias = tmr / (1 + (V / 0.5)**2)
                R_bias = R_p * (1 + tmr_bias/100)
                
                I = V / R_bias
                return I
            
            iv_function = default_iv_function
        
        # Create parameter sliders
        param_widgets = {}
        for param, (min_val, max_val, default) in device_params.items():
            param_widgets[param] = widgets.FloatSlider(
                value=default,
                min=min_val,
                max=max_val,
                step=(max_val - min_val) / 100,
                description=param,
                continuous_update=True
            )
        
        # Output widget for plot
        output = widgets.Output()
        
        def update_plot(**kwargs):
            """Update I-V plot based on slider values"""
            with output:
                output.clear_output(wait=True)
                
                # Get current parameter values
                params = {name: widget.value for name, widget in param_widgets.items()}
                
                # Calculate I-V curves
                V = np.linspace(voltage_range[0], voltage_range[1], 200)
                I = np.array([iv_function(v, params) for v in V])
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=V, y=I * 1e6,  # Convert to μA
                    mode='lines',
                    line=dict(color=self.colors['berkeley_blue'], width=3),
                    name='I-V Characteristic'
                ))
                
                fig.update_layout(
                    title='Interactive I-V Characteristic Explorer',
                    xaxis_title='Voltage (V)',
                    yaxis_title='Current (μA)',
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )
                
                # Add parameter annotations
                param_text = '<br>'.join([f'{name}: {value:.2f}' 
                                        for name, value in params.items()])
                
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=param_text,
                    showarrow=False,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='black',
                    borderwidth=1
                )
                
                fig.show()
        
        # Connect sliders to update function
        interactive_plot = widgets.interactive(update_plot, **param_widgets)
        
        # Arrange widgets
        param_box = widgets.VBox(list(param_widgets.values()))
        
        container = widgets.VBox([
            widgets.HTML('<h3>Interactive I-V Explorer</h3>'),
            param_box,
            output
        ])
        
        # Initial plot
        update_plot(**{name: widget.value for name, widget in param_widgets.items()})
        
        return container
    
    def interactive_magnetization_sphere(self, time_data: np.ndarray,
                                       magnetization_data: np.ndarray,
                                       title: str = 'Interactive Magnetization Dynamics') -> go.Figure:
        """
        Create interactive 3D magnetization trajectory on unit sphere
        
        Args:
            time_data: Time array
            magnetization_data: Magnetization trajectory (3 x N)
            title: Plot title
            
        Returns:
            Plotly figure with interactive controls
        """
        
        # Extract components
        mx, my, mz = magnetization_data[0, :], magnetization_data[1, :], magnetization_data[2, :]
        
        # Create unit sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Create figure
        fig = go.Figure()
        
        # Add unit sphere
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, 'rgba(200,200,200,0.2)'], [1, 'rgba(200,200,200,0.2)']],
            showscale=False,
            name='Unit Sphere'
        ))
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=mx, y=my, z=mz,
            mode='lines+markers',
            line=dict(color=time_data, colorscale='Viridis', width=4),
            marker=dict(size=3, color=time_data, colorscale='Viridis'),
            name='Trajectory',
            hovertemplate='<b>Time: %{marker.color:.2e} s</b><br>' +
                         'mx: %{x:.3f}<br>' +
                         'my: %{y:.3f}<br>' +
                         'mz: %{z:.3f}<extra></extra>'
        ))
        
        # Add start and end points
        fig.add_trace(go.Scatter3d(
            x=[mx[0]], y=[my[0]], z=[mz[0]],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Start',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[mx[-1]], y=[my[-1]], z=[mz[-1]],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='End',
            showlegend=True
        ))
        
        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='mx',
                yaxis_title='my',
                zaxis_title='mz',
                aspectmode='cube'
            ),
            height=600
        )
        
        return fig
    
    def interactive_parameter_sweep(self, sweep_function: Callable,
                                  sweep_params: Dict[str, Tuple[float, float, int]],
                                  fixed_params: Dict[str, float] = None,
                                  output_names: List[str] = None) -> widgets.VBox:
        """
        Create interactive parameter sweep visualization
        
        Args:
            sweep_function: Function to evaluate
            sweep_params: Sweep parameters {name: (min, max, n_points)}
            fixed_params: Fixed parameters
            output_names: Names of output quantities
            
        Returns:
            Interactive widget container
        """
        
        fixed_params = fixed_params or {}
        output_names = output_names or ['Output']
        
        # Create sweep selector
        param_names = list(sweep_params.keys())
        sweep_selector = widgets.Dropdown(
            options=param_names,
            value=param_names[0],
            description='Sweep Parameter:'
        )
        
        # Create output selector
        output_selector = widgets.Dropdown(
            options=output_names,
            value=output_names[0],
            description='Output:'
        )
        
        # Fixed parameter sliders
        fixed_widgets = {}
        if fixed_params:
            for param, default_value in fixed_params.items():
                # Estimate reasonable bounds
                min_val = default_value * 0.1
                max_val = default_value * 10
                
                fixed_widgets[param] = widgets.FloatSlider(
                    value=default_value,
                    min=min_val,
                    max=max_val,
                    step=(max_val - min_val) / 100,
                    description=param,
                    continuous_update=True
                )
        
        # Output
        output = widgets.Output()
        
        def update_sweep_plot(sweep_param, output_param, **fixed_values):
            """Update parameter sweep plot"""
            with output:
                output.clear_output(wait=True)
                
                # Create sweep array
                min_val, max_val, n_points = sweep_params[sweep_param]
                sweep_array = np.linspace(min_val, max_val, n_points)
                
                # Calculate outputs
                results = []
                for value in sweep_array:
                    params = fixed_values.copy()
                    params[sweep_param] = value
                    
                    try:
                        result = sweep_function(params)
                        if isinstance(result, dict):
                            results.append(result[output_param])
                        else:
                            results.append(result)
                    except Exception as e:
                        logger.warning(f"Sweep calculation failed: {e}")
                        results.append(np.nan)
                
                results = np.array(results)
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=sweep_array, y=results,
                    mode='lines+markers',
                    line=dict(color=self.colors['berkeley_blue'], width=3),
                    marker=dict(size=6),
                    name=f'{output_param} vs {sweep_param}'
                ))
                
                fig.update_layout(
                    title=f'Parameter Sweep: {output_param} vs {sweep_param}',
                    xaxis_title=sweep_param,
                    yaxis_title=output_param,
                    template='plotly_white',
                    height=500
                )
                
                fig.show()
                
                # Show statistics
                valid_results = results[~np.isnan(results)]
                if len(valid_results) > 0:
                    stats_html = f"""
                    <div style='padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc;'>
                    <b>Statistics:</b><br>
                    Mean: {np.mean(valid_results):.3e}<br>
                    Std: {np.std(valid_results):.3e}<br>
                    Min: {np.min(valid_results):.3e}<br>
                    Max: {np.max(valid_results):.3e}
                    </div>
                    """
                    display(widgets.HTML(stats_html))
        
        # Create interactive widget
        widget_dict = {'sweep_param': sweep_selector, 'output_param': output_selector}
        widget_dict.update(fixed_widgets)
        
        interactive_plot = widgets.interactive(update_sweep_plot, **widget_dict)
        
        # Container
        controls = widgets.HBox([sweep_selector, output_selector])
        fixed_controls = widgets.VBox(list(fixed_widgets.values()))
        
        container = widgets.VBox([
            widgets.HTML('<h3>Interactive Parameter Sweep</h3>'),
            controls,
            fixed_controls,
            output
        ])
        
        return container
    
    def interactive_sensitivity_analysis(self, sensitivity_data: Dict[str, Dict[str, float]],
                                       title: str = 'Interactive Sensitivity Analysis') -> go.Figure:
        """
        Create interactive sensitivity analysis visualization
        
        Args:
            sensitivity_data: Nested dict {output: {parameter: sensitivity}}
            title: Plot title
            
        Returns:
            Interactive Plotly figure
        """
        
        # Prepare data for plotting
        outputs = list(sensitivity_data.keys())
        all_params = set()
        for output_data in sensitivity_data.values():
            all_params.update(output_data.keys())
        all_params = sorted(list(all_params))
        
        # Create subplots
        fig = make_subplots(
            rows=len(outputs), cols=1,
            subplot_titles=[f'Sensitivity: {output}' for output in outputs],
            vertical_spacing=0.1
        )
        
        colors = [self.colors['berkeley_blue'], self.colors['california_gold'],
                 self.colors['green_dark'], self.colors['rose_dark']]
        
        for i, output in enumerate(outputs):
            output_data = sensitivity_data[output]
            
            # Sort parameters by sensitivity
            sorted_params = sorted(all_params, key=lambda p: output_data.get(p, 0), reverse=True)
            sensitivities = [output_data.get(param, 0) for param in sorted_params]
            
            fig.add_trace(
                go.Bar(
                    x=sorted_params, y=sensitivities,
                    name=output,
                    marker_color=colors[i % len(colors)],
                    showlegend=True
                ),
                row=i+1, col=1
            )
            
            # Update y-axis title
            fig.update_yaxes(title_text='Sensitivity Index', row=i+1, col=1)
        
        fig.update_layout(
            title=title,
            height=300 * len(outputs),
            template='plotly_white'
        )
        
        return fig
    
    def interactive_uncertainty_distribution(self, uncertainty_data: Dict[str, np.ndarray],
                                           title: str = 'Interactive Uncertainty Analysis') -> go.Figure:
        """
        Create interactive uncertainty distribution plots
        
        Args:
            uncertainty_data: Dictionary {parameter: samples}
            title: Plot title
            
        Returns:
            Interactive Plotly figure
        """
        
        # Create dropdown for parameter selection
        param_names = list(uncertainty_data.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Distribution', 'Box Plot', 'Q-Q Plot', 'CDF'],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Initial parameter
        initial_param = param_names[0]
        data = uncertainty_data[initial_param]
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=data, nbinsx=50, name='Distribution',
                        marker_color=self.colors['berkeley_blue'], opacity=0.7),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=data, name='Box Plot', marker_color=self.colors['california_gold']),
            row=1, col=2
        )
        
        # Q-Q plot (simplified)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        sample_quantiles = np.sort(data)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', name='Q-Q Plot',
                      marker=dict(color=self.colors['green_dark'])),
            row=2, col=1
        )
        
        # Add reference line for Q-Q plot
        fig.add_trace(
            go.Scatter(x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                      y=[sample_quantiles.min(), sample_quantiles.max()],
                      mode='lines', name='Perfect Normal',
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # CDF
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig.add_trace(
            go.Scatter(x=sorted_data, y=cdf, mode='lines',
                      name='Empirical CDF', line=dict(color=self.colors['rose_dark'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def interactive_device_explorer(self, device_data: Dict[str, Any],
                                  device_type: str = 'mtj') -> widgets.VBox:
        """
        Create interactive device parameter explorer
        
        Args:
            device_data: Device simulation data
            device_type: Type of device ('mtj', 'spin_valve', 'asl')
            
        Returns:
            Interactive widget container
        """
        
        # Create visualization selector
        viz_options = ['I-V Characteristic', 'Resistance vs Field', 'TMR vs Bias',
                      'Energy vs Angle', 'Switching Dynamics']
        
        viz_selector = widgets.Dropdown(
            options=viz_options,
            value=viz_options[0],
            description='Visualization:'
        )
        
        # Parameter controls (example for MTJ)
        if device_type.lower() == 'mtj':
            param_controls = {
                'TMR': widgets.FloatSlider(value=200, min=50, max=500, description='TMR (%)'),
                'RA': widgets.FloatSlider(value=10, min=1, max=100, description='RA (Ω-μm²)'),
                'thickness': widgets.FloatSlider(value=1.2, min=0.8, max=3.0, description='t_ox (nm)')
            }
        else:
            param_controls = {}
        
        # Output
        output = widgets.Output()
        
        def update_device_plot(viz_type, **params):
            """Update device visualization"""
            with output:
                output.clear_output(wait=True)
                
                fig = go.Figure()
                
                if viz_type == 'I-V Characteristic':
                    # Generate I-V data based on parameters
                    V = np.linspace(-1, 1, 200)
                    # Simplified I-V calculation
                    tmr = params.get('TMR', 200)
                    ra = params.get('RA', 10)
                    
                    R_p = ra * 1e-6
                    R_ap = R_p * (1 + tmr/100)
                    
                    I_p = V / R_p
                    I_ap = V / R_ap
                    
                    fig.add_trace(go.Scatter(x=V, y=I_p*1e6, name='Parallel',
                                           line=dict(color=self.colors['berkeley_blue'])))
                    fig.add_trace(go.Scatter(x=V, y=I_ap*1e6, name='Antiparallel',
                                           line=dict(color=self.colors['california_gold'])))
                    
                    fig.update_layout(xaxis_title='Voltage (V)', yaxis_title='Current (μA)')
                
                elif viz_type == 'TMR vs Bias':
                    V = np.linspace(0, 2, 100)
                    tmr_0 = params.get('TMR', 200)
                    
                    # Bias-dependent TMR
                    tmr_bias = tmr_0 / (1 + (V / 0.5)**2)
                    
                    fig.add_trace(go.Scatter(x=V, y=tmr_bias, name='TMR vs Bias',
                                           line=dict(color=self.colors['green_dark'])))
                    
                    fig.update_layout(xaxis_title='Bias (V)', yaxis_title='TMR (%)')
                
                # Add more visualization types here...
                
                fig.update_layout(
                    title=f'{device_type.upper()} - {viz_type}',
                    template='plotly_white',
                    height=500
                )
                
                fig.show()
        
        # Create interactive widget
        widget_dict = {'viz_type': viz_selector}
        widget_dict.update(param_controls)
        
        interactive_plot = widgets.interactive(update_device_plot, **widget_dict)
        
        # Container
        controls = widgets.VBox([viz_selector] + list(param_controls.values()))
        
        container = widgets.VBox([
            widgets.HTML(f'<h3>Interactive {device_type.upper()} Explorer</h3>'),
            controls,
            output
        ])
        
        return container
    
    def create_dashboard(self, simulation_results: Dict[str, Any],
                        title: str = 'SpinCirc Analysis Dashboard') -> go.Figure:
        """
        Create comprehensive analysis dashboard
        
        Args:
            simulation_results: Complete simulation results
            title: Dashboard title
            
        Returns:
            Multi-panel dashboard figure
        """
        
        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Transport Properties', 'Magnetization Dynamics',
                          'Parameter Sensitivity', 'Uncertainty Analysis'],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Panel 1: Transport properties
        if 'transport' in simulation_results:
            transport = simulation_results['transport']
            fig.add_trace(
                go.Scatter(x=transport.get('voltage', []), 
                          y=transport.get('current', []),
                          mode='lines', name='I-V',
                          line=dict(color=self.colors['berkeley_blue'])),
                row=1, col=1
            )
        
        # Panel 2: Magnetization dynamics
        if 'magnetization' in simulation_results:
            mag_data = simulation_results['magnetization']
            time = mag_data.get('time', [])
            mx = mag_data.get('mx', [])
            
            fig.add_trace(
                go.Scatter(x=time, y=mx, mode='lines', name='mx',
                          line=dict(color=self.colors['california_gold'])),
                row=1, col=2
            )
        
        # Panel 3: Sensitivity
        if 'sensitivity' in simulation_results:
            sens_data = simulation_results['sensitivity']
            params = list(sens_data.keys())
            values = list(sens_data.values())
            
            fig.add_trace(
                go.Bar(x=params, y=values, name='Sensitivity',
                      marker_color=self.colors['green_dark']),
                row=2, col=1
            )
        
        # Panel 4: Uncertainty
        if 'uncertainty' in simulation_results:
            unc_data = simulation_results['uncertainty']
            fig.add_trace(
                go.Histogram(x=unc_data, name='Distribution',
                           marker_color=self.colors['rose_dark'], opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig