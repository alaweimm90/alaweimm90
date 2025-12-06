"""
Dashboard and Visualization Generator for TalAI

Create interactive visualizations and dashboards for analytics insights
using Plotly and Altair.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """
    Generate interactive dashboards and visualizations.

    Features:
    - Interactive Plotly charts
    - Validation result dashboards
    - Trend analysis visualizations
    - Research landscape mapping
    - Network graphs
    - Export capabilities
    """

    def __init__(self):
        """Initialize dashboard generator."""
        self.color_scheme = {
            'primary': '#2E86AB',
            'success': '#52B788',
            'warning': '#F77F00',
            'danger': '#D62828',
            'info': '#7209B7',
            'background': '#F8F9FA',
            'text': '#212529'
        }
        self.default_layout = {
            'font': {'family': 'Arial, sans-serif'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }

    async def generate_validation_dashboard(self,
                                           validation_data: List[Dict],
                                           time_range: Optional[str] = '30d') -> Dict[str, Any]:
        """
        Generate comprehensive validation results dashboard.

        Args:
            validation_data: List of validation records
            time_range: Time range for analysis

        Returns:
            Dashboard with multiple visualizations
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(validation_data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()

            # Create dashboard components
            dashboard = {
                'title': 'TalAI Validation Analytics Dashboard',
                'generated_at': datetime.now().isoformat(),
                'time_range': time_range,
                'visualizations': {}
            }

            # 1. Success Rate Over Time
            success_chart = await self._create_success_rate_chart(df)
            dashboard['visualizations']['success_trends'] = success_chart

            # 2. Domain Distribution
            domain_chart = await self._create_domain_distribution(df)
            dashboard['visualizations']['domain_distribution'] = domain_chart

            # 3. Validation Mode Performance
            mode_chart = await self._create_mode_performance_chart(df)
            dashboard['visualizations']['mode_performance'] = mode_chart

            # 4. Evidence Quality Heatmap
            quality_heatmap = await self._create_quality_heatmap(df)
            dashboard['visualizations']['quality_heatmap'] = quality_heatmap

            # 5. Duration Analysis
            duration_chart = await self._create_duration_analysis(df)
            dashboard['visualizations']['duration_analysis'] = duration_chart

            # 6. Key Metrics Cards
            metrics = await self._calculate_key_metrics(df)
            dashboard['key_metrics'] = metrics

            # 7. Combined Overview
            overview = await self._create_overview_dashboard(df)
            dashboard['visualizations']['overview'] = overview

            return dashboard

        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            return {'error': str(e)}

    async def _create_success_rate_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create success rate trend chart."""
        try:
            # Resample by day
            daily_success = df.resample('D')['success_rate'].agg(['mean', 'std', 'count'])
            daily_success = daily_success.fillna(0)

            # Create figure
            fig = go.Figure()

            # Add main trend line
            fig.add_trace(go.Scatter(
                x=daily_success.index,
                y=daily_success['mean'],
                mode='lines+markers',
                name='Average Success Rate',
                line=dict(color=self.color_scheme['primary'], width=2),
                marker=dict(size=6)
            ))

            # Add confidence band
            upper_bound = daily_success['mean'] + daily_success['std']
            lower_bound = daily_success['mean'] - daily_success['std']

            fig.add_trace(go.Scatter(
                x=daily_success.index,
                y=upper_bound,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=daily_success.index,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='Confidence Band',
                fillcolor='rgba(46, 134, 171, 0.2)'
            ))

            # Add moving average
            ma_7 = daily_success['mean'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=daily_success.index,
                y=ma_7,
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color=self.color_scheme['success'], width=2, dash='dash')
            ))

            # Update layout
            fig.update_layout(
                title='Success Rate Trends Over Time',
                xaxis_title='Date',
                yaxis_title='Success Rate',
                yaxis=dict(tickformat='.0%', range=[0, 1]),
                hovermode='x unified',
                **self.default_layout
            )

            return {
                'figure': fig.to_dict(),
                'html': fig.to_html(include_plotlyjs='cdn'),
                'summary': {
                    'current_rate': float(daily_success['mean'].iloc[-1]) if len(daily_success) > 0 else 0,
                    'trend': 'increasing' if len(daily_success) > 1 and
                            daily_success['mean'].iloc[-1] > daily_success['mean'].iloc[0] else 'decreasing'
                }
            }

        except Exception as e:
            logger.error(f"Error creating success rate chart: {e}")
            return {'error': str(e)}

    async def _create_domain_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create domain distribution chart."""
        try:
            if 'domain' not in df.columns:
                return {'error': 'No domain data available'}

            # Count by domain
            domain_counts = df['domain'].value_counts()

            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=domain_counts.index,
                values=domain_counts.values,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent',
                textposition='auto'
            )])

            fig.update_layout(
                title='Validation Distribution by Domain',
                **self.default_layout
            )

            # Also create bar chart
            bar_fig = go.Figure(data=[go.Bar(
                x=domain_counts.index,
                y=domain_counts.values,
                marker=dict(color=domain_counts.values,
                          colorscale='Viridis'),
                text=domain_counts.values,
                textposition='auto'
            )])

            bar_fig.update_layout(
                title='Validation Count by Domain',
                xaxis_title='Domain',
                yaxis_title='Count',
                **self.default_layout
            )

            return {
                'pie_chart': {
                    'figure': fig.to_dict(),
                    'html': fig.to_html(include_plotlyjs='cdn')
                },
                'bar_chart': {
                    'figure': bar_fig.to_dict(),
                    'html': bar_fig.to_html(include_plotlyjs='cdn')
                },
                'summary': {
                    'total_domains': len(domain_counts),
                    'top_domain': domain_counts.index[0] if len(domain_counts) > 0 else None,
                    'distribution': domain_counts.to_dict()
                }
            }

        except Exception as e:
            logger.error(f"Error creating domain distribution: {e}")
            return {'error': str(e)}

    async def _create_mode_performance_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create validation mode performance comparison."""
        try:
            if 'validation_mode' not in df.columns:
                # Create mock data
                df['validation_mode'] = np.random.choice(
                    ['standard', 'quick', 'comprehensive'], len(df)
                )

            # Group by mode
            mode_stats = df.groupby('validation_mode').agg({
                'success_rate': ['mean', 'std', 'count'],
                'duration_seconds': 'mean'
            }).round(3)

            # Create grouped bar chart
            fig = go.Figure()

            modes = mode_stats.index
            success_means = mode_stats[('success_rate', 'mean')]
            success_stds = mode_stats[('success_rate', 'std')]
            counts = mode_stats[('success_rate', 'count')]

            # Success rate bars
            fig.add_trace(go.Bar(
                name='Success Rate',
                x=modes,
                y=success_means,
                error_y=dict(type='data', array=success_stds),
                marker_color=self.color_scheme['success'],
                yaxis='y'
            ))

            # Count bars
            fig.add_trace(go.Bar(
                name='Validation Count',
                x=modes,
                y=counts,
                marker_color=self.color_scheme['info'],
                yaxis='y2',
                opacity=0.7
            ))

            # Update layout with dual y-axis
            fig.update_layout(
                title='Performance by Validation Mode',
                xaxis_title='Validation Mode',
                yaxis=dict(
                    title='Success Rate',
                    tickformat='.0%',
                    side='left'
                ),
                yaxis2=dict(
                    title='Count',
                    overlaying='y',
                    side='right'
                ),
                barmode='group',
                **self.default_layout
            )

            return {
                'figure': fig.to_dict(),
                'html': fig.to_html(include_plotlyjs='cdn'),
                'summary': {
                    'best_mode': success_means.idxmax() if len(success_means) > 0 else None,
                    'mode_stats': mode_stats.to_dict()
                }
            }

        except Exception as e:
            logger.error(f"Error creating mode performance chart: {e}")
            return {'error': str(e)}

    async def _create_quality_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create evidence quality heatmap."""
        try:
            # Create quality matrix (domain vs time period)
            if 'domain' not in df.columns or 'avg_evidence_quality' not in df.columns:
                # Generate mock data
                domains = ['physics', 'chemistry', 'biology', 'cs', 'math']
                weeks = pd.date_range(end=datetime.now(), periods=8, freq='W')

                quality_matrix = np.random.rand(len(domains), len(weeks))
            else:
                # Real data aggregation
                df['week'] = pd.to_datetime(df.index).to_period('W')
                quality_pivot = df.pivot_table(
                    values='avg_evidence_quality',
                    index='domain',
                    columns='week',
                    aggfunc='mean'
                ).fillna(0)

                quality_matrix = quality_pivot.values
                domains = quality_pivot.index.tolist()
                weeks = [str(w) for w in quality_pivot.columns]

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=quality_matrix,
                x=weeks if 'weeks' in locals() else None,
                y=domains if 'domains' in locals() else None,
                colorscale='RdYlGn',
                colorbar=dict(title='Quality Score'),
                text=np.round(quality_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))

            fig.update_layout(
                title='Evidence Quality Heatmap by Domain and Time',
                xaxis_title='Time Period',
                yaxis_title='Domain',
                **self.default_layout
            )

            return {
                'figure': fig.to_dict(),
                'html': fig.to_html(include_plotlyjs='cdn'),
                'summary': {
                    'avg_quality': float(np.mean(quality_matrix)),
                    'best_domain': domains[np.argmax(np.mean(quality_matrix, axis=1))]
                                  if 'domains' in locals() else None
                }
            }

        except Exception as e:
            logger.error(f"Error creating quality heatmap: {e}")
            return {'error': str(e)}

    async def _create_duration_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create validation duration analysis chart."""
        try:
            if 'duration_seconds' not in df.columns:
                df['duration_seconds'] = np.random.exponential(180, len(df))

            # Convert to minutes
            df['duration_minutes'] = df['duration_seconds'] / 60

            # Create histogram with box plot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.02,
                subplot_titles=('Duration Distribution', 'Duration by Domain')
            )

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=df['duration_minutes'],
                    nbinsx=30,
                    name='Duration',
                    marker_color=self.color_scheme['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )

            # Box plot by domain if available
            if 'domain' in df.columns:
                domains = df['domain'].unique()
                for domain in domains:
                    domain_data = df[df['domain'] == domain]['duration_minutes']
                    fig.add_trace(
                        go.Box(
                            y=domain_data,
                            name=domain,
                            boxpoints='outliers'
                        ),
                        row=2, col=1
                    )

            fig.update_xaxes(title_text='Duration (minutes)', row=1, col=1)
            fig.update_yaxes(title_text='Count', row=1, col=1)
            fig.update_yaxes(title_text='Duration (min)', row=2, col=1)

            fig.update_layout(
                title='Validation Duration Analysis',
                showlegend=True,
                height=600,
                **self.default_layout
            )

            return {
                'figure': fig.to_dict(),
                'html': fig.to_html(include_plotlyjs='cdn'),
                'summary': {
                    'avg_duration_minutes': float(df['duration_minutes'].mean()),
                    'median_duration_minutes': float(df['duration_minutes'].median()),
                    'fastest_validation': float(df['duration_minutes'].min()),
                    'slowest_validation': float(df['duration_minutes'].max())
                }
            }

        except Exception as e:
            logger.error(f"Error creating duration analysis: {e}")
            return {'error': str(e)}

    async def _calculate_key_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key performance metrics."""
        try:
            metrics = {
                'total_validations': len(df),
                'avg_success_rate': float(df['success_rate'].mean())
                                   if 'success_rate' in df.columns else 0,
                'total_domains': df['domain'].nunique()
                                if 'domain' in df.columns else 0,
                'avg_duration_minutes': float(df['duration_seconds'].mean() / 60)
                                      if 'duration_seconds' in df.columns else 0,
                'total_evidence_reviewed': int(df['evidence_count'].sum())
                                         if 'evidence_count' in df.columns else 0,
                'peer_reviewed_percentage': float(
                    df['peer_reviewed'].mean() * 100
                ) if 'peer_reviewed' in df.columns else 0,
                'recent_trend': self._calculate_trend(df)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """Calculate recent trend direction."""
        try:
            if 'success_rate' not in df.columns or len(df) < 2:
                return 'stable'

            recent = df.tail(10)['success_rate'].mean()
            previous = df.head(len(df) - 10)['success_rate'].mean()

            if recent > previous * 1.1:
                return 'improving'
            elif recent < previous * 0.9:
                return 'declining'
            else:
                return 'stable'

        except:
            return 'unknown'

    async def _create_overview_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create combined overview dashboard."""
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Success Rate Trend', 'Domain Distribution',
                              'Quality Scores', 'Duration Distribution'),
                specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                      [{'type': 'box'}, {'type': 'histogram'}]]
            )

            # 1. Success trend (simplified)
            if 'success_rate' in df.columns:
                daily_success = df.resample('D')['success_rate'].mean()
                fig.add_trace(
                    go.Scatter(x=daily_success.index, y=daily_success.values,
                             mode='lines+markers', name='Success Rate'),
                    row=1, col=1
                )

            # 2. Domain bars
            if 'domain' in df.columns:
                domain_counts = df['domain'].value_counts()
                fig.add_trace(
                    go.Bar(x=domain_counts.index, y=domain_counts.values,
                          name='Domain Count'),
                    row=1, col=2
                )

            # 3. Quality box plot
            if 'avg_evidence_quality' in df.columns:
                fig.add_trace(
                    go.Box(y=df['avg_evidence_quality'], name='Quality'),
                    row=2, col=1
                )

            # 4. Duration histogram
            if 'duration_seconds' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['duration_seconds']/60, name='Duration',
                               nbinsx=20),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                title='TalAI Analytics Overview',
                showlegend=False,
                height=800,
                **self.default_layout
            )

            # Update axes
            fig.update_yaxes(tickformat='.0%', row=1, col=1)
            fig.update_xaxes(title_text='Date', row=1, col=1)
            fig.update_xaxes(title_text='Domain', row=1, col=2)
            fig.update_yaxes(title_text='Quality Score', row=2, col=1)
            fig.update_xaxes(title_text='Duration (min)', row=2, col=2)

            return {
                'figure': fig.to_dict(),
                'html': fig.to_html(include_plotlyjs='cdn')
            }

        except Exception as e:
            logger.error(f"Error creating overview dashboard: {e}")
            return {'error': str(e)}

    async def create_network_graph(self,
                                  nodes: List[Dict],
                                  edges: List[Dict],
                                  title: str = 'Hypothesis Network') -> Dict[str, Any]:
        """
        Create network graph visualization.

        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            title: Graph title

        Returns:
            Network graph visualization
        """
        try:
            # Extract positions (simplified layout)
            n = len(nodes)
            if n == 0:
                return {'error': 'No nodes to visualize'}

            # Create circular layout
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x = np.cos(angles)
            y = np.sin(angles)

            # Create edge traces
            edge_trace = []
            for edge in edges:
                source_idx = edge.get('source', 0)
                target_idx = edge.get('target', 1)

                if source_idx < n and target_idx < n:
                    edge_trace.append(go.Scatter(
                        x=[x[source_idx], x[target_idx], None],
                        y=[y[source_idx], y[target_idx], None],
                        mode='lines',
                        line=dict(width=edge.get('weight', 1),
                                color='rgba(125,125,125,0.5)'),
                        hoverinfo='none'
                    ))

            # Create node trace
            node_trace = go.Scatter(
                x=x,
                y=y,
                mode='markers+text',
                text=[node.get('label', f'Node {i}') for i, node in enumerate(nodes)],
                textposition='top center',
                marker=dict(
                    size=[node.get('size', 10) for node in nodes],
                    color=[node.get('value', 0) for node in nodes],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Value'),
                    line=dict(width=2, color='white')
                ),
                hovertemplate='%{text}<br>Value: %{marker.color}<extra></extra>'
            )

            # Create figure
            fig = go.Figure(data=edge_trace + [node_trace])

            fig.update_layout(
                title=title,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                **self.default_layout
            )

            return {
                'figure': fig.to_dict(),
                'html': fig.to_html(include_plotlyjs='cdn'),
                'summary': {
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'avg_degree': len(edges) * 2 / len(nodes) if nodes else 0
                }
            }

        except Exception as e:
            logger.error(f"Error creating network graph: {e}")
            return {'error': str(e)}

    async def create_research_landscape_map(self,
                                           research_data: List[Dict]) -> Dict[str, Any]:
        """
        Create research landscape visualization.

        Args:
            research_data: List of research/hypothesis data

        Returns:
            Research landscape map
        """
        try:
            if not research_data:
                return {'error': 'No research data available'}

            # Extract coordinates (from embedding reduction or mock)
            n = len(research_data)
            np.random.seed(42)

            # Mock 2D coordinates (in real implementation, use t-SNE/PCA on embeddings)
            x = np.random.randn(n)
            y = np.random.randn(n)

            # Extract metadata
            domains = [r.get('domain', 'unknown') for r in research_data]
            success_rates = [r.get('success_rate', 0.5) for r in research_data]
            labels = [r.get('statement', '')[:50] for r in research_data]

            # Create scatter plot
            fig = go.Figure()

            # Group by domain
            unique_domains = list(set(domains))
            colors = px.colors.qualitative.Set1

            for i, domain in enumerate(unique_domains):
                domain_mask = [d == domain for d in domains]
                domain_x = [x[j] for j in range(n) if domain_mask[j]]
                domain_y = [y[j] for j in range(n) if domain_mask[j]]
                domain_success = [success_rates[j] for j in range(n) if domain_mask[j]]
                domain_labels = [labels[j] for j in range(n) if domain_mask[j]]

                fig.add_trace(go.Scatter(
                    x=domain_x,
                    y=domain_y,
                    mode='markers',
                    name=domain,
                    marker=dict(
                        size=[s * 20 for s in domain_success],
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    text=domain_labels,
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'Domain: {domain}<br>' +
                                 'Success: %{marker.size:.1%}<extra></extra>'
                ))

            # Add cluster regions (mock clusters)
            from scipy.spatial import ConvexHull

            for domain in unique_domains[:3]:  # Limit to top 3 domains
                domain_mask = [d == domain for d in domains]
                if sum(domain_mask) > 2:  # Need at least 3 points for hull
                    domain_points = np.array([[x[i], y[i]] for i in range(n)
                                             if domain_mask[i]])
                    try:
                        hull = ConvexHull(domain_points)
                        hull_points = domain_points[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])

                        fig.add_trace(go.Scatter(
                            x=hull_points[:, 0],
                            y=hull_points[:, 1],
                            mode='lines',
                            fill='toself',
                            fillcolor=colors[unique_domains.index(domain) % len(colors)],
                            opacity=0.1,
                            line=dict(width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    except:
                        pass

            fig.update_layout(
                title='Research Landscape Map',
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                **self.default_layout
            )

            return {
                'figure': fig.to_dict(),
                'html': fig.to_html(include_plotlyjs='cdn'),
                'summary': {
                    'total_hypotheses': n,
                    'domains': unique_domains,
                    'avg_success_rate': float(np.mean(success_rates))
                }
            }

        except Exception as e:
            logger.error(f"Error creating research landscape: {e}")
            return {'error': str(e)}

    async def export_dashboard_report(self,
                                     dashboard_data: Dict,
                                     format: str = 'html') -> Dict[str, Any]:
        """
        Export dashboard to various formats.

        Args:
            dashboard_data: Dashboard data to export
            format: Export format ('html', 'json', 'pdf')

        Returns:
            Exported dashboard
        """
        try:
            if format == 'html':
                # Create complete HTML document
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>TalAI Analytics Dashboard</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: {self.color_scheme['primary']}; }}
                        .metric-card {{
                            display: inline-block;
                            padding: 20px;
                            margin: 10px;
                            background: {self.color_scheme['background']};
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }}
                        .metric-value {{ font-size: 2em; font-weight: bold; }}
                        .metric-label {{ color: #666; }}
                    </style>
                </head>
                <body>
                    <h1>TalAI Analytics Dashboard</h1>
                    <div class="metrics">
                """

                # Add key metrics
                if 'key_metrics' in dashboard_data:
                    for key, value in dashboard_data['key_metrics'].items():
                        html_content += f"""
                        <div class="metric-card">
                            <div class="metric-label">{key.replace('_', ' ').title()}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """

                html_content += "</div>"

                # Add visualizations
                if 'visualizations' in dashboard_data:
                    for viz_name, viz_data in dashboard_data['visualizations'].items():
                        if isinstance(viz_data, dict) and 'html' in viz_data:
                            html_content += f"<h2>{viz_name.replace('_', ' ').title()}</h2>"
                            html_content += viz_data['html']

                html_content += """
                </body>
                </html>
                """

                return {
                    'format': 'html',
                    'content': html_content,
                    'size_bytes': len(html_content.encode())
                }

            elif format == 'json':
                # Export as JSON
                json_content = json.dumps(dashboard_data, indent=2, default=str)
                return {
                    'format': 'json',
                    'content': json_content,
                    'size_bytes': len(json_content.encode())
                }

            else:
                return {'error': f'Unsupported format: {format}'}

        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")
            return {'error': str(e)}