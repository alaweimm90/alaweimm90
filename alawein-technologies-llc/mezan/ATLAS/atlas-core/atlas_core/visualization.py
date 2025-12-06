"""
Visualization Tools for MEZAN

Provides visualization capabilities for:
- Causal graphs
- Solver performance comparison
- Benchmark results
- Deep analysis insights

Author: MEZAN Research Team
Date: 2025-11-18
Version: 1.0
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CausalGraphVisualizer:
    """
    Visualize causal graphs as ASCII art, Graphviz DOT, or Mermaid diagrams

    Supports multiple output formats for different use cases:
    - ASCII: Terminal-friendly, no dependencies
    - DOT: Graphviz format for publication-quality diagrams
    - Mermaid: GitHub/GitLab markdown-compatible diagrams
    """

    def __init__(self):
        self.available_formats = ["ascii", "dot", "mermaid"]

    def render_ascii(
        self,
        nodes: Dict[str, Any],
        edges: List[Any],
        max_width: int = 70,
    ) -> str:
        """
        Render causal graph as ASCII art

        Args:
            nodes: Dictionary of node_id -> node object
            edges: List of edge objects
            max_width: Maximum width of diagram

        Returns:
            ASCII art representation of graph
        """
        output = []
        output.append("=" * max_width)
        output.append(" CAUSAL GRAPH (ASCII)")
        output.append("=" * max_width)
        output.append("")

        # Group edges by source
        edge_map = {}
        for edge in edges:
            source = edge.source if hasattr(edge, 'source') else edge.get('source')
            if source not in edge_map:
                edge_map[source] = []
            edge_map[source].append(edge)

        # Render nodes and edges
        output.append("NODES:")
        output.append("-" * max_width)
        for node_id, node in nodes.items():
            name = node.name if hasattr(node, 'name') else node.get('name', node_id)
            node_type = node.type if hasattr(node, 'type') else node.get('type', 'unknown')
            output.append(f"[{node_id}] {name} ({node_type})")
        output.append("")

        output.append("CAUSAL RELATIONSHIPS:")
        output.append("-" * max_width)
        for source, source_edges in edge_map.items():
            source_name = nodes.get(source, {}).name if hasattr(nodes.get(source, {}), 'name') else source
            for edge in source_edges:
                target = edge.target if hasattr(edge, 'target') else edge.get('target')
                target_name = nodes.get(target, {}).name if hasattr(nodes.get(target, {}), 'name') else target
                strength = edge.strength if hasattr(edge, 'strength') else edge.get('strength', 1.0)

                # Visual strength indicator
                if strength > 0.8:
                    arrow = "═══>"
                elif strength > 0.6:
                    arrow = "═══>"
                elif strength > 0.4:
                    arrow = "──>"
                else:
                    arrow = "- ->"

                output.append(f"{source_name:20s} {arrow} {target_name:20s} [{strength:.2f}]")

        output.append("")
        output.append("=" * max_width)
        return "\n".join(output)

    def render_dot(
        self,
        nodes: Dict[str, Any],
        edges: List[Any],
        graph_name: str = "causal_graph",
    ) -> str:
        """
        Render causal graph as Graphviz DOT format

        Args:
            nodes: Dictionary of node_id -> node object
            edges: List of edge objects
            graph_name: Name for the graph

        Returns:
            DOT format string
        """
        dot = []
        dot.append(f'digraph {graph_name} {{')
        dot.append('    rankdir=LR;')
        dot.append('    node [shape=box, style=rounded];')
        dot.append('')

        # Add nodes with styling by type
        dot.append('    // Nodes')
        for node_id, node in nodes.items():
            name = node.name if hasattr(node, 'name') else node.get('name', node_id)
            node_type = node.type if hasattr(node, 'type') else node.get('type', 'unknown')

            # Color by type
            if 'problem' in node_type:
                color = 'lightblue'
            elif 'algorithm' in node_type:
                color = 'lightgreen'
            elif 'performance' in node_type:
                color = 'lightyellow'
            else:
                color = 'lightgray'

            dot.append(f'    {node_id} [label="{name}", fillcolor={color}, style=filled];')

        dot.append('')
        dot.append('    // Edges')

        # Add edges with weight-based styling
        for edge in edges:
            source = edge.source if hasattr(edge, 'source') else edge.get('source')
            target = edge.target if hasattr(edge, 'target') else edge.get('target')
            strength = edge.strength if hasattr(edge, 'strength') else edge.get('strength', 1.0)

            # Edge style based on strength
            if strength > 0.8:
                style = 'bold'
                penwidth = '3.0'
            elif strength > 0.6:
                style = 'solid'
                penwidth = '2.0'
            else:
                style = 'dashed'
                penwidth = '1.0'

            label = f'{strength:.2f}'
            dot.append(f'    {source} -> {target} [label="{label}", style={style}, penwidth={penwidth}];')

        dot.append('}')
        return '\n'.join(dot)

    def render_mermaid(
        self,
        nodes: Dict[str, Any],
        edges: List[Any],
    ) -> str:
        """
        Render causal graph as Mermaid diagram

        Args:
            nodes: Dictionary of node_id -> node object
            edges: List of edge objects

        Returns:
            Mermaid format string
        """
        mermaid = []
        mermaid.append('```mermaid')
        mermaid.append('graph LR')
        mermaid.append('')

        # Add nodes with shapes
        mermaid.append('    %% Nodes')
        for node_id, node in nodes.items():
            name = node.name if hasattr(node, 'name') else node.get('name', node_id)
            # Escape special characters
            name = name.replace('"', '\\"')
            mermaid.append(f'    {node_id}["{name}"]')

        mermaid.append('')
        mermaid.append('    %% Causal relationships')

        # Add edges
        for edge in edges:
            source = edge.source if hasattr(edge, 'source') else edge.get('source')
            target = edge.target if hasattr(edge, 'target') else edge.get('target')
            strength = edge.strength if hasattr(edge, 'strength') else edge.get('strength', 1.0)

            # Different arrow styles for different strengths
            if strength > 0.8:
                arrow = '==>'
            elif strength > 0.6:
                arrow = '-->'
            else:
                arrow = '-.->'

            label = f'{strength:.2f}'
            mermaid.append(f'    {source} {arrow}|{label}| {target}')

        mermaid.append('```')
        return '\n'.join(mermaid)

    def visualize(
        self,
        nodes: Dict[str, Any],
        edges: List[Any],
        format: str = "ascii",
        **kwargs
    ) -> str:
        """
        Visualize causal graph in specified format

        Args:
            nodes: Dictionary of node_id -> node object
            edges: List of edge objects
            format: Output format ("ascii", "dot", "mermaid")
            **kwargs: Format-specific arguments

        Returns:
            Visualization string in specified format
        """
        format = format.lower()

        if format == "ascii":
            return self.render_ascii(nodes, edges, **kwargs)
        elif format == "dot":
            return self.render_dot(nodes, edges, **kwargs)
        elif format == "mermaid":
            return self.render_mermaid(nodes, edges)
        else:
            raise ValueError(
                f"Unknown format: {format}. Available: {self.available_formats}"
            )


class PerformanceVisualizer:
    """
    Visualize solver performance and benchmark results
    """

    def __init__(self):
        pass

    def render_comparison_ascii(
        self,
        comparison: Dict[str, float],
        title: str = "Solver Comparison",
        metric: str = "time",
        max_width: int = 70,
    ) -> str:
        """
        Render solver comparison as ASCII bar chart

        Args:
            comparison: Dict mapping solver names to metric values
            title: Chart title
            metric: Metric being compared
            max_width: Maximum width of chart

        Returns:
            ASCII bar chart
        """
        output = []
        output.append("=" * max_width)
        output.append(f" {title}")
        output.append("=" * max_width)
        output.append("")

        if not comparison:
            output.append("No data to display")
            output.append("")
            return "\n".join(output)

        # Find max value for scaling
        max_value = max(comparison.values())
        if max_value == 0:
            max_value = 1.0

        # Calculate bar width
        label_width = max(len(name) for name in comparison.keys())
        value_width = 10
        bar_width = max_width - label_width - value_width - 5

        # Sort by value
        sorted_items = sorted(comparison.items(), key=lambda x: x[1])

        # Render bars
        for name, value in sorted_items:
            # Calculate bar length
            bar_len = int((value / max_value) * bar_width)
            bar = "█" * bar_len

            # Format value
            if metric == "time":
                value_str = f"{value:8.4f}s"
            else:
                value_str = f"{value:8.4f}"

            output.append(f"{name:{label_width}s} {bar} {value_str}")

        output.append("")
        output.append("=" * max_width)
        return "\n".join(output)

    def render_benchmark_summary_ascii(
        self,
        statistics: Dict[str, Any],
        max_width: int = 70,
    ) -> str:
        """
        Render benchmark statistics summary

        Args:
            statistics: Statistics dictionary from BenchmarkSuite
            max_width: Maximum width

        Returns:
            ASCII formatted summary
        """
        output = []
        output.append("=" * max_width)
        output.append(" BENCHMARK STATISTICS SUMMARY")
        output.append("=" * max_width)
        output.append("")

        # Overview
        output.append("OVERVIEW:")
        output.append(f"  Total Benchmarks: {statistics.get('total', 0)}")
        output.append(f"  Successful: {statistics.get('successful', 0)}")
        output.append(f"  Failed: {statistics.get('failed', 0)}")
        output.append("")

        # Time statistics
        if 'time' in statistics:
            time_stats = statistics['time']
            output.append("TIME STATISTICS:")
            output.append(f"  Mean:     {time_stats.get('mean', 0):.4f}s")
            output.append(f"  Median:   {time_stats.get('median', 0):.4f}s")
            output.append(f"  Std Dev:  {time_stats.get('stdev', 0):.4f}s")
            output.append(f"  Min:      {time_stats.get('min', 0):.4f}s")
            output.append(f"  Max:      {time_stats.get('max', 0):.4f}s")
            output.append(f"  Total:    {time_stats.get('total', 0):.4f}s")
            output.append("")

        # Objective statistics
        if 'objective' in statistics:
            obj_stats = statistics['objective']
            output.append("OBJECTIVE VALUE STATISTICS:")
            output.append(f"  Mean:     {obj_stats.get('mean', 0):.4f}")
            output.append(f"  Median:   {obj_stats.get('median', 0):.4f}")
            output.append(f"  Std Dev:  {obj_stats.get('stdev', 0):.4f}")
            output.append(f"  Min:      {obj_stats.get('min', 0):.4f}")
            output.append(f"  Max:      {obj_stats.get('max', 0):.4f}")
            output.append("")

        # Iterations statistics
        if 'iterations' in statistics:
            iter_stats = statistics['iterations']
            output.append("ITERATION STATISTICS:")
            output.append(f"  Mean:     {iter_stats.get('mean', 0):.1f}")
            output.append(f"  Median:   {iter_stats.get('median', 0)}")
            output.append(f"  Min:      {iter_stats.get('min', 0)}")
            output.append(f"  Max:      {iter_stats.get('max', 0)}")
            output.append("")

        output.append("=" * max_width)
        return "\n".join(output)


# Convenience functions
def visualize_causal_graph(
    causal_engine,
    format: str = "ascii",
    **kwargs
) -> str:
    """
    Visualize causal graph from CausalReasoningEngine

    Args:
        causal_engine: CausalReasoningEngine instance
        format: Output format ("ascii", "dot", "mermaid")
        **kwargs: Format-specific arguments

    Returns:
        Visualization string
    """
    visualizer = CausalGraphVisualizer()
    return visualizer.visualize(
        nodes=causal_engine.nodes,
        edges=causal_engine.edges,
        format=format,
        **kwargs
    )


def visualize_solver_comparison(
    comparison: Dict[str, float],
    metric: str = "time",
    title: Optional[str] = None,
    **kwargs
) -> str:
    """
    Visualize solver performance comparison

    Args:
        comparison: Dict mapping solver names to metric values
        metric: Metric being compared
        title: Chart title
        **kwargs: Additional arguments

    Returns:
        ASCII bar chart
    """
    visualizer = PerformanceVisualizer()
    if title is None:
        title = f"Solver Comparison ({metric})"
    return visualizer.render_comparison_ascii(
        comparison=comparison,
        title=title,
        metric=metric,
        **kwargs
    )


def visualize_benchmark_results(
    benchmark_suite,
    **kwargs
) -> str:
    """
    Visualize benchmark suite results

    Args:
        benchmark_suite: BenchmarkSuite instance
        **kwargs: Additional arguments

    Returns:
        ASCII formatted summary
    """
    visualizer = PerformanceVisualizer()
    statistics = benchmark_suite.get_statistics()
    return visualizer.render_benchmark_summary_ascii(statistics, **kwargs)
