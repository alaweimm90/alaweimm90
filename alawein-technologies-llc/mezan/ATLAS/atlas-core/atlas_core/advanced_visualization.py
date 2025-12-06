"""
Advanced Visualization System for ORCHEX
Provides interactive workflow DAG visualization, real-time execution monitoring,
performance analysis, and multi-format export capabilities.
"""

import json
import base64
import hashlib
import colorsys
from io import BytesIO, StringIO
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import re

# For visualization export
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
    from matplotlib.path import Path
    import matplotlib.patheffects as path_effects
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# For graph analysis
import networkx as nx

# For export formats
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class NodeStatus(Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"
    RETRYING = "retrying"


class VisualizationType(Enum):
    """Types of visualizations."""
    DAG = "dag"
    FLOW = "flow"
    HEATMAP = "heatmap"
    FLAME_GRAPH = "flame_graph"
    DEPENDENCY = "dependency"
    TIMELINE = "timeline"
    RESOURCE = "resource"
    PERFORMANCE = "performance"


@dataclass
class NodeMetrics:
    """Metrics for a workflow node."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    input_size: float = 0.0
    output_size: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    throughput: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0


@dataclass
class WorkflowNode:
    """Represents a node in the workflow."""
    id: str
    name: str
    type: str
    status: NodeStatus = NodeStatus.PENDING
    position: Optional[Tuple[float, float]] = None
    dependencies: List[str] = field(default_factory=list)
    metrics: NodeMetrics = field(default_factory=NodeMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    children: List[str] = field(default_factory=list)


@dataclass
class WorkflowEdge:
    """Represents an edge in the workflow."""
    source: str
    target: str
    label: Optional[str] = None
    weight: float = 1.0
    type: str = "normal"  # normal, conditional, parallel
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedVisualizationEngine:
    """Advanced visualization engine for ORCHEX workflows."""

    def __init__(self):
        """Initialize the visualization engine."""
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self.graph = nx.DiGraph()
        self.layout_cache: Dict[str, Any] = {}
        self.color_scheme = self._initialize_color_scheme()
        self.animation_frames: List[Dict[str, Any]] = []

    def _initialize_color_scheme(self) -> Dict[str, str]:
        """Initialize color scheme for visualization."""
        return {
            NodeStatus.PENDING: "#E5E7EB",
            NodeStatus.RUNNING: "#3B82F6",
            NodeStatus.COMPLETED: "#10B981",
            NodeStatus.FAILED: "#EF4444",
            NodeStatus.SKIPPED: "#9CA3AF",
            NodeStatus.PAUSED: "#F59E0B",
            NodeStatus.RETRYING: "#8B5CF6",
            "background": "#1F2937",
            "edge": "#6B7280",
            "text": "#F9FAFB",
            "grid": "#374151"
        }

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the visualization."""
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            label=node.name,
            type=node.type,
            status=node.status.value
        )

    def add_edge(self, edge: WorkflowEdge) -> None:
        """Add an edge to the visualization."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            label=edge.label,
            weight=edge.weight,
            type=edge.type
        )

    def calculate_layout(self, algorithm: str = "hierarchical") -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified layout algorithm."""

        if algorithm in self.layout_cache:
            return self.layout_cache[algorithm]

        if algorithm == "hierarchical":
            pos = self._hierarchical_layout()
        elif algorithm == "force":
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif algorithm == "circular":
            pos = nx.circular_layout(self.graph)
        elif algorithm == "spectral":
            pos = nx.spectral_layout(self.graph)
        elif algorithm == "dagre":
            pos = self._dagre_layout()
        else:
            pos = nx.kamada_kawai_layout(self.graph)

        # Normalize positions to [0, 1] range
        pos = self._normalize_positions(pos)

        # Update node positions
        for node_id, (x, y) in pos.items():
            if node_id in self.nodes:
                self.nodes[node_id].position = (x, y)

        self.layout_cache[algorithm] = pos
        return pos

    def _hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Calculate hierarchical layout for DAG."""
        # Calculate levels using topological sort
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # Graph has cycles, fall back to spring layout
            return nx.spring_layout(self.graph)

        levels = {}
        for node in topo_order:
            predecessors = list(self.graph.predecessors(node))
            if not predecessors:
                levels[node] = 0
            else:
                levels[node] = max(levels[p] for p in predecessors) + 1

        # Group nodes by level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        # Calculate positions
        pos = {}
        max_level = max(level_nodes.keys()) if level_nodes else 0

        for level, nodes in level_nodes.items():
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                x = (i + 0.5) / n_nodes if n_nodes > 0 else 0.5
                y = 1 - (level / (max_level + 1)) if max_level > 0 else 0.5
                pos[node] = (x, y)

        return pos

    def _dagre_layout(self) -> Dict[str, Tuple[float, float]]:
        """Calculate layout using Dagre-like algorithm."""
        # Simplified Dagre-like layout
        # This is a basic implementation - real Dagre is more complex

        # First, create layers using longest path
        layers = self._create_layers()

        # Then minimize crossings
        layers = self._minimize_crossings(layers)

        # Finally, assign coordinates
        pos = {}
        max_layer = len(layers) - 1 if layers else 0

        for layer_idx, layer_nodes in enumerate(layers):
            n_nodes = len(layer_nodes)
            for node_idx, node in enumerate(layer_nodes):
                x = (node_idx + 0.5) / n_nodes if n_nodes > 0 else 0.5
                y = layer_idx / (max_layer + 1) if max_layer > 0 else 0.5
                pos[node] = (x, y)

        return pos

    def _create_layers(self) -> List[List[str]]:
        """Create layers for Dagre layout."""
        layers = []
        remaining = set(self.graph.nodes())

        while remaining:
            # Find nodes with no dependencies in remaining set
            layer = []
            for node in remaining:
                preds = set(self.graph.predecessors(node)) & remaining
                if not preds:
                    layer.append(node)

            if not layer:
                # Cycle detected, add remaining nodes
                layer = list(remaining)

            layers.append(layer)
            remaining -= set(layer)

        return layers

    def _minimize_crossings(self, layers: List[List[str]]) -> List[List[str]]:
        """Minimize edge crossings between layers."""
        # Simple barycentric heuristic
        for i in range(1, len(layers)):
            # Sort nodes in layer based on average position of predecessors
            prev_layer = layers[i-1]
            curr_layer = layers[i]

            positions = {}
            for node in curr_layer:
                preds = [n for n in self.graph.predecessors(node) if n in prev_layer]
                if preds:
                    avg_pos = sum(prev_layer.index(p) for p in preds) / len(preds)
                else:
                    avg_pos = len(prev_layer) / 2
                positions[node] = avg_pos

            layers[i] = sorted(curr_layer, key=lambda n: positions[n])

        return layers

    def _normalize_positions(self, pos: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Normalize positions to [0, 1] range."""
        if not pos:
            return pos

        x_coords = [x for x, y in pos.values()]
        y_coords = [y for x, y in pos.values()]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        normalized = {}
        for node, (x, y) in pos.items():
            nx = (x - x_min) / x_range
            ny = (y - y_min) / y_range
            normalized[node] = (nx, ny)

        return normalized

    def generate_dag_visualization(self, layout: str = "hierarchical",
                                 show_metrics: bool = True) -> Dict[str, Any]:
        """Generate interactive DAG visualization data."""
        pos = self.calculate_layout(layout)

        # Prepare nodes data
        nodes_data = []
        for node_id, node in self.nodes.items():
            x, y = node.position or (0.5, 0.5)

            node_data = {
                "id": node_id,
                "label": node.name,
                "x": x * 1000,  # Scale to pixel coordinates
                "y": y * 600,
                "type": node.type,
                "status": node.status.value,
                "color": self.color_scheme[node.status],
                "size": 30 + math.log(1 + node.metrics.execution_time) * 5
            }

            if show_metrics:
                node_data["metrics"] = {
                    "execution_time": f"{node.metrics.execution_time:.2f}s",
                    "memory": f"{node.metrics.memory_usage:.1f}MB",
                    "cpu": f"{node.metrics.cpu_usage:.1f}%",
                    "throughput": f"{node.metrics.throughput:.0f}/s"
                }

            nodes_data.append(node_data)

        # Prepare edges data
        edges_data = []
        for edge in self.edges:
            source_pos = self.nodes[edge.source].position or (0, 0)
            target_pos = self.nodes[edge.target].position or (1, 1)

            edges_data.append({
                "source": edge.source,
                "target": edge.target,
                "sourceX": source_pos[0] * 1000,
                "sourceY": source_pos[1] * 600,
                "targetX": target_pos[0] * 1000,
                "targetY": target_pos[1] * 600,
                "label": edge.label,
                "type": edge.type,
                "animated": self.nodes[edge.target].status == NodeStatus.RUNNING
            })

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "layout": layout,
            "timestamp": datetime.now().isoformat()
        }

    def generate_flow_animation(self, duration: int = 5000) -> List[Dict[str, Any]]:
        """Generate animation frames for workflow execution flow."""
        frames = []

        # Calculate execution timeline
        timeline = self._calculate_execution_timeline()

        # Generate frames
        num_frames = 60  # 60 FPS for smooth animation
        frame_duration = duration / num_frames

        for frame_idx in range(num_frames):
            timestamp = frame_idx * frame_duration
            frame_data = {
                "frame": frame_idx,
                "timestamp": timestamp,
                "nodes": {},
                "edges": {},
                "particles": []
            }

            # Update node states based on timeline
            for node_id, node in self.nodes.items():
                if node.start_time:
                    start_ms = 0  # Simplified
                    end_ms = node.metrics.execution_time * 1000

                    if timestamp < start_ms:
                        status = NodeStatus.PENDING
                    elif timestamp < end_ms:
                        status = NodeStatus.RUNNING
                        # Add particle effect
                        frame_data["particles"].append({
                            "nodeId": node_id,
                            "progress": (timestamp - start_ms) / (end_ms - start_ms)
                        })
                    else:
                        status = NodeStatus.COMPLETED
                else:
                    status = node.status

                frame_data["nodes"][node_id] = {
                    "status": status.value,
                    "color": self.color_scheme[status]
                }

            # Animate edges
            for edge in self.edges:
                source_status = frame_data["nodes"][edge.source]["status"]
                target_status = frame_data["nodes"][edge.target]["status"]

                if source_status == "completed" and target_status == "running":
                    frame_data["edges"][f"{edge.source}-{edge.target}"] = {
                        "animated": True,
                        "flow": True
                    }

            frames.append(frame_data)

        return frames

    def _calculate_execution_timeline(self) -> List[Tuple[float, str, str]]:
        """Calculate execution timeline for animation."""
        timeline = []

        for node_id, node in self.nodes.items():
            if node.start_time and node.end_time:
                start = 0  # Simplified - would calculate from actual times
                end = node.metrics.execution_time
                timeline.append((start, node_id, "start"))
                timeline.append((end, node_id, "end"))

        timeline.sort(key=lambda x: x[0])
        return timeline

    def generate_heatmap(self, metric: str = "execution_time") -> Dict[str, Any]:
        """Generate heatmap visualization for resource usage."""
        # Create grid for heatmap
        grid_size = 20
        heatmap_data = np.zeros((grid_size, grid_size))

        # Map nodes to grid cells
        pos = self.calculate_layout()

        for node_id, node in self.nodes.items():
            x, y = pos.get(node_id, (0.5, 0.5))
            grid_x = int(x * (grid_size - 1))
            grid_y = int(y * (grid_size - 1))

            # Get metric value
            if metric == "execution_time":
                value = node.metrics.execution_time
            elif metric == "memory_usage":
                value = node.metrics.memory_usage
            elif metric == "cpu_usage":
                value = node.metrics.cpu_usage
            elif metric == "error_count":
                value = node.metrics.error_count
            else:
                value = 0

            # Apply Gaussian kernel for smooth heatmap
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    gx = grid_x + dx
                    gy = grid_y + dy
                    if 0 <= gx < grid_size and 0 <= gy < grid_size:
                        distance = math.sqrt(dx*dx + dy*dy)
                        weight = math.exp(-distance/2)
                        heatmap_data[gy, gx] += value * weight

        # Normalize heatmap
        if heatmap_data.max() > 0:
            heatmap_data /= heatmap_data.max()

        return {
            "data": heatmap_data.tolist(),
            "metric": metric,
            "min": float(heatmap_data.min()),
            "max": float(heatmap_data.max()),
            "grid_size": grid_size,
            "color_scale": "viridis"
        }

    def generate_flame_graph(self) -> Dict[str, Any]:
        """Generate flame graph for performance analysis."""
        # Build call stack tree
        root = {
            "name": "root",
            "value": 0,
            "children": []
        }

        # Create stack frames from workflow execution
        for node_id, node in self.nodes.items():
            frame = {
                "name": node.name,
                "value": node.metrics.execution_time,
                "type": node.type,
                "metrics": {
                    "cpu": node.metrics.cpu_usage,
                    "memory": node.metrics.memory_usage
                },
                "children": []
            }

            # Find parent based on dependencies
            if node.dependencies:
                # Simplified - would need proper tree construction
                root["children"].append(frame)
            else:
                root["children"].append(frame)

            root["value"] += node.metrics.execution_time

        return self._flatten_flame_graph(root)

    def _flatten_flame_graph(self, node: Dict[str, Any], level: int = 0) -> Dict[str, Any]:
        """Flatten flame graph for rendering."""
        flattened = {
            "levels": [],
            "max_level": 0
        }

        def flatten_recursive(n: Dict[str, Any], l: int, x: float, width: float):
            if l >= len(flattened["levels"]):
                flattened["levels"].append([])

            flattened["levels"][l].append({
                "name": n["name"],
                "x": x,
                "width": width,
                "value": n["value"],
                "metrics": n.get("metrics", {})
            })

            flattened["max_level"] = max(flattened["max_level"], l)

            # Process children
            if n.get("children"):
                child_x = x
                total_value = sum(c["value"] for c in n["children"])

                for child in n["children"]:
                    if total_value > 0:
                        child_width = (child["value"] / total_value) * width
                        flatten_recursive(child, l + 1, child_x, child_width)
                        child_x += child_width

        flatten_recursive(node, 0, 0, 1)
        return flattened

    def generate_dependency_graph(self, highlight_critical_path: bool = True) -> Dict[str, Any]:
        """Generate dependency graph with critical path highlighting."""
        # Find critical path
        critical_path = []
        if highlight_critical_path:
            critical_path = self._find_critical_path()

        # Generate graph data
        graph_data = {
            "nodes": [],
            "edges": [],
            "critical_path": critical_path
        }

        for node_id, node in self.nodes.items():
            is_critical = node_id in critical_path

            graph_data["nodes"].append({
                "id": node_id,
                "label": node.name,
                "is_critical": is_critical,
                "color": "#EF4444" if is_critical else "#3B82F6",
                "size": 40 if is_critical else 30
            })

        for edge in self.edges:
            is_critical = edge.source in critical_path and edge.target in critical_path

            graph_data["edges"].append({
                "source": edge.source,
                "target": edge.target,
                "is_critical": is_critical,
                "width": 3 if is_critical else 1,
                "color": "#EF4444" if is_critical else "#6B7280"
            })

        return graph_data

    def _find_critical_path(self) -> List[str]:
        """Find critical path in the workflow."""
        # Use networkx to find longest path (critical path)
        try:
            # For DAG, find longest path by weight
            if nx.is_directed_acyclic_graph(self.graph):
                # Add execution time as edge weight
                for node_id, node in self.nodes.items():
                    for successor in self.graph.successors(node_id):
                        self.graph[node_id][successor]['weight'] = node.metrics.execution_time

                # Find longest path
                critical_path = nx.dag_longest_path(
                    self.graph,
                    weight='weight',
                    default_weight=1
                )
                return critical_path
            else:
                return []
        except:
            return []

    def export_to_svg(self, visualization_type: VisualizationType = VisualizationType.DAG) -> str:
        """Export visualization to SVG format."""
        svg_elements = []

        # SVG header
        svg_elements.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_elements.append('<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">')

        # Background
        svg_elements.append(f'<rect width="1200" height="800" fill="{self.color_scheme["background"]}"/>')

        # Calculate positions
        pos = self.calculate_layout()

        # Draw edges
        for edge in self.edges:
            source_pos = self.nodes[edge.source].position or (0, 0)
            target_pos = self.nodes[edge.target].position or (1, 1)

            x1, y1 = source_pos[0] * 1100 + 50, source_pos[1] * 700 + 50
            x2, y2 = target_pos[0] * 1100 + 50, target_pos[1] * 700 + 50

            # Draw arrow
            svg_elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{self.color_scheme["edge"]}" stroke-width="2" '
                f'marker-end="url(#arrowhead)"/>'
            )

        # Draw nodes
        for node_id, node in self.nodes.items():
            x, y = node.position or (0.5, 0.5)
            x = x * 1100 + 50
            y = y * 700 + 50

            # Node circle
            color = self.color_scheme[node.status]
            svg_elements.append(
                f'<circle cx="{x}" cy="{y}" r="25" fill="{color}" '
                f'stroke="white" stroke-width="2"/>'
            )

            # Node label
            svg_elements.append(
                f'<text x="{x}" y="{y + 5}" text-anchor="middle" '
                f'fill="white" font-size="12" font-family="Arial">{node.name}</text>'
            )

        # Arrow marker definition
        svg_elements.insert(2, '''
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7"
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#6B7280"/>
                </marker>
            </defs>
        ''')

        # SVG footer
        svg_elements.append('</svg>')

        return '\n'.join(svg_elements)

    def export_to_json(self, include_metrics: bool = True) -> str:
        """Export visualization data to JSON format."""
        export_data = {
            "workflow": {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "node_count": len(self.nodes),
                    "edge_count": len(self.edges)
                }
            }
        }

        # Export nodes
        for node_id, node in self.nodes.items():
            node_data = {
                "id": node_id,
                "name": node.name,
                "type": node.type,
                "status": node.status.value,
                "position": node.position,
                "dependencies": node.dependencies
            }

            if include_metrics:
                node_data["metrics"] = {
                    "execution_time": node.metrics.execution_time,
                    "memory_usage": node.metrics.memory_usage,
                    "cpu_usage": node.metrics.cpu_usage,
                    "error_count": node.metrics.error_count,
                    "retry_count": node.metrics.retry_count
                }

            export_data["workflow"]["nodes"].append(node_data)

        # Export edges
        for edge in self.edges:
            export_data["workflow"]["edges"].append({
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "type": edge.type,
                "weight": edge.weight
            })

        return json.dumps(export_data, indent=2)

    def export_to_html(self, visualization_type: VisualizationType = VisualizationType.DAG,
                       interactive: bool = True) -> str:
        """Export visualization to HTML format with D3.js/Cytoscape.js."""
        if interactive:
            return self._generate_interactive_html()
        else:
            return self._generate_static_html()

    def _generate_interactive_html(self) -> str:
        """Generate interactive HTML visualization."""
        viz_data = self.generate_dag_visualization()

        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ORCHEX Workflow Visualization</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #1F2937;
                }}
                #visualization {{
                    width: 100%;
                    height: 800px;
                    border: 2px solid #374151;
                    border-radius: 8px;
                    background: #111827;
                }}
                .node {{
                    cursor: pointer;
                }}
                .node-label {{
                    fill: white;
                    font-size: 12px;
                    pointer-events: none;
                }}
                .edge {{
                    stroke: #6B7280;
                    stroke-width: 2;
                    fill: none;
                }}
                .edge-animated {{
                    stroke-dasharray: 5, 5;
                    animation: dash 1s linear infinite;
                }}
                @keyframes dash {{
                    to {{
                        stroke-dashoffset: -10;
                    }}
                }}
                .tooltip {{
                    position: absolute;
                    text-align: left;
                    padding: 10px;
                    font-size: 12px;
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    border-radius: 4px;
                    pointer-events: none;
                    opacity: 0;
                    transition: opacity 0.3s;
                }}
            </style>
        </head>
        <body>
            <div id="visualization"></div>
            <div class="tooltip"></div>
            <script>
                const data = {0};

                const width = 1200;
                const height = 800;

                const svg = d3.select("#visualization")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);

                const tooltip = d3.select(".tooltip");

                // Create edges
                const edges = svg.selectAll(".edge")
                    .data(data.edges)
                    .enter()
                    .append("path")
                    .attr("class", d => d.animated ? "edge edge-animated" : "edge")
                    .attr("d", d => {{
                        return `M${{d.sourceX}},${{d.sourceY}} L${{d.targetX}},${{d.targetY}}`;
                    }});

                // Create nodes
                const nodes = svg.selectAll(".node")
                    .data(data.nodes)
                    .enter()
                    .append("g")
                    .attr("class", "node")
                    .attr("transform", d => `translate(${{d.x}}, ${{d.y}})`);

                nodes.append("circle")
                    .attr("r", d => d.size)
                    .attr("fill", d => d.color)
                    .attr("stroke", "white")
                    .attr("stroke-width", 2);

                nodes.append("text")
                    .attr("class", "node-label")
                    .attr("text-anchor", "middle")
                    .attr("dy", ".35em")
                    .text(d => d.label);

                // Hover effects
                nodes.on("mouseenter", function(event, d) {{
                    if (d.metrics) {{
                        tooltip.transition().duration(200).style("opacity", 1);
                        tooltip.html(`
                            <strong>${{d.label}}</strong><br>
                            Status: ${{d.status}}<br>
                            Time: ${{d.metrics.execution_time}}<br>
                            Memory: ${{d.metrics.memory}}<br>
                            CPU: ${{d.metrics.cpu}}
                        `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                    }}
                }})
                .on("mouseleave", function() {{
                    tooltip.transition().duration(500).style("opacity", 0);
                }});
            </script>
        </body>
        </html>
        """.format(json.dumps(viz_data))

        return html_template

    def _generate_static_html(self) -> str:
        """Generate static HTML visualization."""
        svg_content = self.export_to_svg()

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ORCHEX Workflow Visualization</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #1F2937;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }}
            </style>
        </head>
        <body>
            {svg_content}
        </body>
        </html>
        """

        return html_template

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze workflow performance and provide insights."""
        analysis = {
            "summary": {},
            "bottlenecks": [],
            "optimization_suggestions": [],
            "resource_usage": {},
            "execution_patterns": {}
        }

        # Summary statistics
        total_time = sum(node.metrics.execution_time for node in self.nodes.values())
        avg_time = total_time / len(self.nodes) if self.nodes else 0
        max_time_node = max(self.nodes.values(),
                          key=lambda n: n.metrics.execution_time,
                          default=None)

        analysis["summary"] = {
            "total_execution_time": total_time,
            "average_node_time": avg_time,
            "slowest_node": max_time_node.name if max_time_node else None,
            "slowest_node_time": max_time_node.metrics.execution_time if max_time_node else 0,
            "total_nodes": len(self.nodes),
            "failed_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED)
        }

        # Identify bottlenecks
        threshold = avg_time * 2  # Nodes taking 2x average time
        for node in self.nodes.values():
            if node.metrics.execution_time > threshold:
                analysis["bottlenecks"].append({
                    "node": node.name,
                    "execution_time": node.metrics.execution_time,
                    "factor": node.metrics.execution_time / avg_time
                })

        # Resource usage analysis
        analysis["resource_usage"] = {
            "total_memory": sum(n.metrics.memory_usage for n in self.nodes.values()),
            "peak_memory": max((n.metrics.memory_usage for n in self.nodes.values()), default=0),
            "average_cpu": sum(n.metrics.cpu_usage for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0
        }

        # Execution patterns
        critical_path = self._find_critical_path()
        analysis["execution_patterns"] = {
            "critical_path_length": len(critical_path),
            "critical_path_time": sum(self.nodes[n].metrics.execution_time for n in critical_path if n in self.nodes),
            "parallelism_factor": self._calculate_parallelism_factor()
        }

        # Optimization suggestions
        if analysis["bottlenecks"]:
            analysis["optimization_suggestions"].append(
                f"Optimize bottleneck nodes: {', '.join(b['node'] for b in analysis['bottlenecks'][:3])}"
            )

        if analysis["execution_patterns"]["parallelism_factor"] < 0.3:
            analysis["optimization_suggestions"].append(
                "Increase parallelism - current workflow is mostly sequential"
            )

        if analysis["resource_usage"]["peak_memory"] > 1000:  # > 1GB
            analysis["optimization_suggestions"].append(
                "Consider memory optimization for high-memory nodes"
            )

        return analysis

    def _calculate_parallelism_factor(self) -> float:
        """Calculate parallelism factor of the workflow."""
        if not self.nodes:
            return 0

        # Count nodes at each level
        levels = {}
        for node_id, node in self.nodes.items():
            level = len(nx.ancestors(self.graph, node_id))
            if level not in levels:
                levels[level] = 0
            levels[level] += 1

        # Calculate average parallelism
        if levels:
            max_parallel = max(levels.values())
            total_levels = len(levels)
            return max_parallel / total_levels if total_levels > 0 else 0

        return 0


# Utility functions for standalone use
def create_sample_workflow() -> AdvancedVisualizationEngine:
    """Create a sample workflow for demonstration."""
    engine = AdvancedVisualizationEngine()

    # Add sample nodes
    nodes = [
        WorkflowNode("1", "Data Ingestion", "input", NodeStatus.COMPLETED,
                    metrics=NodeMetrics(execution_time=5.2, memory_usage=256, cpu_usage=45)),
        WorkflowNode("2", "Validation", "validation", NodeStatus.COMPLETED,
                    dependencies=["1"],
                    metrics=NodeMetrics(execution_time=2.1, memory_usage=128, cpu_usage=30)),
        WorkflowNode("3", "Transform", "processing", NodeStatus.RUNNING,
                    dependencies=["2"],
                    metrics=NodeMetrics(execution_time=8.5, memory_usage=512, cpu_usage=75)),
        WorkflowNode("4", "ML Model", "ml", NodeStatus.PENDING,
                    dependencies=["3"],
                    metrics=NodeMetrics(execution_time=0, memory_usage=0, cpu_usage=0)),
        WorkflowNode("5", "Export", "output", NodeStatus.PENDING,
                    dependencies=["4"],
                    metrics=NodeMetrics(execution_time=0, memory_usage=0, cpu_usage=0))
    ]

    for node in nodes:
        engine.add_node(node)

    # Add edges
    edges = [
        WorkflowEdge("1", "2", "validated"),
        WorkflowEdge("2", "3", "transform"),
        WorkflowEdge("3", "4", "model"),
        WorkflowEdge("4", "5", "export")
    ]

    for edge in edges:
        engine.add_edge(edge)

    return engine