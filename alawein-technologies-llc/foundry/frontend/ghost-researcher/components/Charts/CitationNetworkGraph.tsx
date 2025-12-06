/**
 * Citation Network Graph Component
 * Interactive visualization of paper citation relationships
 */

import React, { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';
import { Card } from '../../../components/Card';
import { Button } from '../../../components/Button';
import { Badge } from '../../../components/Badge';
import { ZoomIn, ZoomOut, Maximize2, Download, Filter } from 'lucide-react';

export interface CitationNode {
  id: string;
  title: string;
  authors: string[];
  year: number;
  citations: number;
  type: 'source' | 'cited' | 'citing';
  abstract?: string;
  doi?: string;
  url?: string;
}

export interface CitationLink {
  source: string;
  target: string;
  strength: number;
  type: 'cites' | 'citedBy';
}

export interface CitationNetworkGraphProps {
  nodes: CitationNode[];
  links: CitationLink[];
  selectedNodeId?: string;
  onNodeClick?: (node: CitationNode) => void;
  onNodeHover?: (node: CitationNode | null) => void;
  className?: string;
  height?: number;
  width?: number;
}

export const CitationNetworkGraph: React.FC<CitationNetworkGraphProps> = ({
  nodes,
  links,
  selectedNodeId,
  onNodeClick,
  onNodeHover,
  className = '',
  height = 600,
  width,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<any>(null);
  const [zoom, setZoom] = React.useState(1);
  const [filterType, setFilterType] = React.useState<'all' | 'cited' | 'citing'>('all');

  // Filter nodes based on type
  const filteredData = useMemo(() => {
    if (filterType === 'all') {
      return { nodes, links };
    }

    const filteredNodes = nodes.filter(
      (node) => node.type === 'source' || node.type === filterType
    );
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredLinks = links.filter(
      (link) => nodeIds.has(link.source) && nodeIds.has(link.target)
    );

    return { nodes: filteredNodes, links: filteredLinks };
  }, [nodes, links, filterType]);

  useEffect(() => {
    if (!svgRef.current || filteredData.nodes.length === 0) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove();

    // Get dimensions
    const svgWidth = width || containerRef.current?.clientWidth || 800;
    const svgHeight = height;

    // Create SVG groups
    const svg = d3.select(svgRef.current);

    // Add zoom behavior
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    const container = svg.append('g');

    // Define arrow markers for directed edges
    svg.append('defs')
      .selectAll('marker')
      .data(['cites', 'citedBy'])
      .join('marker')
      .attr('id', (d) => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', (d) => d === 'cites' ? '#3b82f6' : '#10b981');

    // Create force simulation
    const simulation = d3.forceSimulation<CitationNode>(filteredData.nodes)
      .force('link', d3.forceLink<CitationNode, CitationLink>(filteredData.links)
        .id((d) => d.id)
        .distance((d) => 100 - d.strength * 20))
      .force('charge', d3.forceManyBody().strength(-500))
      .force('center', d3.forceCenter(svgWidth / 2, svgHeight / 2))
      .force('collision', d3.forceCollide().radius((d: any) => getNodeRadius(d) + 10));

    simulationRef.current = simulation;

    // Create links
    const linkGroup = container.append('g').attr('class', 'links');
    const link = linkGroup
      .selectAll('line')
      .data(filteredData.links)
      .join('line')
      .attr('stroke', (d) => d.type === 'cites' ? '#3b82f6' : '#10b981')
      .attr('stroke-opacity', 0.4)
      .attr('stroke-width', (d) => Math.sqrt(d.strength) * 2)
      .attr('marker-end', (d) => `url(#arrow-${d.type})`);

    // Create node groups
    const nodeGroup = container.append('g').attr('class', 'nodes');

    const node = nodeGroup
      .selectAll('g')
      .data(filteredData.nodes)
      .join('g')
      .attr('class', 'node-group')
      .call(d3.drag<any, CitationNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      );

    // Add circles for nodes
    node.append('circle')
      .attr('r', getNodeRadius)
      .attr('fill', getNodeColor)
      .attr('stroke', (d) => d.id === selectedNodeId ? '#1f2937' : '#e5e7eb')
      .attr('stroke-width', (d) => d.id === selectedNodeId ? 3 : 1)
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick?.(d);
      })
      .on('mouseenter', (event, d) => {
        onNodeHover?.(d);
        // Highlight connected nodes and links
        link
          .attr('stroke-opacity', (l: any) =>
            l.source.id === d.id || l.target.id === d.id ? 0.8 : 0.2
          );
        node
          .select('circle')
          .attr('opacity', (n) => {
            if (n.id === d.id) return 1;
            const connected = filteredData.links.some(
              (l: any) =>
                (l.source.id === d.id && l.target.id === n.id) ||
                (l.target.id === d.id && l.source.id === n.id)
            );
            return connected ? 1 : 0.3;
          });
      })
      .on('mouseleave', () => {
        onNodeHover?.(null);
        link.attr('stroke-opacity', 0.4);
        node.select('circle').attr('opacity', 1);
      });

    // Add labels
    node.append('text')
      .text((d) => {
        const title = d.title.length > 20 ? d.title.substring(0, 20) + '...' : d.title;
        return title;
      })
      .attr('x', 0)
      .attr('y', (d) => getNodeRadius(d) + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#4b5563')
      .attr('pointer-events', 'none');

    // Add citation count badges
    node.append('text')
      .text((d) => d.citations > 0 ? d.citations.toString() : '')
      .attr('x', 0)
      .attr('y', 3)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .attr('pointer-events', 'none');

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // Cleanup on unmount
    return () => {
      simulation.stop();
    };
  }, [filteredData, selectedNodeId, onNodeClick, onNodeHover, width, height]);

  // Helper functions
  function getNodeRadius(node: CitationNode): number {
    const basSize = 15;
    const citationBonus = Math.min(Math.sqrt(node.citations) * 2, 20);
    return basSize + citationBonus;
  }

  function getNodeColor(node: CitationNode): string {
    switch (node.type) {
      case 'source':
        return '#3b82f6'; // Blue for source paper
      case 'cited':
        return '#10b981'; // Green for papers cited by source
      case 'citing':
        return '#f59e0b'; // Amber for papers citing source
      default:
        return '#6b7280'; // Gray for unknown
    }
  }

  // Control functions
  const handleZoomIn = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      1.3
    );
  };

  const handleZoomOut = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      0.7
    );
  };

  const handleResetZoom = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().transform as any,
      d3.zoomIdentity
    );
    setZoom(1);
  };

  const handleDownload = () => {
    if (!svgRef.current) return;

    const svgElement = svgRef.current;
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'citation-network.svg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Card className={className}>
      {/* Header Controls */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Citation Network
            </h3>
            <div className="flex items-center space-x-2">
              <Badge variant="primary">{filteredData.nodes.length} Papers</Badge>
              <Badge variant="secondary">{filteredData.links.length} Citations</Badge>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {/* Filter Buttons */}
            <div className="flex items-center space-x-1 mr-2">
              <Button
                size="sm"
                variant={filterType === 'all' ? 'primary' : 'ghost'}
                onClick={() => setFilterType('all')}
              >
                All
              </Button>
              <Button
                size="sm"
                variant={filterType === 'cited' ? 'primary' : 'ghost'}
                onClick={() => setFilterType('cited')}
              >
                Cited
              </Button>
              <Button
                size="sm"
                variant={filterType === 'citing' ? 'primary' : 'ghost'}
                onClick={() => setFilterType('citing')}
              >
                Citing
              </Button>
            </div>

            {/* Zoom Controls */}
            <Button size="sm" variant="ghost" onClick={handleZoomIn}>
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button size="sm" variant="ghost" onClick={handleZoomOut}>
              <ZoomOut className="w-4 h-4" />
            </Button>
            <Button size="sm" variant="ghost" onClick={handleResetZoom}>
              <Maximize2 className="w-4 h-4" />
            </Button>
            <Button size="sm" variant="ghost" onClick={handleDownload}>
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Graph Container */}
      <div ref={containerRef} className="relative">
        <svg
          ref={svgRef}
          width={width || '100%'}
          height={height}
          className="w-full"
          style={{ cursor: 'grab' }}
        />

        {/* Zoom Indicator */}
        <div className="absolute bottom-4 right-4">
          <Badge variant="secondary" size="sm">
            {Math.round(zoom * 100)}%
          </Badge>
        </div>

        {/* Legend */}
        <div className="absolute top-4 left-4 bg-white dark:bg-gray-900 p-3 rounded-lg shadow-lg">
          <p className="text-xs font-semibold mb-2 text-gray-700 dark:text-gray-300">
            Legend
          </p>
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-xs text-gray-600 dark:text-gray-400">Source Paper</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-xs text-gray-600 dark:text-gray-400">Cited Papers</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-amber-500" />
              <span className="text-xs text-gray-600 dark:text-gray-400">Citing Papers</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default CitationNetworkGraph;