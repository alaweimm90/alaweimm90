import React, { useMemo, useCallback, useState } from 'react';
import Plot from 'react-plotly.js';
import { PlotParams } from 'react-plotly.js';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Download, Camera, RotateCcw, Settings, Maximize2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

export interface PlotData {
  x?: number[];
  y?: number[];
  z?: number[];
  values?: number[][];
  type: 'scatter' | 'heatmap' | 'surface' | 'contour' | 'scatter3d' | 'mesh3d' | 'line';
  name?: string;
  mode?: 'lines' | 'markers' | 'lines+markers';
  line?: {
    color?: string;
    width?: number;
    dash?: string;
  };
  marker?: {
    color?: string | number[];
    size?: number | number[];
    colorscale?: string;
    showscale?: boolean;
    colorbar?: any;
  };
  colorscale?: string;
  showscale?: boolean;
  opacity?: number;
  hovertemplate?: string;
  customdata?: any[];
}

export interface AdvancedPlotProps {
  data: PlotData[];
  title?: string;
  subtitle?: string;
  xLabel?: string;
  yLabel?: string;
  zLabel?: string;
  xUnit?: string;
  yUnit?: string;
  zUnit?: string;
  layout?: any;
  config?: any;
  className?: string;
  height?: number;
  width?: number;
  theme?: 'light' | 'dark';
  annotations?: Array<{
    x: number;
    y: number;
    z?: number;
    text: string;
    color?: string;
  }>;
  onExport?: () => void;
  exportData?: any;
  interactive?: boolean;
  autoResize?: boolean;
}

export const AdvancedPlot: React.FC<AdvancedPlotProps> = ({
  data,
  title,
  subtitle,
  xLabel,
  yLabel,
  zLabel,
  xUnit,
  yUnit,
  zUnit,
  layout: customLayout = {},
  config: customConfig = {},
  className,
  height = 500,
  width,
  theme = 'dark',
  annotations = [],
  onExport,
  exportData,
  interactive = true,
  autoResize = true,
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [plotKey, setPlotKey] = useState(0);

  // Debug plotting data - only once
  if (!data || data.length === 0) {
    console.log('AdvancedPlot: No data received');
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <p>No plot data available</p>
      </div>
    );
  }

  // Convert our data format to Plotly format
  const plotlyData = useMemo(() => {
    return data.map((item, index) => {
      const baseTrace: any = {
        name: item.name || `Trace ${index + 1}`,
        type: item.type,
        hovertemplate: item.hovertemplate,
        customdata: item.customdata,
      };

      switch (item.type) {
        case 'scatter':
        case 'line':
          return {
            ...baseTrace,
            x: item.x,
            y: item.y,
            mode: item.mode || 'lines',
            line: {
              color: item.line?.color || (theme === 'dark' ? '#60A5FA' : '#3B82F6'),
              width: item.line?.width || 2,
              dash: item.line?.dash,
            },
            marker: item.marker ? {
              color: item.marker.color,
              size: item.marker.size || 6,
            } : undefined,
          };
        
        case 'scatter3d':
          return {
            ...baseTrace,
            x: item.x,
            y: item.y,
            z: item.z,
            mode: item.mode || 'markers',
            marker: {
              color: item.marker?.color || item.z,
              colorscale: item.marker?.colorscale || 'Viridis',
              size: item.marker?.size || 3,
              showscale: item.marker?.showscale !== false,
              colorbar: item.marker?.colorbar || {
                title: zLabel ? `${zLabel}${zUnit ? ` (${zUnit})` : ''}` : '',
                titleside: 'right',
              },
            },
          };
        
        case 'heatmap':
          return {
            ...baseTrace,
            z: item.values,
            x: item.x,
            y: item.y,
            colorscale: item.colorscale || 'Viridis',
            showscale: item.showscale !== false,
            colorbar: {
              title: zLabel ? `${zLabel}${zUnit ? ` (${zUnit})` : ''}` : '',
              titleside: 'right',
            },
          };
        
        case 'surface':
          return {
            ...baseTrace,
            z: item.values || item.z,
            x: item.x,
            y: item.y,
            colorscale: item.colorscale || 'Viridis',
            showscale: item.showscale !== false,
            opacity: item.opacity || 1,
            colorbar: {
              title: zLabel ? `${zLabel}${zUnit ? ` (${zUnit})` : ''}` : '',
              titleside: 'right',
            },
          };
        
        case 'contour':
          return {
            ...baseTrace,
            z: item.values || item.z,
            x: item.x,
            y: item.y,
            colorscale: item.colorscale || 'Viridis',
            showscale: item.showscale !== false,
            contours: {
              showlabels: true,
              labelfont: {
                color: theme === 'dark' ? 'white' : 'black',
              },
            },
            colorbar: {
              title: zLabel ? `${zLabel}${zUnit ? ` (${zUnit})` : ''}` : '',
              titleside: 'right',
            },
          };
        
        default:
          return baseTrace;
      }
    });
  }, [data, theme, zLabel, zUnit]);

  // Generate layout with unified configuration
  const layout = useMemo(() => {
    const isDark = theme === 'dark';
    const is3D = data.some(d => d.type === 'scatter3d' || d.type === 'surface' || d.type === 'mesh3d');
    
    const baseLayout: any = {
      ...PLOT.layout,
      title: title ? {
        text: `<b>${title}</b>${subtitle ? `<br><span style="font-size: 14px;">${subtitle}</span>` : ''}`,
        font: PLOT.layout.font,
        x: 0.05,
      } : undefined,
      height: isFullscreen ? window.innerHeight - 100 : height,
      width: isFullscreen ? window.innerWidth - 100 : width,
      autosize: autoResize,
    };

    if (is3D) {
      baseLayout.scene = {
        xaxis: {
          title: xLabel ? `${xLabel}${xUnit ? ` (${xUnit})` : ''}` : '',
          titlefont: PLOT.layout.font,
          tickfont: PLOT.layout.font,
          gridcolor: 'transparent',
          showgrid: false
        },
        yaxis: {
          title: yLabel ? `${yLabel}${yUnit ? ` (${yUnit})` : ''}` : '',
          titlefont: PLOT.layout.font,
          tickfont: PLOT.layout.font,
          gridcolor: 'transparent',
          showgrid: false
        },
        zaxis: {
          title: zLabel ? `${zLabel}${zUnit ? ` (${zUnit})` : ''}` : '',
          titlefont: PLOT.layout.font,
          tickfont: PLOT.layout.font,
          gridcolor: 'transparent',
          showgrid: false
        },
        bgcolor: 'transparent',
        camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } },
      };
    } else {
      baseLayout.xaxis = {
        ...PLOT.layout.xaxis,
        title: xLabel ? `${xLabel}${xUnit ? ` (${xUnit})` : ''}` : ''
      };
      baseLayout.yaxis = {
        ...PLOT.layout.yaxis,
        title: yLabel ? `${yLabel}${yUnit ? ` (${yUnit})` : ''}` : ''
      };
    }

    // Add annotations
    if (annotations.length > 0) {
      baseLayout.annotations = annotations.map(ann => ({
        x: ann.x,
        y: ann.y,
        text: ann.text,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        arrowcolor: ann.color || (isDark ? '#60A5FA' : '#3B82F6'),
        font: {
          color: ann.color || (isDark ? '#F3F4F6' : '#1F2937'),
          size: 12,
        },
        bgcolor: isDark ? 'rgba(30, 41, 59, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        bordercolor: ann.color || (isDark ? '#60A5FA' : '#3B82F6'),
        borderwidth: 1,
      }));
    }

    return { ...baseLayout, ...customLayout };
  }, [theme, title, subtitle, xLabel, yLabel, zLabel, xUnit, yUnit, zUnit, height, width, annotations, customLayout, data, isFullscreen, autoResize]);

  // Generate config with unified settings
  const config = useMemo(() => {
    const baseConfig: any = {
      ...PLOT.config,
      responsive: autoResize,
      displayModeBar: interactive,
      toImageButtonOptions: {
        format: 'png',
        filename: title ? title.replace(/[^a-z0-9]/gi, '_').toLowerCase() : 'plot',
        height: height,
        width: width || 800,
        scale: 2,
      },
    };

    return { ...baseConfig, ...customConfig };
  }, [interactive, autoResize, title, height, width, customConfig]);

  const handleExportData = useCallback(() => {
    if (onExport) {
      onExport();
    } else if (exportData) {
      const dataStr = JSON.stringify(exportData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${title ? title.replace(/[^a-z0-9]/gi, '_').toLowerCase() : 'plot'}_data.json`;
      link.click();
      URL.revokeObjectURL(url);
    }
  }, [onExport, exportData, title]);

  const handleReset = useCallback(() => {
    setPlotKey(prev => prev + 1);
  }, []);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  return (
    <Card className={cn(
      "relative overflow-hidden",
      isFullscreen && "fixed inset-4 z-50",
      className
    )}>
      {/* Header Controls */}
      <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
        {onExport || exportData ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleExportData}
            className="bg-background/80 backdrop-blur-sm"
          >
            <Download className="w-4 h-4" />
          </Button>
        ) : null}
        
        <Button
          variant="ghost"
          size="sm"
          onClick={handleReset}
          className="bg-background/80 backdrop-blur-sm"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
        
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleFullscreen}
          className="bg-background/80 backdrop-blur-sm"
        >
          <Maximize2 className="w-4 h-4" />
        </Button>

        <Badge variant="outline" className="bg-background/80 backdrop-blur-sm">
          {data.length} trace{data.length !== 1 ? 's' : ''}
        </Badge>
      </div>

      {/* Plot */}
      <div className="w-full h-full">
        <Plot
          key={plotKey}
          data={plotlyData}
          layout={layout}
          config={config}
          className="w-full h-full"
          useResizeHandler={autoResize}
        />
      </div>
    </Card>
  );
};

export default AdvancedPlot;