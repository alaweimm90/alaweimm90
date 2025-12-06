import React from 'react';
import Plot from 'react-plotly.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, BarChart3, Activity, Zap, Target, GitBranch } from 'lucide-react';
import { useIsingStore } from './IsingSimulationStore';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

interface IsingPlotsProps {
  height?: number;
}

export const IsingPlots: React.FC<IsingPlotsProps> = ({ height = 400 }) => {
  const {
    magnetizationHistory,
    energyHistory,
    heatCapacityHistory,
    susceptibilityHistory,
    magnetizationHistogram,
    phaseTransitionData,
    correlationData,
    temperature,
    step
  } = useIsingStore();

  const plotConfig = {
    ...PLOT.config,
    displayModeBar: false,
  } as const;

  const plotLayout = {
    ...PLOT.layout,
    height,
    margin: { l: 60, r: 30, t: 50, b: 60 },
    font: PLOT.layout.font,
  xaxis: {
      ...PLOT.layout.xaxis
    },
    yaxis: {
      ...PLOT.layout.yaxis
    }
  } as const;
  return (
    <div className="space-y-6">
      <Tabs defaultValue="timeseries" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="timeseries" className="flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Time Series
          </TabsTrigger>
          <TabsTrigger value="phase" className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Phase Transition
          </TabsTrigger>
          <TabsTrigger value="histogram" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Distributions
          </TabsTrigger>
          <TabsTrigger value="correlation" className="flex items-center gap-2">
            <GitBranch className="w-4 h-4" />
            Correlations
          </TabsTrigger>
        </TabsList>

        {/* Time Series Plots */}
        <TabsContent value="timeseries" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Magnetization vs Time */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Target className="w-5 h-5 text-blue-400" />
                  Magnetization Evolution
                </CardTitle>
                <div className="text-xs text-muted-foreground">
                  Order parameter: ⟨m⟩ = ⟨Σσᵢ⟩/N
                </div>
              </CardHeader>
              <CardContent>
                <Plot
                  data={[{
                    x: Array.from({ length: magnetizationHistory.length }, (_, i) => i),
                    y: magnetizationHistory,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#3b82f6', width: 2 },
                    name: 'Magnetization'
                  }]}
                  layout={{
                    ...plotLayout,
                    title: { text: '', font: { size: 14 } },
                    xaxis: { 
                      ...plotLayout.xaxis,
                      title: { text: 'Monte Carlo Steps', font: { size: 12 } },
                      range: [Math.max(0, step - 1000), step]
                    },
                    yaxis: { 
                      ...plotLayout.yaxis,
                      title: { text: 'Magnetization ⟨m⟩', font: { size: 12 } },
                      range: [-1.1, 1.1]
                    }
                  }}
                  config={plotConfig}
                />
              </CardContent>
            </Card>

            {/* Energy vs Time */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Zap className="w-5 h-5 text-red-400" />
                  Energy Evolution
                </CardTitle>
                <div className="text-xs text-muted-foreground">
                  Energy per spin: E/N = -J⟨σᵢσⱼ⟩ - h⟨σᵢ⟩
                </div>
              </CardHeader>
              <CardContent>
                <Plot
                  data={[{
                    x: Array.from({ length: energyHistory.length }, (_, i) => i),
                    y: energyHistory,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#ef4444', width: 2 },
                    name: 'Energy'
                  }]}
                  layout={{
                    ...plotLayout,
                    xaxis: { 
                      ...plotLayout.xaxis,
                      title: { text: 'Monte Carlo Steps', font: { size: 12 } },
                      range: [Math.max(0, step - 1000), step]
                    },
                    yaxis: { 
                      ...plotLayout.yaxis,
                      title: { text: 'Energy per spin', font: { size: 12 } }
                    }
                  }}
                  config={plotConfig}
                />
              </CardContent>
            </Card>

            {/* Heat Capacity vs Time */}
            {heatCapacityHistory.length > 0 && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Activity className="w-5 h-5 text-orange-400" />
                    Heat Capacity
                  </CardTitle>
                  <div className="text-xs text-muted-foreground">
                    Cᵥ = (⟨E²⟩ - ⟨E⟩²)/T²
                  </div>
                </CardHeader>
                <CardContent>
                  <Plot
                    data={[{
                      x: Array.from({ length: heatCapacityHistory.length }, (_, i) => i),
                      y: heatCapacityHistory,
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#f97316', width: 2 },
                      name: 'Heat Capacity'
                    }]}
                    layout={{
                      ...plotLayout,
                      xaxis: { 
                        ...plotLayout.xaxis,
                        title: { text: 'Monte Carlo Steps', font: { size: 12 } }
                      },
                      yaxis: { 
                        ...plotLayout.yaxis,
                        title: { text: 'Heat Capacity Cᵥ', font: { size: 12 } }
                      }
                    }}
                    config={plotConfig}
                  />
                </CardContent>
              </Card>
            )}

            {/* Susceptibility vs Time */}
            {susceptibilityHistory.length > 0 && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Target className="w-5 h-5 text-green-400" />
                    Magnetic Susceptibility
                  </CardTitle>
                  <div className="text-xs text-muted-foreground">
                    χ = (⟨M²⟩ - ⟨M⟩²)/T
                  </div>
                </CardHeader>
                <CardContent>
                  <Plot
                    data={[{
                      x: Array.from({ length: susceptibilityHistory.length }, (_, i) => i),
                      y: susceptibilityHistory,
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#22c55e', width: 2 },
                      name: 'Susceptibility'
                    }]}
                    layout={{
                      ...plotLayout,
                      xaxis: { 
                        ...plotLayout.xaxis,
                        title: { text: 'Monte Carlo Steps', font: { size: 12 } }
                      },
                      yaxis: { 
                        ...plotLayout.yaxis,
                        title: { text: 'Susceptibility χ', font: { size: 12 } }
                      }
                    }}
                    config={plotConfig}
                  />
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* Phase Transition Analysis */}
        <TabsContent value="phase" className="space-y-6">
          {phaseTransitionData ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Order Parameter vs Temperature</CardTitle>
                  <div className="text-xs text-muted-foreground">
                    Spontaneous magnetization below Tc ≈ 2.269
                  </div>
                </CardHeader>
                <CardContent>
                  <Plot
                    data={[
                      {
                        x: phaseTransitionData.temperatures,
                        y: phaseTransitionData.magnetizations,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#3b82f6', width: 3 },
                        marker: { size: 6, color: '#1d4ed8' },
                        name: '⟨|m|⟩'
                      },
                      {
                        x: [2.269, 2.269],
                        y: [0, 1],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'Tc (exact)'
                      }
                    ]}
                    layout={{
                      ...plotLayout,
                      xaxis: { 
                        ...plotLayout.xaxis,
                        title: { text: 'Temperature T', font: { size: 12 } }
                      },
                      yaxis: { 
                        ...plotLayout.yaxis,
                        title: { text: 'Magnetization ⟨|m|⟩', font: { size: 12 } },
                        range: [0, 1]
                      },
                      showlegend: true,
                      legend: { x: 0.7, y: 0.9 }
                    }}
                    config={plotConfig}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Heat Capacity vs Temperature</CardTitle>
                  <div className="text-xs text-muted-foreground">
                    Peak at critical temperature indicates phase transition
                  </div>
                </CardHeader>
                <CardContent>
                  <Plot
                    data={[
                      {
                        x: phaseTransitionData.temperatures,
                        y: phaseTransitionData.heatCapacities,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#f97316', width: 3 },
                        marker: { size: 6, color: '#ea580c' },
                        name: 'Cᵥ'
                      },
                      {
                        x: [2.269, 2.269],
                        y: [0, Math.max(...phaseTransitionData.heatCapacities)],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'Tc'
                      }
                    ]}
                    layout={{
                      ...plotLayout,
                      xaxis: { 
                        ...plotLayout.xaxis,
                        title: { text: 'Temperature T', font: { size: 12 } }
                      },
                      yaxis: { 
                        ...plotLayout.yaxis,
                        title: { text: 'Heat Capacity Cᵥ', font: { size: 12 } }
                      },
                      showlegend: true,
                      legend: { x: 0.7, y: 0.9 }
                    }}
                    config={plotConfig}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Magnetic Susceptibility vs Temperature</CardTitle>
                  <div className="text-xs text-muted-foreground">
                    Diverges at critical temperature (finite-size peak)
                  </div>
                </CardHeader>
                <CardContent>
                  <Plot
                    data={[
                      {
                        x: phaseTransitionData.temperatures,
                        y: phaseTransitionData.susceptibilities,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#22c55e', width: 3 },
                        marker: { size: 6, color: '#16a34a' },
                        name: 'χ'
                      },
                      {
                        x: [2.269, 2.269],
                        y: [0, Math.max(...phaseTransitionData.susceptibilities)],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'Tc'
                      }
                    ]}
                    layout={{
                      ...plotLayout,
                      xaxis: { 
                        ...plotLayout.xaxis,
                        title: { text: 'Temperature T', font: { size: 12 } }
                      },
                      yaxis: { 
                        ...plotLayout.yaxis,
                        title: { text: 'Susceptibility χ', font: { size: 12 } }
                      },
                      showlegend: true,
                      legend: { x: 0.7, y: 0.9 }
                    }}
                    config={plotConfig}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Critical Exponents</CardTitle>
                  <div className="text-xs text-muted-foreground">
                    Power law behavior near Tc
                  </div>
                </CardHeader>
                <CardContent className="flex items-center justify-center">
                  <div className="text-center space-y-2">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <Badge variant="outline">β ≈ 1/8 (magnetization)</Badge>
                      <Badge variant="outline">γ ≈ 7/4 (susceptibility)</Badge>
                      <Badge variant="outline">α ≈ 0 (heat capacity)</Badge>
                      <Badge variant="outline">ν = 1 (correlation length)</Badge>
                    </div>
                    <div className="text-xs text-muted-foreground mt-4">
                      Exact 2D Ising model critical exponents
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center h-64">
                <div className="text-center text-muted-foreground">
                  <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Run phase transition analysis to see critical behavior</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Magnetization Histogram */}
        <TabsContent value="histogram" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Magnetization Distribution</CardTitle>
              <div className="text-xs text-muted-foreground">
                Bimodal distribution below Tc shows spontaneous symmetry breaking
              </div>
            </CardHeader>
            <CardContent>
              {magnetizationHistogram.bins.length > 0 ? (
                <Plot
                  data={[{
                    x: magnetizationHistogram.bins,
                    y: magnetizationHistogram.counts,
                    type: 'bar',
                    marker: { 
                      color: '#8b5cf6',
                      opacity: 0.7,
                      line: { color: '#7c3aed', width: 1 }
                    },
                    name: 'P(M)'
                  }]}
                  layout={{
                    ...plotLayout,
                    xaxis: { 
                      ...plotLayout.xaxis,
                      title: { text: 'Magnetization M', font: { size: 12 } }
                    },
                    yaxis: { 
                      ...plotLayout.yaxis,
                      title: { text: 'Probability Density P(M)', font: { size: 12 } }
                    },
                    annotations: [{
                      x: 0,
                      y: Math.max(...magnetizationHistogram.counts) * 0.8,
                      text: `T = ${temperature.toFixed(3)}`,
                      showarrow: false,
                      font: { size: 14, color: '#ffffff' },
                      bgcolor: 'rgba(0,0,0,0.5)',
                      bordercolor: '#ffffff',
                      borderwidth: 1
                    }]
                  }}
                  config={plotConfig}
                />
              ) : (
                <div className="flex items-center justify-center h-96 text-muted-foreground">
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Run simulation to generate magnetization histogram</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Correlation Analysis */}
        <TabsContent value="correlation" className="space-y-6">
          {correlationData ? (
            <Card>
              <CardHeader>
                <CardTitle>Spin Correlation Function</CardTitle>
                <div className="text-xs text-muted-foreground">
                  ⟨σ(0)σ(r)⟩ vs distance - exponential decay with correlation length ξ
                </div>
              </CardHeader>
              <CardContent>
                <Plot
                  data={[{
                    x: correlationData.distances,
                    y: correlationData.correlations,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#06b6d4', width: 3 },
                    marker: { size: 6, color: '#0891b2' },
                    name: 'C(r)'
                  }]}
                  layout={{
                    ...plotLayout,
                    xaxis: { 
                      ...plotLayout.xaxis,
                      title: { text: 'Distance r', font: { size: 12 } }
                    },
                    yaxis: { 
                      ...plotLayout.yaxis,
                      title: { text: 'Correlation ⟨σ(0)σ(r)⟩', font: { size: 12 } }
                    },
                    annotations: [{
                      x: correlationData.distances[Math.floor(correlationData.distances.length / 2)],
                      y: Math.max(...correlationData.correlations) * 0.8,
                      text: `ξ ≈ ${correlationData.correlationLength.toFixed(1)}`,
                      showarrow: true,
                      font: { size: 14, color: '#ffffff' },
                      bgcolor: 'rgba(0,0,0,0.5)',
                      bordercolor: '#ffffff',
                      borderwidth: 1
                    }]
                  }}
                  config={plotConfig}
                />
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center h-64">
                <div className="text-center text-muted-foreground">
                  <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Run simulation to see correlation analysis</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default IsingPlots;