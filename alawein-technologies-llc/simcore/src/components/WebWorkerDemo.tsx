/**
 * WebWorker Integration Demo Component
 * Demonstrates multi-threaded physics calculations
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Play, Square, BarChart3, Cpu, Activity } from 'lucide-react';
import { useWebWorkerPhysics } from '@/hooks/use-webworker-physics';
import { generateHighSymmetryPath } from '@/lib/graphene-physics';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

interface WebWorkerDemoProps {
  className?: string;
}

export function WebWorkerDemo({ className }: WebWorkerDemoProps) {
  const [workerCount, setWorkerCount] = useState(2);
  const [isRunning, setIsRunning] = useState(false);
  const [stats, setStats] = useState<any>(null);

  const {
    isInitialized,
    useGrapheneBandStructure,
    useMonteCarloSimulation,
    useMatrixDiagonalization,
    getStats,
    scaleWorkers
  } = useWebWorkerPhysics({
    autoInitialize: true,
    maxWorkers: 4,
    onError: (error) => console.error('WebWorker error:', error),
    onProgress: (progress) => console.log('Progress:', progress)
  });

  // Graphene calculation state
  const grapheneCalc = useGrapheneBandStructure();
  
  // Monte Carlo calculation state
  const monteCarloCalc = useMonteCarloSimulation();
  
  // Matrix calculation state
  const matrixCalc = useMatrixDiagonalization();

  // Update stats periodically
  useEffect(() => {
    const interval = setInterval(() => {
      const currentStats = getStats();
      setStats(currentStats);
    }, 1000);

    return () => clearInterval(interval);
  }, [getStats]);

  // Handle worker scaling
  const handleScaleWorkers = (count: number) => {
    setWorkerCount(count);
    scaleWorkers(count);
  };

  // Run graphene band structure calculation
  const runGrapheneCalculation = async () => {
    setIsRunning(true);
    try {
      const { kPoints } = generateHighSymmetryPath(200);
      await grapheneCalc.calculate({
        kPoints,
        parameters: {
          t1: -2.8,
          t2: -0.1,
          onsite: 0,
          strain: { xx: 0, yy: 0, xy: 0 }
        }
      });
    } finally {
      setIsRunning(false);
    }
  };

  // Run Monte Carlo simulation
  const runMonteCarloCalculation = async () => {
    setIsRunning(true);
    try {
      await monteCarloCalc.calculate({
        steps: 10000,
        parameters: {
          temperature: 2.5,
          systemSize: 50,
          coupling: 1.0
        }
      });
    } finally {
      setIsRunning(false);
    }
  };

  // Run matrix diagonalization
  const runMatrixCalculation = async () => {
    setIsRunning(true);
    try {
      // Create a random symmetric matrix
      const size = 100;
      const matrix = Array(size).fill(0).map(() => Array(size).fill(0));
      for (let i = 0; i < size; i++) {
        for (let j = i; j < size; j++) {
          const value = Math.random() * 2 - 1;
          matrix[i][j] = value;
          matrix[j][i] = value;
        }
      }

      await matrixCalc.calculate({ matrix });
    } finally {
      setIsRunning(false);
    }
  };

  if (!isInitialized) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>WebWorker Science Engine</CardTitle>
          <CardDescription>Initializing multi-threaded physics calculations...</CardDescription>
        </CardHeader>
        <CardContent>
          <Progress value={50} className="w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Statistics Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            WebWorker Science Engine Stats
          </CardTitle>
          <CardDescription>
            Real-time performance monitoring of the multi-threaded science engine
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Active Workers</div>
              <div className="text-2xl font-bold">{stats?.totalWorkers || 0}</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Available Workers</div>
              <div className="text-2xl font-bold text-green-600">{stats?.availableWorkers || 0}</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Queued Tasks</div>
              <div className="text-2xl font-bold text-orange-600">{stats?.queuedTasks || 0}</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Completed Tasks</div>
              <div className="text-2xl font-bold text-blue-600">{stats?.completedTasks || 0}</div>
            </div>
          </div>

          <div className="mt-4 flex items-center gap-2">
            <span className="text-sm">Worker Count:</span>
            {[1, 2, 4, 8].map(count => (
              <Button
                key={count}
                size="sm"
                variant={workerCount === count ? "default" : "outline"}
                onClick={() => handleScaleWorkers(count)}
                disabled={isRunning}
              >
                {count}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Calculation Demos */}
      <Tabs defaultValue="graphene" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="graphene">Graphene Bands</TabsTrigger>
          <TabsTrigger value="montecarlo">Monte Carlo</TabsTrigger>
          <TabsTrigger value="matrix">Matrix Diagonalization</TabsTrigger>
        </TabsList>

        <TabsContent value="graphene" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Graphene Band Structure Calculation</CardTitle>
              <CardDescription>
                Calculate electronic band structure using WebWorkers
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <Button
                  onClick={runGrapheneCalculation}
                  disabled={isRunning || grapheneCalc.loading}
                  className="flex items-center gap-2"
                >
                  {grapheneCalc.loading ? <Square className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  {grapheneCalc.loading ? 'Calculating...' : 'Run Calculation'}
                </Button>
                
                {grapheneCalc.computationTime && (
                  <Badge variant="secondary">
                    {grapheneCalc.computationTime.toFixed(2)}ms
                  </Badge>
                )}
              </div>

              {grapheneCalc.loading && (
                <Progress value={grapheneCalc.progress} className="w-full" />
              )}

              {grapheneCalc.error && (
                <div className="text-red-600 text-sm">{grapheneCalc.error}</div>
              )}

              {grapheneCalc.data && (
                <div className="mt-4">
                  <Plot
                    data={[
                      {
                        y: grapheneCalc.data.energyPlus,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Conduction Band',
                        line: PLOT.lineStyles.bands.conduction
                      },
                      {
                        y: grapheneCalc.data.energyMinus,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Valence Band',
                        line: PLOT.lineStyles.bands.valence
                      }
                    ]}
                    layout={
                      PLOT.createLayout(
                        'Graphene Band Structure (WebWorker)',
                        'k-path',
                        'Energy (eV)',
                        { energyRange: PLOT.energyRanges.graphene, showLegend: true, height: 400 }
                      )
                    }
                    config={PLOT.config}
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="montecarlo" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Monte Carlo Simulation</CardTitle>
              <CardDescription>
                Run statistical physics simulations using WebWorkers
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <Button
                  onClick={runMonteCarloCalculation}
                  disabled={isRunning || monteCarloCalc.loading}
                  className="flex items-center gap-2"
                >
                  {monteCarloCalc.loading ? <Square className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  {monteCarloCalc.loading ? 'Simulating...' : 'Run Simulation'}
                </Button>
                
                {monteCarloCalc.computationTime && (
                  <Badge variant="secondary">
                    {monteCarloCalc.computationTime.toFixed(2)}ms
                  </Badge>
                )}
              </div>

              {monteCarloCalc.loading && (
                <Progress value={monteCarloCalc.progress} className="w-full" />
              )}

              {monteCarloCalc.error && (
                <div className="text-red-600 text-sm">{monteCarloCalc.error}</div>
              )}

              {monteCarloCalc.data && (
                <div className="text-sm text-muted-foreground">
                  Simulation completed successfully. Average energy: {monteCarloCalc.data.averageEnergy?.toFixed(4)}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="matrix" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Matrix Diagonalization</CardTitle>
              <CardDescription>
                Eigenvalue computation using WebWorkers
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <Button
                  onClick={runMatrixCalculation}
                  disabled={isRunning || matrixCalc.loading}
                  className="flex items-center gap-2"
                >
                  {matrixCalc.loading ? <Square className="h-4 w-4" /> : <Cpu className="h-4 w-4" />}
                  {matrixCalc.loading ? 'Computing...' : 'Diagonalize Matrix (100x100)'}
                </Button>
                
                {matrixCalc.computationTime && (
                  <Badge variant="secondary">
                    {matrixCalc.computationTime.toFixed(2)}ms
                  </Badge>
                )}
              </div>

              {matrixCalc.loading && (
                <Progress value={matrixCalc.progress} className="w-full" />
              )}

              {matrixCalc.error && (
                <div className="text-red-600 text-sm">{matrixCalc.error}</div>
              )}

              {matrixCalc.data && (
                <div className="text-sm text-muted-foreground">
                  Matrix diagonalization completed. Found {matrixCalc.data.eigenvalues?.length} eigenvalues.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default WebWorkerDemo;