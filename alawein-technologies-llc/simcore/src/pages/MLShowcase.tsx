import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { useNavigate } from 'react-router-dom';
import { ScientificPlot } from '@/components/ScientificPlot';
import { MLAnalysis } from '@/components/MLAnalysis';
import { 
  Atom, 
  BarChart3, 
  Brain, 
  Zap, 
  TrendingUp,
  Target,
  Layers,
  Activity,
  ChevronRight,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useSEO } from '@/hooks/use-seo';

const MLShowcase = () => {
  useSEO({ title: 'ML-Enhanced Physics – SimCore', description: 'Machine learning integration for quantum simulations and materials science.' });

  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'ML-Enhanced Physics',
    description: 'Machine learning integration for quantum simulations and materials science.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'machine learning, physics, simulations',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const navigate = useNavigate();
  
  // Demo parameters
  const [demoRunning, setDemoRunning] = useState(false);
  const [selectedDemo, setSelectedDemo] = useState('quantum_tunneling');
  const [trainingEpochs, setTrainingEpochs] = useState(50);
  const [learningRate, setLearningRate] = useState(0.01);
  const [noiseLevel, setNoiseLevel] = useState(0.1);
  const [currentEpoch, setCurrentEpoch] = useState(0);

  // ML Demo Data Generation
  const mlDemoData = useMemo(() => {
    const generateQuantumTunneling = () => {
      const energies = Array.from({ length: 100 }, (_, i) => i * 0.1);
      const actualTransmission = energies.map(E => {
        const barrier = 5.0;
        const width = 1.0;
        if (E > barrier) return 1.0;
        const k = Math.sqrt(2 * (barrier - E));
        return 1 / (1 + Math.sinh(k * width) ** 2);
      });
      
      // Add noise for training data
      const noisyData = actualTransmission.map(t => 
        Math.max(0, Math.min(1, t + (Math.random() - 0.5) * noiseLevel))
      );
      
      // ML prediction (simulated improvement over epochs)
      const convergence = Math.min(1, currentEpoch / trainingEpochs);
      const mlPrediction = actualTransmission.map((actual, i) => {
        const noise = noisyData[i] - actual;
        return actual + noise * (1 - convergence);
      });

      return {
        type: 'quantum_tunneling' as const,
        xData: energies,
        actual: actualTransmission,
        experimental: noisyData,
        predicted: mlPrediction,
        error: actualTransmission.map((a, i) => Math.abs(a - mlPrediction[i]))
      };
    };

    const generateBandStructure = () => {
      const kPoints = Array.from({ length: 100 }, (_, i) => (i - 50) * 0.1);
      const actualBands = kPoints.map(k => Math.sqrt(1 + k * k));
      
      // Simulate experimental data with gaps/noise
      const experimentalData = actualBands.map((band, i) => {
        const noise = (Math.random() - 0.5) * noiseLevel;
        const gap = Math.random() < 0.05 ? NaN : 0; // 5% missing data
        return isNaN(gap) ? NaN : band + noise;
      });
      
      // ML interpolation
      const convergence = Math.min(1, currentEpoch / trainingEpochs);
      const mlPrediction = experimentalData.map((exp, i) => {
        if (isNaN(exp)) {
          // Interpolate missing data
          return actualBands[i] * convergence + (actualBands[i] * 0.8) * (1 - convergence);
        }
        return exp * (1 - convergence) + actualBands[i] * convergence;
      });

      return {
        type: 'band_structure' as const,
        xData: kPoints,
        actual: actualBands,
        experimental: experimentalData,
        predicted: mlPrediction,
        error: actualBands.map((a, i) => Math.abs(a - mlPrediction[i]))
      };
    };

    switch (selectedDemo) {
      case 'quantum_tunneling':
        return generateQuantumTunneling();
      case 'band_structure':
        return generateBandStructure();
      default:
        return generateQuantumTunneling();
    }
  }, [selectedDemo, noiseLevel, currentEpoch, trainingEpochs]);

  // Training simulation
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (demoRunning && currentEpoch < trainingEpochs) {
      interval = setInterval(() => {
        setCurrentEpoch(prev => Math.min(prev + 1, trainingEpochs));
      }, 100);
    } else if (currentEpoch >= trainingEpochs) {
      setDemoRunning(false);
    }
    return () => clearInterval(interval);
  }, [demoRunning, currentEpoch, trainingEpochs]);

  const handleStartDemo = () => {
    setCurrentEpoch(0);
    setDemoRunning(true);
  };

  const handleResetDemo = () => {
    setDemoRunning(false);
    setCurrentEpoch(0);
  };

  const currentError = useMemo(() => {
    if (!mlDemoData.error) return 0;
    return mlDemoData.error.reduce((sum, err) => sum + err, 0) / mlDemoData.error.length;
  }, [mlDemoData]);

  // Plot data for current demo
  const plotData = useMemo(() => {
    const xData = mlDemoData.xData;
    const xLabel = selectedDemo === 'quantum_tunneling' ? 'Energy (eV)' : 'k-vector (1/Å)';
    const yLabel = selectedDemo === 'quantum_tunneling' ? 'Transmission Probability' : 'Energy (eV)';
    
    const datasets = [
      {
        x: xData,
        y: mlDemoData.actual,
        label: 'Theoretical',
        color: '#10b981',
        lineStyle: 'solid',
        lineWidth: 3
      },
      {
        x: xData,
        y: mlDemoData.experimental,
        label: selectedDemo === 'quantum_tunneling' ? 'Noisy Data' : 'Experimental',
        color: '#ef4444',
        mode: 'markers',
        opacity: 0.6
      },
      {
        x: xData,
        y: mlDemoData.predicted,
        label: 'ML Prediction',
        color: '#3b82f6',
        lineStyle: 'dashed',
        lineWidth: 2
      }
    ];

    return {
      datasets,
      xLabel,
      yLabel,
      title: selectedDemo === 'quantum_tunneling' ? 'Quantum Tunneling ML Demo' : 'Band Structure ML Demo'
    };
  }, [mlDemoData, selectedDemo]);

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="ML-Enhanced Physics Analysis"
        description="Advanced machine learning applications for quantum simulations and materials science"
        category="Machine Learning"
        difficulty="Advanced"
        equation="\\hat{f}(x) = \\sum_{i=1}^{n} \\alpha_i K(x, x_i) + \\epsilon"
      />
      
      <div className="max-w-7xl mx-auto space-y-6">
        {/* ML Capabilities Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Brain className="w-4 h-4 text-blue-400" />
                Neural Network Regression
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground mb-2">
                Deep learning models for function approximation and parameter optimization
              </div>
              <Badge variant="default" className="bg-blue-500/20 text-blue-300">Active</Badge>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-400" />
                Pattern Recognition
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground mb-2">
                Automated detection of phase transitions and critical phenomena
              </div>
              <Badge variant="default" className="bg-green-500/20 text-green-300">Active</Badge>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-purple-500/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Target className="w-4 h-4 text-purple-400" />
                Inverse Design
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground mb-2">
                Design materials with desired properties using ML-guided optimization
              </div>
              <Badge variant="default" className="bg-purple-500/20 text-purple-300">Active</Badge>
            </CardContent>
          </Card>
        </div>

        {/* Interactive ML Demo */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Interactive ML Training Demo
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={selectedDemo} onValueChange={setSelectedDemo} className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="quantum_tunneling">Quantum Tunneling</TabsTrigger>
                <TabsTrigger value="band_structure">Band Structure</TabsTrigger>
              </TabsList>

              <TabsContent value="quantum_tunneling" className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Train a neural network to predict quantum tunneling transmission coefficients from noisy experimental data.
                </div>
              </TabsContent>

              <TabsContent value="band_structure" className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Use ML to interpolate missing experimental band structure data and reduce measurement noise.
                </div>
              </TabsContent>
            </Tabs>

            {/* Training Controls */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 my-6 p-4 bg-muted/30 rounded-lg">
              <div className="space-y-2">
                <Label>Training Epochs: {trainingEpochs}</Label>
                <Slider
                  value={[trainingEpochs]}
                  onValueChange={([value]) => setTrainingEpochs(value)}
                  min={10}
                  max={200}
                  step={10}
                  disabled={demoRunning}
                />
              </div>
              
              <div className="space-y-2">
                <Label>Learning Rate: {learningRate}</Label>
                <Slider
                  value={[learningRate * 100]}
                  onValueChange={([value]) => setLearningRate(value / 100)}
                  min={0.1}
                  max={10}
                  step={0.1}
                  disabled={demoRunning}
                />
              </div>
              
              <div className="space-y-2">
                <Label>Noise Level: {noiseLevel.toFixed(2)}</Label>
                <Slider
                  value={[noiseLevel * 100]}
                  onValueChange={([value]) => setNoiseLevel(value / 100)}
                  min={1}
                  max={50}
                  step={1}
                  disabled={demoRunning}
                />
              </div>
              
              <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                  <Button
                    onClick={demoRunning ? () => setDemoRunning(false) : handleStartDemo}
                    size="sm"
                    className="flex-1"
                  >
                    {demoRunning ? (
                      <>
                        <Pause className="w-4 h-4 mr-1" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-1" />
                        Train
                      </>
                    )}
                  </Button>
                  <Button onClick={handleResetDemo} size="sm" variant="outline">
                    <RotateCcw className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Training Progress */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <Card className="bg-card/50">
                <CardContent className="p-4">
                  <div className="text-2xl font-bold text-primary">
                    {currentEpoch}/{trainingEpochs}
                  </div>
                  <div className="text-sm text-muted-foreground">Training Epochs</div>
                </CardContent>
              </Card>
              
              <Card className="bg-card/50">
                <CardContent className="p-4">
                  <div className="text-2xl font-bold text-accent">
                    {(currentError * 100).toFixed(2)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Average Error</div>
                </CardContent>
              </Card>
              
              <Card className="bg-card/50">
                <CardContent className="p-4">
                  <div className="text-2xl font-bold text-primary-glow">
                    {((1 - currentError) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Accuracy</div>
                </CardContent>
              </Card>
            </div>

            {/* ML Demo Visualization */}
            <div className="h-96 mb-6">
              <ScientificPlot
                title={plotData.title}
                data={plotData}
                plotType="2d"
                xLabel={plotData.xLabel}
                yLabel={plotData.yLabel}
                showGrid={true}
                showLegend={true}
              />
            </div>
          </CardContent>
        </Card>

        {/* Advanced ML Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Advanced ML Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <MLAnalysis
              data={[
                [trainingEpochs, learningRate, noiseLevel, currentError],
                [trainingEpochs * 0.8, learningRate * 1.2, noiseLevel * 0.9, currentError * 1.1],
                [trainingEpochs * 1.1, learningRate * 0.9, noiseLevel * 1.2, currentError * 0.9]
              ]}
              parameters={{
                epochs: trainingEpochs,
                learningRate: learningRate,
                noiseLevel: noiseLevel,
                currentEpoch: currentEpoch
              }}
              simulationType="ml_training_demo"
            />
          </CardContent>
        </Card>

        {/* Real-world Applications */}
        <Card>
          <CardHeader>
            <CardTitle>Real-world ML Applications in Physics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  title: "Drug Discovery",
                  description: "ML-assisted molecular dynamics for pharmaceutical development",
                  icon: Atom,
                  color: "text-blue-400"
                },
                {
                  title: "Quantum Computing",
                  description: "Error correction and quantum state optimization",
                  icon: Zap,
                  color: "text-purple-400"
                },
                {
                  title: "Materials Design",
                  description: "Predicting material properties for next-gen technologies",
                  icon: Layers,
                  color: "text-green-400"
                },
                {
                  title: "Climate Modeling",
                  description: "Enhanced weather prediction and climate simulation",
                  icon: TrendingUp,
                  color: "text-orange-400"
                }
              ].map((app, index) => (
                <Card key={index} className="bg-card/30 hover:bg-card/50 transition-colors">
                  <CardContent className="p-4 flex items-start gap-3">
                    <app.icon className={`w-6 h-6 ${app.color} flex-shrink-0 mt-1`} />
                    <div>
                      <h4 className="font-semibold mb-1">{app.title}</h4>
                      <p className="text-sm text-muted-foreground">{app.description}</p>
                    </div>
                    <ChevronRight className="w-4 h-4 text-muted-foreground ml-auto flex-shrink-0 mt-1" />
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </PhysicsModuleLayout>
  );
};

export default MLShowcase;