import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { StatusChip } from '@/components/ui/status-chip';
import { Progress } from '@/components/ui/progress';
import { useAccessibilityContext } from '@/components/AccessibilityProvider';
import { Play, Pause, Square, RotateCcw, TrendingUp, Activity, Brain, Zap } from 'lucide-react';
import { trackQuantumEvents } from '@/lib/analytics';

export const TrainingDashboard = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [accuracy, setAccuracy] = useState(0.5);
  const [isPaused, setIsPaused] = useState(false);
  
  const { announce } = useAccessibilityContext();

  // Mock training simulation
  useEffect(() => {
    if (!isTraining || isPaused) return;

    const interval = setInterval(() => {
      setEpoch((prev) => prev + 1);
      setLoss((prev) => Math.max(0.01, prev * 0.95 + (Math.random() - 0.5) * 0.05));
      setAccuracy((prev) => Math.min(0.99, prev + 0.01 + (Math.random() - 0.5) * 0.02));
    }, 200);

    return () => clearInterval(interval);
  }, [isTraining, isPaused]);

  const handleStartTraining = () => {
    setIsTraining(true);
    setIsPaused(false);
    setEpoch(0);
    setLoss(1.0);
    setAccuracy(0.5);
    
    announce('Training started');
    trackQuantumEvents.trainingStart('iris', 100);
  };

  const handlePauseTraining = () => {
    setIsPaused(!isPaused);
    announce(isPaused ? 'Training resumed' : 'Training paused');
    trackQuantumEvents.trainingStop('iris', epoch, loss);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setIsPaused(false);
    
    announce(`Training stopped at epoch ${epoch}`);
    trackQuantumEvents.trainingStop('iris', epoch, loss);
  };

  const handleReset = () => {
    setIsTraining(false);
    setIsPaused(false);
    setEpoch(0);
    setLoss(1.0);
    setAccuracy(0.5);
    
    announce('Training reset');
    trackQuantumEvents.trainingStop('iris', epoch, loss);
  };

  // Generate mock loss curve data
  const generateLossData = () => {
    const points = [];
    for (let i = 0; i <= epoch && i <= 100; i += 2) {
      const y = Math.max(0.01, 1.0 * Math.exp(-i * 0.05) + Math.random() * 0.1);
      points.push(`${i * 2},${60 - y * 50}`);
    }
    return points.join(" ");
  };

  return (
    <Card className="relative rounded-2xl border border-state-pure/20 bg-surface-1/50 backdrop-blur-sm shadow-high hover:border-state-pure/40 transition-all duration-300">
      {/* State accent bar */}
      <div className="absolute left-0 top-0 bottom-0 w-1 bg-state-pure rounded-r-full"></div>
      
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-accent rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <CardTitle className="text-h4 font-semibold text-text">Training Dashboard</CardTitle>
              <p className="text-small text-muted">Monitor your quantum model training</p>
            </div>
          </div>
          <div className="flex gap-2">
            <StatusChip 
              variant={isTraining ? (isPaused ? "idle" : "running") : "idle"}
              icon={isTraining ? (isPaused ? <Pause className="w-3 h-3" /> : <Activity className="w-3 h-3" />) : <Brain className="w-3 h-3" />}
            >
              {isTraining ? (isPaused ? "Paused" : "Training") : "Ready"}
            </StatusChip>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Control buttons */}
        <div className="flex gap-3">
          <Button
            onClick={handleStartTraining}
            disabled={isTraining && !isPaused}
            variant={isTraining && !isPaused ? "secondary" : "primary"}
            size="sm"
            className="flex-1"
            aria-label={isPaused ? "Resume quantum model training" : "Start quantum model training"}
            aria-busy={isTraining && !isPaused}
          >
            {isTraining && !isPaused ? (
              <>
                <Activity className="w-4 h-4 mr-2 animate-pulse" aria-hidden="true" />
                <span aria-live="polite">Training in progress...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                {isPaused ? "Resume" : "Start Training"}
              </>
            )}
          </Button>
          
          {isTraining && (
            <Button
              onClick={handlePauseTraining}
              variant="outline"
              size="sm"
              aria-label={isPaused ? "Resume training" : "Pause training"}
            >
              {isPaused ? (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Resume
                </>
              ) : (
                <>
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </>
              )}
            </Button>
          )}
          
          <Button
            onClick={handleReset}
            variant="outline"
            size="sm"
            disabled={isTraining && !isPaused}
            aria-label="Reset training progress and start over"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
        </div>

        {/* Progress */}
        {isTraining && (
          <div className="space-y-2" aria-live="polite" aria-label="Training progress">
            <div className="flex justify-between text-small text-muted">
              <span>Epoch Progress</span>
              <span className="font-mono">{epoch}/100</span>
            </div>
            <Progress 
              value={(epoch / 100) * 100} 
              className="h-2" 
              aria-label={`Training progress: ${epoch} of 100 epochs completed`}
            />
          </div>
        )}

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-surface-2/30 rounded-lg p-3 border border-border">
            <div className="text-small text-muted mb-1">Epoch</div>
            <div className="text-h2 font-mono font-semibold text-text">{epoch}</div>
          </div>
          <div className="bg-surface-2/30 rounded-lg p-3 border border-border">
            <div className="text-small text-muted mb-1">Loss</div>
            <div className="text-h2 font-mono font-semibold text-text">{loss.toFixed(4)}</div>
          </div>
          <div className="bg-surface-2/30 rounded-lg p-3 border border-border">
            <div className="text-small text-muted mb-1">Accuracy</div>
            <div className="text-h2 font-mono font-semibold text-text">{(accuracy * 100).toFixed(1)}%</div>
          </div>
        </div>

        {/* Loss curve visualization */}
        <div className="bg-surface-2/30 rounded-lg p-4 border border-border">
          <h4 className="text-small font-medium text-text mb-3">Loss Curve</h4>
          <div className="h-24 relative">
            <svg 
              viewBox="0 0 200 60" 
              className="w-full h-full"
              role="img"
              aria-label="Training loss curve showing convergence over epochs"
            >
              {/* Grid */}
              <defs>
                <pattern id="grid" width="20" height="12" patternUnits="userSpaceOnUse">
                  <path d="M 20 0 L 0 0 0 12" fill="none" stroke="hsl(var(--border))" strokeWidth="0.2" opacity="0.3"/>
                </pattern>
              </defs>
              <rect width="200" height="60" fill="url(#grid)" />
              
              {/* Loss curve */}
              <polyline
                fill="none"
                stroke="hsl(var(--state-pure))"
                strokeWidth="2"
                points={generateLossData()}
                className="drop-shadow-sm"
              />
              
              {/* Current point */}
              {epoch > 0 && (
                <circle
                  cx={epoch * 2}
                  cy={60 - (1 - loss) * 50}
                  r="3"
                  fill="hsl(var(--state-pure))"
                  className="drop-shadow-md"
                />
              )}
              
              {/* Axes labels */}
              <text x="5" y="55" className="text-xs fill-muted font-mono" fontSize="6">0</text>
              <text x="190" y="55" className="text-xs fill-muted font-mono" fontSize="6">100</text>
              <text x="5" y="10" className="text-xs fill-muted font-mono" fontSize="6">1.0</text>
              <text x="5" y="58" className="text-xs fill-muted font-mono" fontSize="6">0.0</text>
            </svg>
          </div>
        </div>

        {/* Model info */}
        <div className="space-y-3 pt-2 border-t border-border">
          <div className="flex justify-between text-small">
            <span className="text-muted">Dataset:</span>
            <span className="text-text font-mono">Quantum Iris (150 samples)</span>
          </div>
          <div className="flex justify-between text-small">
            <span className="text-muted">Model:</span>
            <span className="text-text font-mono">VQC (4 qubits, 3 layers)</span>
          </div>
          <div className="flex justify-between text-small">
            <span className="text-muted">Optimizer:</span>
            <span className="text-text font-mono">SPSA (lr=0.01)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};