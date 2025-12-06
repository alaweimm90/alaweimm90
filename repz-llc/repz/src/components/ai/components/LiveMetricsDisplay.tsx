import React from 'react';

interface LiveMetrics {
  heartRate?: number;
  calories: number;
  duration: number;
  currentExercise: string;
  setCount: number;
  repCount: number;
}

interface LiveMetricsDisplayProps {
  metrics: LiveMetrics;
  isActive: boolean;
}

export const LiveMetricsDisplay: React.FC<LiveMetricsDisplayProps> = ({ metrics, isActive }) => {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isActive) return null;

  return (
    <div className="space-y-4">
      {/* Live Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-primary">{formatDuration(metrics.duration)}</div>
          <div className="text-xs text-muted-foreground">Duration</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-primary">{Math.round(metrics.calories)}</div>
          <div className="text-xs text-muted-foreground">Calories</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-primary">{metrics.setCount}</div>
          <div className="text-xs text-muted-foreground">Sets</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-primary">{metrics.repCount}</div>
          <div className="text-xs text-muted-foreground">Reps</div>
        </div>
      </div>

      {/* Current Exercise Display */}
      <div className="text-center p-4 bg-accent/50 rounded-lg">
        <h3 className="font-semibold text-lg">{metrics.currentExercise}</h3>
        <p className="text-sm text-muted-foreground">Current Exercise</p>
      </div>
    </div>
  );
};