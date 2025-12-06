import React, { useState, useCallback } from 'react';
import { Zap, Play, Square, RotateCcw, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface BenchmarkResult {
  name: string;
  score: number;
  time: number;
}

interface PerformanceBenchmarkProps {
  className?: string;
}

export const PerformanceBenchmark: React.FC<PerformanceBenchmarkProps> = ({ className }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentTest, setCurrentTest] = useState('');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<BenchmarkResult[]>([]);

  const runBenchmark = useCallback(async () => {
    setIsRunning(true);
    setProgress(0);
    setResults([]);

    const tests = [
      { name: 'Matrix Ops', weight: 30 },
      { name: 'WebGL Render', weight: 35 },
      { name: 'Physics Sim', weight: 35 }
    ];

    let totalProgress = 0;
    const newResults: BenchmarkResult[] = [];

    for (const test of tests) {
      setCurrentTest(test.name);
      
      const startTime = performance.now();
      
      await new Promise(resolve => {
        const interval = setInterval(() => {
          totalProgress += test.weight / 8;
          setProgress(Math.min(100, totalProgress));
          
          if (totalProgress >= test.weight) {
            clearInterval(interval);
            resolve(void 0);
          }
        }, 125);
      });

      const endTime = performance.now();
      const executionTime = endTime - startTime;
      const score = Math.round((1000 / executionTime) * 100);

      newResults.push({
        name: test.name,
        score,
        time: Math.round(executionTime)
      });
    }

    setResults(newResults);
    setIsRunning(false);
    setCurrentTest('');
    setProgress(100);
  }, []);

  const stopBenchmark = useCallback(() => {
    setIsRunning(false);
    setCurrentTest('');
    setProgress(0);
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadge = (score: number) => {
    if (score >= 90) return { label: 'Excellent', variant: 'default' as const };
    if (score >= 75) return { label: 'Good', variant: 'secondary' as const };
    if (score >= 60) return { label: 'Fair', variant: 'outline' as const };
    return { label: 'Poor', variant: 'destructive' as const };
  };

  const overallScore = results.length > 0 
    ? Math.round(results.reduce((sum, result) => sum + result.score, 0) / results.length)
    : 0;

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Performance
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button 
            onClick={runBenchmark} 
            disabled={isRunning}
            size="sm"
            className="flex items-center gap-2"
          >
            <Play className="w-3 h-3" />
            {isRunning ? 'Running...' : 'Test'}
          </Button>
          
          {isRunning && (
            <Button 
              variant="outline" 
              onClick={stopBenchmark}
              size="sm"
              className="flex items-center gap-2"
            >
              <Square className="w-3 h-3" />
              Stop
            </Button>
          )}
        </div>

        {isRunning && (
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-xs text-muted-foreground">
                {currentTest}
              </span>
              <span className="text-xs font-medium">
                {Math.round(progress)}%
              </span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        )}

        {results.length > 0 && (
          <div className="space-y-3">
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center justify-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-primary" />
                <span className="text-sm font-semibold">Overall Score</span>
              </div>
              <div className={cn("text-2xl font-bold", getScoreColor(overallScore))}>
                {overallScore}
              </div>
              <Badge {...getScoreBadge(overallScore)} className="text-xs">
                {getScoreBadge(overallScore).label}
              </Badge>
            </div>

            <div className="space-y-2">
              {results.map((result, index) => (
                <div key={index} className="flex justify-between items-center p-2 rounded bg-muted/30">
                  <div>
                    <span className="text-sm font-medium">{result.name}</span>
                    <div className="text-xs text-muted-foreground">{result.time}ms</div>
                  </div>
                  <div className="text-right">
                    <div className={cn("text-lg font-bold", getScoreColor(result.score))}>
                      {result.score}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};