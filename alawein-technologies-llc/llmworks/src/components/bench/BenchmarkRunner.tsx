import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { 
  Play, 
  BarChart3, 
  Clock, 
  Target, 
  CheckCircle, 
  AlertTriangle,
  TrendingUp,
  Download
} from "lucide-react";

interface BenchmarkResult {
  model: string;
  accuracy: number;
  brierScore: number;
  citationQuality: number;
  avgTime: number;
  eloRating: number;
}

interface BenchmarkProgress {
  current: number;
  total: number;
  currentTask: string;
  model: string;
}

export const BenchmarkRunner = () => {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<BenchmarkProgress | null>(null);
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [showResults, setShowResults] = useState(false);

  const models = [
    { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
    { id: "claude-3.5-sonnet", name: "Claude 3.5 Sonnet", provider: "Anthropic" },
    { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", provider: "Google" },
    { id: "llama-3.1-70b", name: "Llama 3.1 70B", provider: "Meta" }
  ];

  const benchmarks = [
    { 
      id: "mmlu", 
      name: "MMLU", 
      description: "57 academic subjects", 
      tasks: 15459, 
      estimatedTime: "45 min",
      difficulty: "Advanced"
    },
    { 
      id: "truthfulqa", 
      name: "TruthfulQA", 
      description: "Truthfulness evaluation", 
      tasks: 817, 
      estimatedTime: "20 min",
      difficulty: "Expert"
    },
    { 
      id: "gsm8k", 
      name: "GSM8K", 
      description: "Grade school math", 
      tasks: 1319, 
      estimatedTime: "30 min",
      difficulty: "Intermediate"
    }
  ];

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const toggleBenchmarkSelection = (benchmarkId: string) => {
    setSelectedBenchmarks(prev => 
      prev.includes(benchmarkId) 
        ? prev.filter(id => id !== benchmarkId)
        : [...prev, benchmarkId]
    );
  };

  const startBenchmark = async () => {
    if (selectedModels.length === 0 || selectedBenchmarks.length === 0) return;
    
    setIsRunning(true);
    setResults([]);
    setShowResults(false);

    const totalTasks = selectedBenchmarks.reduce((sum, benchmarkId) => {
      const benchmark = benchmarks.find(b => b.id === benchmarkId);
      return sum + (benchmark?.tasks || 0);
    }, 0) * selectedModels.length;

    let currentTask = 0;

    // Simulate benchmark execution
    for (const modelId of selectedModels) {
      const model = models.find(m => m.id === modelId);
      
      for (const benchmarkId of selectedBenchmarks) {
        const benchmark = benchmarks.find(b => b.id === benchmarkId);
        if (!benchmark || !model) continue;

        for (let i = 0; i < benchmark.tasks; i++) {
          currentTask++;
          setProgress({
            current: currentTask,
            total: totalTasks,
            currentTask: `${benchmark.name} - Task ${i + 1}`,
            model: model.name
          });

          // Simulate processing time
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
    }

    // Generate mock results
    const mockResults: BenchmarkResult[] = selectedModels.map(modelId => {
      const model = models.find(m => m.id === modelId);
      return {
        model: model?.name || modelId,
        accuracy: Math.random() * 0.3 + 0.7, // 70-100%
        brierScore: Math.random() * 0.2 + 0.1, // 0.1-0.3 (lower is better)
        citationQuality: Math.random() * 0.25 + 0.75, // 75-100%
        avgTime: Math.random() * 2000 + 1000, // 1-3 seconds
        eloRating: Math.random() * 400 + 1000 // 1000-1400
      };
    }).sort((a, b) => b.accuracy - a.accuracy);

    setResults(mockResults);
    setIsRunning(false);
    setProgress(null);
    setShowResults(true);
  };

  const exportResults = () => {
    const data = {
      timestamp: new Date().toISOString(),
      benchmarks: selectedBenchmarks,
      results: results
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark-results-${Date.now()}.json`;
    a.click();
  };

  return (
    <div className="space-y-6">
      {/* Configuration Panel */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Model Selection */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Target className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Select Models</h3>
          </div>
          
          <div className="space-y-3">
            {models.map((model) => (
              <div key={model.id} className="flex items-center space-x-2">
                <Checkbox
                  id={model.id}
                  checked={selectedModels.includes(model.id)}
                  onCheckedChange={() => toggleModelSelection(model.id)}
                />
                <label htmlFor={model.id} className="flex-1 cursor-pointer">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-foreground">{model.name}</span>
                    <Badge variant="outline" className="text-xs">{model.provider}</Badge>
                  </div>
                </label>
              </div>
            ))}
          </div>
        </Card>

        {/* Benchmark Selection */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-5 w-5 text-accent" />
            <h3 className="text-lg font-bold text-foreground">Select Benchmarks</h3>
          </div>
          
          <div className="space-y-3">
            {benchmarks.map((benchmark) => (
              <div key={benchmark.id} className="flex items-center space-x-2">
                <Checkbox
                  id={benchmark.id}
                  checked={selectedBenchmarks.includes(benchmark.id)}
                  onCheckedChange={() => toggleBenchmarkSelection(benchmark.id)}
                />
                <label htmlFor={benchmark.id} className="flex-1 cursor-pointer">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-foreground">{benchmark.name}</span>
                      <Badge className={
                        benchmark.difficulty === "Expert" ? "bg-accent/10 text-accent" :
                        benchmark.difficulty === "Advanced" ? "bg-primary/10 text-primary" :
                        "bg-muted text-muted-foreground"
                      }>
                        {benchmark.difficulty}
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {benchmark.description} • {benchmark.tasks.toLocaleString()} tasks • {benchmark.estimatedTime}
                    </div>
                  </div>
                </label>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Control Panel */}
      <Card className="p-6 gradient-surface">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-foreground mb-2">Benchmark Execution</h3>
            <p className="text-sm text-muted-foreground">
              {selectedModels.length} model(s) × {selectedBenchmarks.length} benchmark(s) selected
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={startBenchmark}
              disabled={selectedModels.length === 0 || selectedBenchmarks.length === 0 || isRunning}
              variant="gradient"
            >
              <Play className="h-4 w-4" />
              {isRunning ? "Running..." : "Start Benchmark"}
            </Button>
            {showResults && (
              <Button onClick={exportResults} variant="outline">
                <Download className="h-4 w-4" />
                Export Results
              </Button>
            )}
          </div>
        </div>

        {/* Progress Display */}
        {progress && (
          <div className="mt-6 space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-foreground">
                {progress.model} • {progress.currentTask}
              </span>
              <span className="text-muted-foreground">
                {progress.current.toLocaleString()} / {progress.total.toLocaleString()} tasks
              </span>
            </div>
            <Progress value={(progress.current / progress.total) * 100} className="h-2" />
          </div>
        )}
      </Card>

      {/* Results Display */}
      {showResults && results.length > 0 && (
        <div className="space-y-6">
          {/* Leaderboard */}
          <Card className="p-6">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-bold text-foreground">Benchmark Results</h3>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 text-sm font-medium text-muted-foreground">Rank</th>
                    <th className="text-left py-2 text-sm font-medium text-muted-foreground">Model</th>
                    <th className="text-center py-2 text-sm font-medium text-muted-foreground">Accuracy</th>
                    <th className="text-center py-2 text-sm font-medium text-muted-foreground">Brier Score</th>
                    <th className="text-center py-2 text-sm font-medium text-muted-foreground">Citation Quality</th>
                    <th className="text-center py-2 text-sm font-medium text-muted-foreground">Avg Time (ms)</th>
                    <th className="text-center py-2 text-sm font-medium text-muted-foreground">Elo Rating</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => (
                    <tr key={result.model} className="border-b border-border hover:bg-muted/50">
                      <td className="py-3">
                        <div className="flex items-center gap-2">
                          {index === 0 && <CheckCircle className="h-4 w-4 text-accent" />}
                          <span className="font-medium text-foreground">#{index + 1}</span>
                        </div>
                      </td>
                      <td className="py-3">
                        <span className="font-medium text-foreground">{result.model}</span>
                      </td>
                      <td className="py-3 text-center">
                        <Badge className="bg-primary/10 text-primary">
                          {(result.accuracy * 100).toFixed(1)}%
                        </Badge>
                      </td>
                      <td className="py-3 text-center">
                        <span className="text-foreground">{result.brierScore.toFixed(3)}</span>
                      </td>
                      <td className="py-3 text-center">
                        <Badge className="bg-accent/10 text-accent">
                          {(result.citationQuality * 100).toFixed(1)}%
                        </Badge>
                      </td>
                      <td className="py-3 text-center">
                        <span className="text-muted-foreground">{Math.round(result.avgTime)}</span>
                      </td>
                      <td className="py-3 text-center">
                        <span className="font-medium text-foreground">{Math.round(result.eloRating)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Performance Insights */}
          <Card className="p-6">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="h-5 w-5 text-accent" />
              <h3 className="text-lg font-bold text-foreground">Performance Insights</h3>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium text-foreground">Key Findings</h4>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>• {results[0]?.model} achieved highest accuracy at {(results[0]?.accuracy * 100).toFixed(1)}%</p>
                  <p>• Average citation quality across models: {((results.reduce((sum, r) => sum + r.citationQuality, 0) / results.length) * 100).toFixed(1)}%</p>
                  <p>• Fastest model: {results.sort((a, b) => a.avgTime - b.avgTime)[0]?.model}</p>
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium text-foreground">Recommendations</h4>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>• Consider model ensemble for optimal performance</p>
                  <p>• Monitor citation quality for factual accuracy</p>
                  <p>• Track performance trends over time</p>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};