import { useEffect, useState, Suspense, lazy, memo } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Navigation } from "@/components/Navigation";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { 
  BarChart3, 
  Play, 
  Settings, 
  Download,
  ArrowRight,
  TrendingUp,
  CheckCircle,
  Clock,
  Target
} from "lucide-react";
import { setSEO, injectJsonLd } from "@/lib/seo";
import { trackEvent } from "@/lib/analytics";

const BenchmarkRunner = lazy(() => import("@/components/bench/BenchmarkRunner").then(m => ({ default: m.BenchmarkRunner })));
const CustomTestBuilder = lazy(() => import("@/components/bench/CustomTestBuilder").then(m => ({ default: m.CustomTestBuilder })));
const ResultsViewer = lazy(() => import("@/components/bench/ResultsViewer").then(m => ({ default: m.ResultsViewer })));

const Bench = memo(() => {
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    setSEO({
      title: "The Bench | LLM Works",
      description: "Open‑source benchmarking that runs in your browser. MMLU, TruthfulQA, GSM8K, custom tests.",
      path: "/bench",
    });
    injectJsonLd({
      "@context": "https://schema.org",
      "@type": "WebPage",
      name: "The Bench — AI Benchmarking Suite",
      url: `${window.location.origin}/bench`,
      description: "Standardized benchmarks and custom tests with auditable reports.",
      isPartOf: {
        "@type": "SoftwareApplication",
        name: "LLM Works",
      },
    }, "ld-bench");
  }, []);
  const benchmarks = [
    {
      id: "mmlu",
      name: "MMLU",
      description: "Massive Multitask Language Understanding - 57 academic subjects",
      tasks: 15459,
      avgTime: "~45 min",
      status: "Ready",
      difficulty: "Advanced",
      color: "bg-accent/10 text-accent"
    },
    {
      id: "truthfulqa", 
      name: "TruthfulQA",
      description: "Tests resistance to generating false information and misconceptions",
      tasks: 817,
      avgTime: "~20 min",
      status: "Ready",
      difficulty: "Expert",
      color: "bg-primary/10 text-primary"
    },
    {
      id: "gsm8k",
      name: "GSM8K",
      description: "Grade School Math problems requiring multi-step reasoning",
      tasks: 1319,
      avgTime: "~30 min", 
      status: "Ready",
      difficulty: "Intermediate",
      color: "bg-accent/10 text-accent"
    },
    {
      id: "custom",
      name: "Custom Tests",
      description: "Upload your own evaluation datasets and criteria",
      tasks: "Variable",
      avgTime: "Variable",
      status: "Available",
      difficulty: "Custom",
      color: "bg-muted text-muted-foreground"
    }
  ];

  const metrics = [
    { label: "Accuracy", description: "Percentage of correct answers", icon: Target },
    { label: "Brier Score", description: "Calibration of confidence scores", icon: TrendingUp },
    { label: "Citation Quality", description: "Validity of provided sources", icon: CheckCircle },
    { label: "Elo Rating", description: "Dynamic competitive ranking", icon: BarChart3 }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main id="main" className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6">
            <BarChart3 className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium text-primary neural-text">STANDARDIZED BENCHMARKING</span>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6">
            THE BENCH — BENCHMARK EVALUATION
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Run standardized benchmarks. Generate detailed reports with performance analysis and audit trails.
          </p>
        </div>

        {/* Benchmark Interface */}
        <Tabs value={activeTab} onValueChange={(val) => { setActiveTab(val); trackEvent('bench_tab_change', { tab: val }); }} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="runner">Run Benchmarks</TabsTrigger>
            <TabsTrigger value="custom">Custom Tests</TabsTrigger>
            <TabsTrigger value="results">View Results</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-8">
            {/* Available Benchmarks */}
            <div className="mb-12">
              <h2 className="text-3xl font-bold text-foreground mb-8">Available Benchmarks</h2>
              <div className="grid lg:grid-cols-2 gap-6">
                {benchmarks.map((benchmark) => (
                  <Card key={benchmark.id} className="p-6 shadow-medium hover:shadow-strong transition-all duration-300 group">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold text-foreground mb-2">{benchmark.name}</h3>
                        <Badge variant={benchmark.status === "Ready" ? "default" : "secondary"}>
                          {benchmark.status}
                        </Badge>
                      </div>
                      <Badge className={benchmark.color}>
                        {benchmark.difficulty}
                      </Badge>
                    </div>

                    <p className="text-muted-foreground mb-6">{benchmark.description}</p>

                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="flex items-center gap-2">
                        <Target className="h-4 w-4 text-accent" />
                        <div>
                          <div className="text-sm font-medium text-foreground">{benchmark.tasks}</div>
                          <div className="text-xs text-muted-foreground">Tasks</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-accent" />
                        <div>
                          <div className="text-sm font-medium text-foreground">{benchmark.avgTime}</div>
                          <div className="text-xs text-muted-foreground">Avg Time</div>
                        </div>
                      </div>
                    </div>

                    <Button 
                      variant="gradient" 
                      className="w-full group-hover:shadow-glow"
                      onClick={() => setActiveTab("runner")}
                    >
                      <Play className="h-4 w-4" />
                      Run Benchmark
                    </Button>
                  </Card>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="runner">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Benchmark Runner..." />}>
              <BenchmarkRunner />
            </Suspense>
          </TabsContent>

          <TabsContent value="custom">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Custom Test Builder..." />}>
              <CustomTestBuilder />
            </Suspense>
          </TabsContent>

          <TabsContent value="results">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Results Viewer..." />}>
              <ResultsViewer />
            </Suspense>
          </TabsContent>
        </Tabs>

        {/* Evaluation Metrics */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-foreground mb-8">Evaluation Metrics</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {metrics.map((metric, index) => (
              <Card key={index} className="p-6 text-center shadow-medium hover:shadow-strong transition-all duration-300">
                <metric.icon className="h-8 w-8 text-accent mx-auto mb-4" />
                <h3 className="text-lg font-bold text-foreground mb-2">{metric.label}</h3>
                <p className="text-sm text-muted-foreground">{metric.description}</p>
              </Card>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <Card className="p-8 gradient-surface shadow-medium">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-foreground mb-4">Start Benchmarking</h3>
              <p className="text-muted-foreground mb-4">
                Select your models, choose your benchmarks, and generate detailed evaluation reports 
                with performance analytics and detailed reporting.
              </p>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-primary" />
                  <span className="text-sm text-muted-foreground">Automated scoring</span>
                </div>
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  <span className="text-sm text-muted-foreground">Performance tracking</span>
                </div>
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-primary" />
                  <span className="text-sm text-muted-foreground">Exportable reports</span>
                </div>
              </div>
            </div>
            <div className="flex flex-col gap-3">
              <Button variant="hero" size="lg">
                <Settings className="h-4 w-4" />
                Configure Evaluation
                <ArrowRight className="h-4 w-4" />
              </Button>
              <Button variant="outline">
                View Sample Reports
              </Button>
            </div>
          </div>
        </Card>
      </main>
    </div>
  );
});

Bench.displayName = 'Bench';

export default Bench;