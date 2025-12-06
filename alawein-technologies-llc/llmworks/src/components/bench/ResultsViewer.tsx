import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  Search, 
  Filter, 
  SortAsc, 
  Download, 
  Eye,
  BarChart3,
  Calendar,
  TrendingUp,
  CheckCircle
} from "lucide-react";

interface EvaluationResult {
  id: string;
  name: string;
  type: "benchmark" | "arena" | "custom";
  models: string[];
  timestamp: Date;
  status: "completed" | "running" | "failed";
  accuracy?: number;
  winner?: string;
  score?: number;
}

export const ResultsViewer = () => {
  const results: EvaluationResult[] = [
    {
      id: "result-1",
      name: "MMLU Benchmark Run",
      type: "benchmark",
      models: ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro"],
      timestamp: new Date("2024-01-15T10:30:00"),
      status: "completed",
      accuracy: 87.4
    },
    {
      id: "result-2", 
      name: "AI Ethics Debate",
      type: "arena",
      models: ["GPT-4o", "Claude 3.5 Sonnet"],
      timestamp: new Date("2024-01-14T15:45:00"),
      status: "completed",
      winner: "GPT-4o"
    },
    {
      id: "result-3",
      name: "Medical Diagnosis Test",
      type: "custom",
      models: ["Claude 3.5 Sonnet", "Gemini 1.5 Pro"],
      timestamp: new Date("2024-01-13T09:15:00"),
      status: "completed",
      score: 92.1
    },
    {
      id: "result-4",
      name: "Creative Writing Challenge",
      type: "arena",
      models: ["GPT-4o", "Claude 3.5 Sonnet"],
      timestamp: new Date("2024-01-12T14:20:00"),
      status: "completed",
      score: 85.7
    },
    {
      id: "result-5",
      name: "TruthfulQA Benchmark",
      type: "benchmark", 
      models: ["All Models"],
      timestamp: new Date("2024-01-11T11:00:00"),
      status: "running"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed": return "bg-accent/10 text-accent";
      case "running": return "bg-primary/10 text-primary";
      case "failed": return "bg-destructive/10 text-destructive";
      default: return "bg-muted text-muted-foreground";
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "benchmark": return BarChart3;
      case "arena": return CheckCircle;
      case "custom": return Eye;
      default: return BarChart3;
    }
  };

  const exportResult = (result: EvaluationResult) => {
    const data = {
      id: result.id,
      name: result.name,
      type: result.type,
      models: result.models,
      timestamp: result.timestamp,
      status: result.status,
      exportedAt: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${result.name.toLowerCase().replace(/\s+/g, '-')}-result.json`;
    a.click();
  };

  return (
    <div className="space-y-6">
      {/* Filters & Search */}
      <Card className="p-4">
        <div className="flex flex-col md:flex-row gap-4 items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input 
              placeholder="Search evaluations..." 
              className="pl-10"
            />
          </div>
          
          <div className="flex gap-2">
            <Select defaultValue="all">
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="benchmark">Benchmarks</SelectItem>
                <SelectItem value="arena">Arena</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>

            <Select defaultValue="recent">
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Sort" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="recent">Most Recent</SelectItem>
                <SelectItem value="oldest">Oldest First</SelectItem>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="score">Score</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" size="sm">
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
          </div>
        </div>
      </Card>

      {/* Results Grid */}
      <div className="grid gap-4">
        {results.map((result) => {
          const TypeIcon = getTypeIcon(result.type);
          
          return (
            <Card key={result.id} className="p-6 hover:shadow-medium transition-shadow">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4 flex-1">
                  <div className="p-2 rounded-lg bg-accent/10">
                    <TypeIcon className="h-5 w-5 text-accent" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-foreground truncate">{result.name}</h3>
                      <Badge variant="outline" className="capitalize">
                        {result.type}
                      </Badge>
                      <Badge className={getStatusColor(result.status)}>
                        {result.status}
                      </Badge>
                    </div>
                    
                    <div className="flex items-center gap-4 text-sm text-muted-foreground mb-3">
                      <div className="flex items-center gap-1">
                        <Calendar className="h-4 w-4" />
                        {result.timestamp.toLocaleDateString()} at {result.timestamp.toLocaleTimeString()}
                      </div>
                      <div>{result.models.join(", ")}</div>
                    </div>

                    {/* Result Summary */}
                    <div className="flex items-center gap-4">
                      {result.accuracy && (
                        <div className="flex items-center gap-1 text-sm">
                          <TrendingUp className="h-4 w-4 text-accent" />
                          <span className="text-foreground font-medium">{result.accuracy}% accuracy</span>
                        </div>
                      )}
                      {result.winner && (
                        <div className="flex items-center gap-1 text-sm">
                          <CheckCircle className="h-4 w-4 text-accent" />
                          <span className="text-foreground font-medium">Winner: {result.winner}</span>
                        </div>
                      )}
                      {result.score && (
                        <div className="flex items-center gap-1 text-sm">
                          <BarChart3 className="h-4 w-4 text-primary" />
                          <span className="text-foreground font-medium">{result.score}/100</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 ml-4">
                  <Button variant="ghost" size="sm">
                    <Eye className="h-4 w-4" />
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => exportResult(result)}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          Showing 5 of 47 results
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" disabled>
            Previous
          </Button>
          <Button variant="outline" size="sm">
            Next
          </Button>
        </div>
      </div>
    </div>
  );
};