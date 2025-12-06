import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Brain, BarChart3, Target, AlertTriangle } from 'lucide-react';

interface AttributionVisualizationProps {
  attribution: {
    id: string;
    attribution_type: string;
    confidence_level: string;
    confidence_score: number;
    similarity_score?: number;
    signals: Record<string, unknown>;
    rationale: string[];
    sources?: {
      id: string;
      title?: string;
      source_type: string;
    };
  };
}

const AttributionVisualization = ({ attribution }: AttributionVisualizationProps) => {
  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'High': return 'destructive';
      case 'Medium': return 'warning';
      default: return 'success';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'direct': return 'bg-destructive/10 text-destructive border-destructive/20';
      case 'paraphrase': return 'bg-warning/10 text-warning border-warning/20';
      case 'summary': return 'bg-info/10 text-info border-info/20';
      default: return 'bg-muted/10 text-muted-foreground border-border';
    }
  };

  const signals = attribution.signals || {};
  const combinedAnalysis = signals.combinedAnalysis || null;
  const gltrData = signals.gltr || { tailTokenShare: 0, rankVariance: 0, histogram: [0, 0, 0, 0] };
  const detectgptData = signals.detectgpt || { curvature: 0, numPerturbations: 0 };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="text-lg flex items-center gap-2">
              {attribution.sources?.title || 'Unknown Source'}
              <Badge variant="outline" className={getTypeColor(attribution.attribution_type)}>
                {attribution.attribution_type}
              </Badge>
            </CardTitle>
            <CardDescription>
              {attribution.sources?.source_type} • Confidence: {(attribution.confidence_score * 100).toFixed(1)}%
            </CardDescription>
          </div>
          <Badge variant={getConfidenceColor(attribution.confidence_level) as 'destructive' | 'warning' | 'success' | 'default'}>
            {attribution.confidence_level} Risk
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Multi-Signal Analysis Overview */}
        {combinedAnalysis && (
          <div className="space-y-4">
            <h4 className="font-semibold flex items-center gap-2">
              <Target className="h-4 w-4" />
              Multi-Signal Analysis
            </h4>
            
            <div className="grid grid-cols-3 gap-4">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="text-center p-3 rounded-lg bg-info/10 border border-info/20">
                      <div className="text-2xl font-bold text-info">
                        {(combinedAnalysis.components.semantic * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-info">Semantic</div>
                      <div className="text-xs text-muted-foreground">
                        Weight: {(combinedAnalysis.semanticWeight * 100).toFixed(0)}%
                      </div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Semantic similarity between content and source</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="text-center p-3 rounded-lg bg-green-50 border border-green-200">
                      <div className="text-2xl font-bold text-green-700">
                        {(combinedAnalysis.components.gltr * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-green-600">GLTR</div>
                      <div className="text-xs text-muted-foreground">
                        Weight: {(combinedAnalysis.gltrWeight * 100).toFixed(0)}%
                      </div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Token probability distribution analysis</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="text-center p-3 rounded-lg bg-purple-50 border border-purple-200">
                      <div className="text-2xl font-bold text-purple-700">
                        {(combinedAnalysis.components.detectgpt * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-purple-600">DetectGPT</div>
                      <div className="text-xs text-muted-foreground">
                        Weight: {(combinedAnalysis.detectgptWeight * 100).toFixed(0)}%
                      </div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Curvature analysis for AI generation detection</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Combined Confidence Score</span>
                <span className="font-medium">{(combinedAnalysis.finalScore * 100).toFixed(1)}%</span>
              </div>
              <Progress 
                value={combinedAnalysis.finalScore * 100} 
                className="h-2"
              />
            </div>
          </div>
        )}

        {/* GLTR Analysis Details */}
        <div className="space-y-3">
          <h4 className="font-semibold flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            GLTR Token Analysis
          </h4>
          
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span>Top-10 tokens</span>
              <div className="flex items-center gap-2">
                <Progress value={(gltrData.histogram[0] || 0) * 100} className="w-16 h-2" />
                <span className="text-green-600 font-mono text-xs">
                  {((gltrData.histogram[0] || 0) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            
            <div className="flex justify-between items-center text-sm">
              <span>Top-100 tokens</span>
              <div className="flex items-center gap-2">
                <Progress value={(gltrData.histogram[1] || 0) * 100} className="w-16 h-2" />
                <span className="text-yellow-600 font-mono text-xs">
                  {((gltrData.histogram[1] || 0) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            
            <div className="flex justify-between items-center text-sm">
              <span>Beyond top-1000</span>
              <div className="flex items-center gap-2">
                <Progress value={(gltrData.histogram[3] || 0) * 100} className="w-16 h-2" />
                <span className="text-red-600 font-mono text-xs">
                  {((gltrData.histogram[3] || 0) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground">
            Tail token share: {((gltrData.tailTokenShare || 0) * 100).toFixed(1)}% • 
            Rank variance: {(gltrData.rankVariance || 0).toFixed(3)}
          </div>
        </div>

        {/* DetectGPT Analysis */}
        <div className="space-y-3">
          <h4 className="font-semibold flex items-center gap-2">
            <Brain className="h-4 w-4" />
            DetectGPT Analysis
          </h4>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">Curvature Score</div>
              <div className="font-mono text-lg">
                {(detectgptData.curvature || 0).toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">Perturbations</div>
              <div className="font-mono text-lg">
                {detectgptData.numPerturbations || 0}
              </div>
            </div>
          </div>
          
          <div className="text-xs text-muted-foreground">
            {(detectgptData.curvature || 0) > 0 
              ? "Positive curvature suggests potential AI generation patterns"
              : "Negative curvature indicates more natural language variation"
            }
          </div>
        </div>

        {/* Rationale */}
        <div className="space-y-2">
          <h4 className="font-semibold flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Analysis Rationale
          </h4>
          <ul className="text-sm space-y-1">
            {attribution.rationale.map((reason, index) => (
              <li key={index} className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground mt-2 flex-shrink-0" />
                <span className="text-muted-foreground">{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default AttributionVisualization;