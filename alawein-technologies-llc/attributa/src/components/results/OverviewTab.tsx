import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { FileText, Code, BookOpen, AlertTriangle, Info, Copy, Download } from 'lucide-react';
import { Report } from '@/types';
import { getScoringWeightsExplanation, computeOverallScore } from '@/lib/scoring';
import { useAppStore } from '@/store';
import { useToast } from '@/hooks/use-toast';
import { exportReportAsJSON, exportReportAsCSV } from '@/lib/export';

interface OverviewTabProps {
  report: Report;
}

const OverviewTab = ({ report }: OverviewTabProps) => {
  const { scoringWeights } = useAppStore();
  const { toast } = useToast();

  const handleCopyWeights = () => {
    const text = getScoringWeightsExplanation(scoringWeights);
    navigator.clipboard.writeText(text).then(() => {
      toast({
        title: "Weights copied",
        description: "Scoring weights have been copied to your clipboard.",
      });
    }).catch(() => {
      toast({
        title: "Copy failed",
        description: "Could not copy weights. Please try again.",
        variant: "destructive",
      });
    });
  };

  const handleExportJSON = () => {
    try {
      exportReportAsJSON(report, scoringWeights);
      toast({
        title: "JSON exported",
        description: "Report data has been downloaded as JSON.",
      });
    } catch (error) {
      toast({
        title: "Export failed",
        description: "Could not export JSON. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleExportCSV = () => {
    try {
      exportReportAsCSV(report);
      toast({
        title: "CSV exported",
        description: "Report data has been downloaded as CSV.",
      });
    } catch (error) {
      toast({
        title: "Export failed",
        description: "Could not export CSV. Please try again.",
        variant: "destructive",
      });
    }
  };

  // Calculate overall statistics
  const totalSegments = report.segments.length;
  const hasAnalysis = Object.keys(report.scores || {}).length > 0;
  
  let avgScore = 0;
  let confidence: "Low" | "Medium" | "High" = "Low";
  let topDrivers: string[] = [];
  
  if (hasAnalysis && report.scores) {
    // Use length-weighted scoring instead of simple average
    avgScore = computeOverallScore(report.scores, report.segments);
    
    if (avgScore >= 0.65) confidence = "High";
    else if (avgScore >= 0.45) confidence = "Medium";
    
    // Collect rationale from highest scoring segments
    const scores = Object.values(report.scores);
    const sortedScores = scores.sort((a, b) => b.score - a.score);
    topDrivers = sortedScores.slice(0, 3).flatMap(s => s.rationale).slice(0, 5);
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'code': return <Code className="h-4 w-4" />;
      case 'latex': return <BookOpen className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getConfidenceColor = () => {
    switch (confidence) {
      case 'High': return 'bg-ai-high/10 text-ai-high border-ai-high/20';
      case 'Medium': return 'bg-ai-medium/10 text-ai-medium border-ai-medium/20';
      default: return 'bg-ai-low/10 text-ai-low border-ai-low/20';
    }
  };

  return (
    <div className="space-y-4">
      {/* Export Actions */}
      <div className="flex justify-end gap-2">
        <Button variant="outline" size="sm" onClick={handleExportJSON} aria-label="Export JSON">
          <Download className="h-4 w-4 mr-2" />
          Export JSON
        </Button>
        <Button variant="outline" size="sm" onClick={handleExportCSV} aria-label="Export CSV">
          <Download className="h-4 w-4 mr-2" />
          Export CSV
        </Button>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
      {/* Document Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Document Summary</CardTitle>
          <CardDescription>Basic document statistics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Total Characters</span>
              <span className="font-medium">{report.summary.totalChars.toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Segments</span>
              <span className="font-medium">{totalSegments}</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Content Types</h4>
            <div className="space-y-2">
              {Object.entries(report.summary.types).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    {getTypeIcon(type)}
                    <span className="capitalize">{type}</span>
                  </div>
                  <Badge variant="secondary">{count}</Badge>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Confidence Meter */}
      <Card>
        <CardHeader>
          <CardTitle>Attribution Confidence</CardTitle>
          <CardDescription>Composite analysis across all segments</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {hasAnalysis ? (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Confidence Level</span>
                  <Badge className={getConfidenceColor()}>
                    {confidence} ({(avgScore * 100).toFixed(0)}%)
                  </Badge>
                </div>
                <Progress value={avgScore * 100} className="h-3" />
                <div className="text-xs text-muted-foreground flex items-center gap-1">
                  <Info className="h-3 w-3" />
                  <span>{getScoringWeightsExplanation(scoringWeights)}</span>
                  <button
                    type="button"
                    aria-label="Copy weights to clipboard"
                    onClick={handleCopyWeights}
                    className="ml-1 rounded p-1 hover:bg-accent text-muted-foreground"
                    title="Copy weights"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Document score = length-weighted mean of segment scores (short segments &lt;1000 chars count half)
                </div>
              </div>
              
              <div className="space-y-2">
                <span className="text-sm font-medium">Confidence Band</span>
                <p className="text-xs text-muted-foreground">
                  Confidence reflects signal strength and input length, not authorship certainty.
                </p>
              </div>
            </>
          ) : (
            <div className="flex items-center space-x-2 text-muted-foreground">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm">{totalSegments === 0 ? 'No segments to analyze (input may be too short).' : 'Analysis in progress...'}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Key Drivers */}
      <Card>
        <CardHeader>
          <CardTitle>Key Drivers</CardTitle>
          <CardDescription>Primary signals contributing to the score</CardDescription>
        </CardHeader>
        <CardContent>
          {hasAnalysis && topDrivers.length > 0 ? (
            <ul className="space-y-2">
              {topDrivers.map((driver, index) => (
                <li key={index} className="text-sm text-muted-foreground flex items-start space-x-2">
                  <span className="text-primary font-medium">•</span>
                  <span>{driver}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="flex items-center space-x-2 text-muted-foreground">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm">
                {hasAnalysis ? 'No significant drivers identified' : 'Awaiting analysis completion'}
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Analysis Status */}
      <Card className="lg:col-span-3">
        <CardHeader>
          <CardTitle>Analysis Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center space-y-2">
              <div className="text-2xl font-bold text-primary">
                {hasAnalysis ? '✓' : '⏳'}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Text Analysis</p>
                <p className="text-xs text-muted-foreground">GLTR & DetectGPT</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-2xl font-bold text-primary">
                {report.citations ? '✓' : hasAnalysis ? '✓' : '⏳'}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Citations</p>
                <p className="text-xs text-muted-foreground">Reference validation</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-2xl font-bold text-primary">
                {report.codeFindings ? '✓' : hasAnalysis ? '✓' : '⏳'}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Code Security</p>
                <p className="text-xs text-muted-foreground">Vulnerability scan</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-2xl font-bold text-primary">
                {hasAnalysis ? '✓' : '⏳'}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Scoring</p>
                <p className="text-xs text-muted-foreground">Composite analysis</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      </div>
    </div>
  );
};

export default OverviewTab;