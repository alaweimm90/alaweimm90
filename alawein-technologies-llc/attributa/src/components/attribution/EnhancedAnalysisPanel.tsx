import { useState, useMemo } from 'react';
import { useCurrentProject } from '@/hooks/useProjects';
import { useAnalysis } from '@/hooks/useAnalysis';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Play, 
  Loader2, 
  BarChart3, 
  AlertTriangle, 
  CheckCircle, 
  TrendingUp,
  FileText,
  Target,
  Brain
} from 'lucide-react';
import AttributionVisualization from './AttributionVisualization';
import AttributionFiltersComponent, { type AttributionFilters } from './AttributionFilters';
import ExportOptions from './ExportOptions';
import PerformanceMetrics from './PerformanceMetrics';

export default function EnhancedAnalysisPanel() {
  const { currentProject } = useCurrentProject();
  const { runAnalysis, isAnalyzing, analysisResults, analysisError } = useAnalysis();
  const attributions = analysisResults?.attributions || [];
  const isLoading = isAnalyzing;
  const [filters, setFilters] = useState<AttributionFilters>({
    confidenceRange: [0, 100],
    attributionTypes: [],
    sourceTypes: [],
    sortBy: 'confidence',
    sortOrder: 'desc'
  });
  const [analysisStartTime, setAnalysisStartTime] = useState<number | null>(null);

  const handleRunAnalysis = async () => {
    if (!currentProject?.id) return;
    setAnalysisStartTime(Date.now());
    await runAnalysis(currentProject.id);
  };

  // Filter and sort attributions
  const filteredAttributions = useMemo(() => {
    if (!attributions) return [];
    
    const filtered = attributions.filter(attr => {
      const confidencePercent = attr.confidence_score * 100;
      const matchesConfidence = confidencePercent >= filters.confidenceRange[0] && 
                                confidencePercent <= filters.confidenceRange[1];
      
      const matchesType = filters.attributionTypes.length === 0 || 
                         filters.attributionTypes.includes(attr.attribution_type);
      
      const matchesSource = filters.sourceTypes.length === 0 || 
                           filters.sourceTypes.includes(attr.attribution_type || '');
      
      return matchesConfidence && matchesType && matchesSource;
    });

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;
      switch (filters.sortBy) {
        case 'confidence':
          comparison = a.confidence_score - b.confidence_score;
          break;
        case 'similarity':
          comparison = (a.similarity_score || 0) - (b.similarity_score || 0);
          break;
        case 'created_at':
          comparison = new Date(a.created_at || 0).getTime() - new Date(b.created_at || 0).getTime();
          break;
      }
      return filters.sortOrder === 'desc' ? -comparison : comparison;
    });

    return filtered;
  }, [attributions, filters]);

  const analysisTime = analysisStartTime && !isLoading ? Date.now() - analysisStartTime : undefined;

  if (!currentProject) {
    return (
      <Card>
        <CardContent className="text-center py-12">
          <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">Select a project to view analysis results</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {analysisError && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Analysis failed: {analysisError.message}
          </AlertDescription>
        </Alert>
      )}

      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5" />
            Analysis Control
          </CardTitle>
          <CardDescription>
            Run comprehensive attribution analysis on your project content
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <Button 
              onClick={handleRunAnalysis}
              disabled={isLoading || !currentProject}
              className="flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {isLoading ? 'Analyzing...' : 'Run Analysis'}
            </Button>
            
            {attributions && attributions.length > 0 && (
              <ExportOptions 
                attributions={filteredAttributions}
                projectName={currentProject?.name}
              />
            )}
          </div>
          
          {isLoading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing artifacts...</span>
                <span className="text-muted-foreground">This may take a moment</span>
              </div>
              <div className="w-full bg-secondary rounded-full h-2">
                <div className="bg-primary h-2 rounded-full animate-pulse" style={{ width: '45%' }}></div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      {attributions && (
        <>
          {/* Filters */}
          <AttributionFiltersComponent
            onFiltersChange={setFilters}
            totalCount={attributions.length}
            filteredCount={filteredAttributions.length}
          />

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Attribution Results
                <Badge variant="secondary" className="ml-2">
                  {filteredAttributions.length} of {attributions.length}
                </Badge>
              </CardTitle>
              <CardDescription>
                Detailed attribution analysis with confidence scores and source information
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="list" className="space-y-4">
                <TabsList>
                  <TabsTrigger value="list" className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    List View
                  </TabsTrigger>
                  <TabsTrigger value="metrics" className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" />
                    Performance
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="list" className="space-y-4">
                  {filteredAttributions.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>
                        {attributions.length === 0 
                          ? "No attributions found in this analysis"
                          : "No attributions match the current filters"
                        }
                      </p>
                      <p className="text-sm">
                        {attributions.length === 0 
                          ? "This could indicate low similarity with known sources"
                          : "Try adjusting your filter criteria"
                        }
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {filteredAttributions.map((attribution) => (
                        <AttributionVisualization 
                          key={attribution.id} 
                          attribution={attribution} 
                        />
                      ))}
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="metrics">
                  <PerformanceMetrics 
                    attributions={filteredAttributions}
                    analysisTime={analysisTime}
                    projectStats={{
                      totalArtifacts: 10, // Mock data - would come from actual project stats
                      processedArtifacts: 8,
                      averageProcessingTime: 2500
                    }}
                  />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}