import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useToast } from '@/hooks/use-toast';
import { useCurrentProject } from './useProjects';
import { 
  computeAttributions,
  getAttributions,
  getProjectArtifacts
} from '@/services/attributionApi';

interface AnalysisProgress {
  stage: 'ingesting' | 'analyzing' | 'attributing' | 'complete';
  progress: number;
  message: string;
}

export function useAnalysis() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { currentProject } = useCurrentProject();

  const runAnalysisMutation = useMutation({
    mutationFn: async (artifactId?: string): Promise<AnalysisProgress[]> => {
      if (!currentProject) throw new Error('No project selected');

      const stages: AnalysisProgress[] = [];
      
      // Stage 1: Get artifacts for analysis
      stages.push({
        stage: 'analyzing',
        progress: 25,
        message: 'Analyzing content with GLTR and DetectGPT...'
      });
      
      // Stage 2: Compute attributions with enhanced algorithms
      stages.push({
        stage: 'attributing',
        progress: 75,
        message: 'Computing attributions between sources and artifacts...'
      });
      
      await computeAttributions(currentProject.id, artifactId);
      
      stages.push({
        stage: 'complete',
        progress: 100,
        message: 'Analysis complete with enhanced attribution intelligence'
      });
      
      return stages;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['attributions'] });
      queryClient.invalidateQueries({ queryKey: ['artifacts'] });
      toast({ 
        title: 'Analysis completed', 
        description: 'Enhanced attribution analysis with GLTR and DetectGPT signals' 
      });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Analysis failed',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive'
      });
    },
  });

  const analysisQuery = useQuery({
    queryKey: ['analysis-results', currentProject?.id],
    queryFn: async () => {
      if (!currentProject) return null;
      
      const [artifacts, attributions] = await Promise.all([
        getProjectArtifacts(currentProject.id),
        getAttributions(currentProject.id)
      ]);
      
      return {
        artifacts,
        attributions,
        summary: {
          totalArtifacts: artifacts.length,
          totalAttributions: attributions.length,
          highRiskAttributions: attributions.filter(a => a.confidence_level === 'High').length,
          averageConfidence: attributions.length > 0 
            ? attributions.reduce((sum, a) => sum + a.confidence_score, 0) / attributions.length 
            : 0
        }
      };
    },
    enabled: !!currentProject,
  });

  return {
    runAnalysis: runAnalysisMutation.mutate,
    isAnalyzing: runAnalysisMutation.isPending,
    analysisResults: analysisQuery.data,
    isLoadingResults: analysisQuery.isLoading,
    analysisError: runAnalysisMutation.error || analysisQuery.error,
  };
}

export function useArtifactAnalysis(artifactId?: string) {
  const { currentProject } = useCurrentProject();
  
  return useQuery({
    queryKey: ['artifact-analysis', currentProject?.id, artifactId],
    queryFn: async () => {
      if (!currentProject || !artifactId) return null;
      
      const attributions = await getAttributions(currentProject.id, artifactId);
      
      // Enhanced analysis metrics
      const analysisMetrics = {
        totalAttributions: attributions.length,
        confidenceDistribution: {
          high: attributions.filter(a => a.confidence_level === 'High').length,
          medium: attributions.filter(a => a.confidence_level === 'Medium').length,
          low: attributions.filter(a => a.confidence_level === 'Low').length,
        },
        attributionTypes: {
          direct: attributions.filter(a => a.attribution_type === 'direct').length,
          paraphrase: attributions.filter(a => a.attribution_type === 'paraphrase').length,
          summary: attributions.filter(a => a.attribution_type === 'summary').length,
          influence: attributions.filter(a => a.attribution_type === 'influence').length,
        },
        averageConfidence: attributions.length > 0 
          ? attributions.reduce((sum, a) => sum + a.confidence_score, 0) / attributions.length 
          : 0,
        signals: attributions.length > 0 ? attributions[0].signals : null
      };
      
      return {
        attributions,
        metrics: analysisMetrics
      };
    },
    enabled: !!currentProject && !!artifactId,
  });
}