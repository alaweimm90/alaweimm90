import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  ingestContent,
  getAttributions,
  computeAttributions,
  getCitations,
  createCitation,
  getProjectArtifacts,
  getProjectSources,
  IngestRequest
} from '@/services/attributionApi';
import { useToast } from '@/hooks/use-toast';

export function useIngest() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ingestContent,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['artifacts'] });
      queryClient.invalidateQueries({ queryKey: ['sources'] });
      toast({ 
        title: 'Content ingested successfully', 
        description: `${data.artifactCount} artifacts created` 
      });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Failed to ingest content',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive'
      });
    },
  });
}

export function useAttributions(projectId?: string, artifactId?: string) {
  return useQuery({
    queryKey: ['attributions', projectId, artifactId],
    queryFn: () => getAttributions(projectId!, artifactId),
    enabled: !!projectId,
  });
}

export function useComputeAttributions() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ projectId, artifactId }: { projectId: string; artifactId?: string }) =>
      computeAttributions(projectId, artifactId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['attributions'] });
      toast({ title: 'Attributions computed successfully' });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Failed to compute attributions',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive'
      });
    },
  });
}

export function useProjectArtifacts(projectId?: string) {
  return useQuery({
    queryKey: ['artifacts', projectId],
    queryFn: () => getProjectArtifacts(projectId!),
    enabled: !!projectId,
  });
}

export function useProjectSources(projectId?: string) {
  return useQuery({
    queryKey: ['sources', projectId],
    queryFn: () => getProjectSources(projectId!),
    enabled: !!projectId,
  });
}

export function useCitations(projectId?: string) {
  return useQuery({
    queryKey: ['citations', projectId],
    queryFn: () => getCitations(projectId!),
    enabled: !!projectId,
  });
}

export function useCreateCitation() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ 
      projectId, 
      rawCitation, 
      format 
    }: { 
      projectId: string; 
      rawCitation: string; 
      format?: 'APA' | 'MLA' | 'IEEE' | 'Chicago' 
    }) => createCitation(projectId, rawCitation, format),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['citations'] });
      toast({ title: 'Citation created successfully' });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Failed to create citation',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive'
      });
    },
  });
}