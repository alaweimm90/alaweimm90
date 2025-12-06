import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  createProject, 
  getProjects, 
  updateProject, 
  deleteProject,
  Project 
} from '@/services/attributionApi';
import { useAuth } from './useAuth';
import { useToast } from '@/hooks/use-toast';

export function useProjects() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const projectsQuery = useQuery({
    queryKey: ['projects'],
    queryFn: getProjects,
    enabled: !!user,
  });

  const createProjectMutation = useMutation({
    mutationFn: ({ name, description }: { name: string; description?: string }) =>
      createProject(name, description),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      toast({ title: 'Project created successfully' });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Failed to create project',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive'
      });
    },
  });

  const updateProjectMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: Partial<Project> }) =>
      updateProject(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      toast({ title: 'Project updated successfully' });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Failed to update project',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive'
      });
    },
  });

  const deleteProjectMutation = useMutation({
    mutationFn: deleteProject,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      toast({ title: 'Project deleted successfully' });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Failed to delete project',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive'
      });
    },
  });
  
  // Guards to prevent unauthenticated mutations
  const requireAuth = () => {
    toast({
      title: 'Workspace feature',
      description: 'Create an optional account to manage projects (cloud sync).'
    });
  };

  const createProjectSafe = ({ name, description }: { name: string; description?: string }) => {
    if (!user) return requireAuth();
    createProjectMutation.mutate({ name, description });
  };

  const updateProjectSafe = ({ id, updates }: { id: string; updates: Partial<Project> }) => {
    if (!user) return requireAuth();
    updateProjectMutation.mutate({ id, updates });
  };

  const deleteProjectSafe = (id: string) => {
    if (!user) return requireAuth();
    deleteProjectMutation.mutate(id);
  };

  return {
    projects: projectsQuery.data || [],
    isLoading: projectsQuery.isLoading,
    error: projectsQuery.error,
    createProject: createProjectSafe,
    updateProject: updateProjectSafe,
    deleteProject: deleteProjectSafe,
    isCreating: createProjectMutation.isPending,
    isUpdating: updateProjectMutation.isPending,
    isDeleting: deleteProjectMutation.isPending,
  };
}

export function useCurrentProject() {
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(
    localStorage.getItem('currentProjectId')
  );
  const { projects } = useProjects();

  useEffect(() => {
    if (currentProjectId) {
      localStorage.setItem('currentProjectId', currentProjectId);
    } else {
      localStorage.removeItem('currentProjectId');
    }
  }, [currentProjectId]);

  const currentProject = projects.find(p => p.id === currentProjectId);

  return {
    currentProject,
    currentProjectId,
    setCurrentProjectId,
  };
}