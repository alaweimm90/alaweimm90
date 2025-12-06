import { supabase } from '@/integrations/supabase/client';

export interface Project {
  id: string;
  name: string;
  description?: string;
  settings: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface Source {
  id: string;
  project_id: string;
  source_type: 'file' | 'url' | 'text' | 'github';
  title?: string;
  content_hash: string;
  original_content?: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface Artifact {
  id: string;
  project_id: string;
  source_id?: string;
  artifact_type: 'prose' | 'code' | 'latex' | 'mixed';
  title?: string;
  content: string;
  content_hash: string;
  char_length: number;
  analysis_results: Record<string, unknown>;
  created_at: string;
}

export interface Attribution {
  id: string;
  artifact_id: string;
  source_id: string;
  attribution_type: 'direct' | 'paraphrase' | 'summary' | 'influence';
  confidence_score: number;
  confidence_level: 'Low' | 'Medium' | 'High';
  similarity_score?: number;
  signals: Record<string, unknown>;
  rationale: string[];
  created_at: string;
}

export interface Citation {
  id: string;
  project_id: string;
  raw_citation: string;
  parsed_citation?: Record<string, unknown>;
  citation_format?: 'APA' | 'MLA' | 'IEEE' | 'Chicago';
  doi?: string;
  resolves: boolean;
  validation_status: 'pending' | 'valid' | 'invalid' | 'suggestions';
  suggestions: string[];
  created_at: string;
}

// Project Management
export async function createProject(name: string, description?: string): Promise<Project> {
  const { data, error } = await supabase.functions.invoke('projects', {
    body: { name, description }
  });

  if (error) throw new Error(error.message);
  return data.project;
}

export async function getProjects(): Promise<Project[]> {
  const { data, error } = await supabase.functions.invoke('projects', {
    method: 'GET'
  });

  if (error) throw new Error(error.message);
  return data.projects;
}

export async function updateProject(id: string, updates: Partial<Project>): Promise<Project> {
  const response = await fetch(`https://ayfvtntapmfpzxwaqwrt.supabase.co/functions/v1/projects?id=${id}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`,
      'apikey': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF5ZnZ0bnRhcG1mcHp4d2Fxd3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ2NDkyNDYsImV4cCI6MjA3MDIyNTI0Nn0.Or0UPScDFBjU-eBmY54Y2mnn287Eb_ilMaFlsEqE4jA'
    },
    body: JSON.stringify(updates)
  });

  if (!response.ok) {
    throw new Error('Failed to update project');
  }

  const data = await response.json();
  return data.project;
}

export async function deleteProject(id: string): Promise<void> {
  const response = await fetch(`https://ayfvtntapmfpzxwaqwrt.supabase.co/functions/v1/projects?id=${id}`, {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`,
      'apikey': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF5ZnZ0bnRhcG1mcHp4d2Fxd3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ2NDkyNDYsImV4cCI6MjA3MDIyNTI0Nn0.Or0UPScDFBjU-eBmY54Y2mnn287Eb_ilMaFlsEqE4jA'
    }
  });

  if (!response.ok) {
    throw new Error('Failed to delete project');
  }
}

// Content Ingestion
export interface IngestRequest {
  projectId: string;
  source: 'paste' | 'file' | 'url' | 'github';
  content?: string;
  files?: { name: string; content: string }[];
  githubUrl?: string;
  title?: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResponse {
  sourceId: string;
  artifactCount: number;
  message: string;
  duplicate?: boolean;
}

export async function ingestContent(request: IngestRequest): Promise<IngestResponse> {
  const { data, error } = await supabase.functions.invoke('ingest', {
    body: request
  });

  if (error) throw new Error(error.message);
  return data;
}

// Attribution Analysis
export async function getAttributions(projectId: string, artifactId?: string): Promise<Attribution[]> {
  const params = new URLSearchParams({ projectId });
  if (artifactId) params.append('artifactId', artifactId);

  const response = await fetch(`https://ayfvtntapmfpzxwaqwrt.supabase.co/functions/v1/attributions?${params}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`,
      'apikey': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF5ZnZ0bnRhcG1mcHp4d2Fxd3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ2NDkyNDYsImV4cCI6MjA3MDIyNTI0Nn0.Or0UPScDFBjU-eBmY54Y2mnn287Eb_ilMaFlsEqE4jA'
    }
  });

  if (!response.ok) {
    throw new Error('Failed to get attributions');
  }

  const data = await response.json();
  return data.attributions;
}

export async function computeAttributions(projectId: string, artifactId?: string): Promise<Attribution[]> {
  const { data, error } = await supabase.functions.invoke('attributions', {
    body: { projectId, artifactId }
  });

  if (error) throw new Error(error.message);
  return data.attributions;
}

// Citation Management
export async function getCitations(projectId: string): Promise<Citation[]> {
  const response = await fetch(`https://ayfvtntapmfpzxwaqwrt.supabase.co/functions/v1/citations?projectId=${projectId}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`,
      'apikey': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF5ZnZ0bnRhcG1mcHp4d2Fxd3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ2NDkyNDYsImV4cCI6MjA3MDIyNTI0Nn0.Or0UPScDFBjU-eBmY54Y2mnn287Eb_ilMaFlsEqE4jA'
    }
  });

  if (!response.ok) {
    throw new Error('Failed to get citations');
  }

  const data = await response.json();
  return data.citations;
}

export async function createCitation(
  projectId: string,
  rawCitation: string,
  format?: 'APA' | 'MLA' | 'IEEE' | 'Chicago'
): Promise<{ citation: Citation; formatted: string; validation: Record<string, unknown> }> {
  const { data, error } = await supabase.functions.invoke('citations', {
    body: { projectId, rawCitation, format }
  });

  if (error) throw new Error(error.message);
  return data;
}

// Database queries for direct access when needed
export async function getProjectArtifacts(projectId: string): Promise<Artifact[]> {
  const { data, error } = await supabase
    .from('artifacts')
    .select('*')
    .eq('project_id', projectId)
    .order('created_at', { ascending: false });

  if (error) throw new Error(error.message);
  return data as Artifact[];
}

export async function getProjectSources(projectId: string): Promise<Source[]> {
  const { data, error } = await supabase
    .from('sources')
    .select('*')
    .eq('project_id', projectId)
    .order('created_at', { ascending: false });

  if (error) throw new Error(error.message);
  return data as Source[];
}