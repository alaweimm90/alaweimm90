-- Attributa.dev Database Schema - Phase 1
-- Core tables for attribution platform

-- Projects table - represents user projects/workspaces
CREATE TABLE public.projects (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  settings JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Sources table - original content sources
CREATE TABLE public.sources (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  source_type TEXT NOT NULL CHECK (source_type IN ('file', 'url', 'text', 'github')),
  title TEXT,
  content_hash TEXT NOT NULL,
  original_content TEXT,
  metadata JSONB DEFAULT '{}',
  embeddings vector(384), -- Using pgvector for semantic search
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Artifacts table - processed content segments
CREATE TABLE public.artifacts (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  source_id UUID REFERENCES public.sources(id) ON DELETE SET NULL,
  artifact_type TEXT NOT NULL CHECK (artifact_type IN ('prose', 'code', 'latex', 'mixed')),
  title TEXT,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  char_length INTEGER NOT NULL,
  embeddings vector(384),
  analysis_results JSONB DEFAULT '{}', -- GLTR, DetectGPT, watermark results
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Attributions table - attribution relationships and scores
CREATE TABLE public.attributions (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  artifact_id UUID NOT NULL REFERENCES public.artifacts(id) ON DELETE CASCADE,
  source_id UUID NOT NULL REFERENCES public.sources(id) ON DELETE CASCADE,
  attribution_type TEXT NOT NULL CHECK (attribution_type IN ('direct', 'paraphrase', 'summary', 'influence')),
  confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
  confidence_level TEXT NOT NULL CHECK (confidence_level IN ('Low', 'Medium', 'High')),
  similarity_score DECIMAL(3,2),
  signals JSONB DEFAULT '{}', -- GLTR, DetectGPT, watermark signals
  rationale TEXT[],
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Citations table - bibliographic citations and validation
CREATE TABLE public.citations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  raw_citation TEXT NOT NULL,
  parsed_citation JSONB,
  citation_format TEXT CHECK (citation_format IN ('APA', 'MLA', 'IEEE', 'Chicago')),
  doi TEXT,
  resolves BOOLEAN DEFAULT false,
  validation_status TEXT CHECK (validation_status IN ('pending', 'valid', 'invalid', 'suggestions')),
  suggestions JSONB DEFAULT '[]',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Lineage table - content lineage and provenance tracking
CREATE TABLE public.lineage (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  source_artifact_id UUID NOT NULL REFERENCES public.artifacts(id) ON DELETE CASCADE,
  target_artifact_id UUID NOT NULL REFERENCES public.artifacts(id) ON DELETE CASCADE,
  relationship_type TEXT NOT NULL CHECK (relationship_type IN ('derived', 'influenced', 'cited', 'merged')),
  weight DECIMAL(3,2) DEFAULT 1.0,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  CONSTRAINT no_self_reference CHECK (source_artifact_id != target_artifact_id)
);

-- Reports table - analysis reports and exports
CREATE TABLE public.reports (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  report_type TEXT NOT NULL CHECK (report_type IN ('full', 'summary', 'compliance')),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  results JSONB DEFAULT '{}',
  export_format TEXT CHECK (export_format IN ('json', 'csv', 'pdf')),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  completed_at TIMESTAMP WITH TIME ZONE
);

-- Audit events table - compliance and audit trail
CREATE TABLE public.audit_events (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  project_id UUID REFERENCES public.projects(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  resource_type TEXT,
  resource_id UUID,
  details JSONB DEFAULT '{}',
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security on all tables
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.attributions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.citations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lineage ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_events ENABLE ROW LEVEL SECURITY;

-- RLS Policies for projects
CREATE POLICY "Users can view their own projects" ON public.projects FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can create their own projects" ON public.projects FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own projects" ON public.projects FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own projects" ON public.projects FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for sources
CREATE POLICY "Users can view sources in their projects" ON public.sources FOR SELECT 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = sources.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can create sources in their projects" ON public.sources FOR INSERT 
  WITH CHECK (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = sources.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can update sources in their projects" ON public.sources FOR UPDATE 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = sources.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can delete sources in their projects" ON public.sources FOR DELETE 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = sources.project_id AND projects.user_id = auth.uid()));

-- RLS Policies for artifacts (similar pattern)
CREATE POLICY "Users can view artifacts in their projects" ON public.artifacts FOR SELECT 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = artifacts.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can create artifacts in their projects" ON public.artifacts FOR INSERT 
  WITH CHECK (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = artifacts.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can update artifacts in their projects" ON public.artifacts FOR UPDATE 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = artifacts.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can delete artifacts in their projects" ON public.artifacts FOR DELETE 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = artifacts.project_id AND projects.user_id = auth.uid()));

-- RLS Policies for attributions
CREATE POLICY "Users can view attributions in their projects" ON public.attributions FOR SELECT 
  USING (EXISTS (SELECT 1 FROM public.artifacts JOIN public.projects ON artifacts.project_id = projects.id 
                 WHERE artifacts.id = attributions.artifact_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can create attributions in their projects" ON public.attributions FOR INSERT 
  WITH CHECK (EXISTS (SELECT 1 FROM public.artifacts JOIN public.projects ON artifacts.project_id = projects.id 
                      WHERE artifacts.id = attributions.artifact_id AND projects.user_id = auth.uid()));

-- RLS Policies for citations
CREATE POLICY "Users can view citations in their projects" ON public.citations FOR SELECT 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = citations.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can create citations in their projects" ON public.citations FOR INSERT 
  WITH CHECK (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = citations.project_id AND projects.user_id = auth.uid()));

-- RLS Policies for lineage
CREATE POLICY "Users can view lineage in their projects" ON public.lineage FOR SELECT 
  USING (EXISTS (SELECT 1 FROM public.artifacts JOIN public.projects ON artifacts.project_id = projects.id 
                 WHERE artifacts.id = lineage.source_artifact_id AND projects.user_id = auth.uid()));

-- RLS Policies for reports
CREATE POLICY "Users can view reports in their projects" ON public.reports FOR SELECT 
  USING (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = reports.project_id AND projects.user_id = auth.uid()));
CREATE POLICY "Users can create reports in their projects" ON public.reports FOR INSERT 
  WITH CHECK (EXISTS (SELECT 1 FROM public.projects WHERE projects.id = reports.project_id AND projects.user_id = auth.uid()));

-- RLS Policies for audit events
CREATE POLICY "Users can view their own audit events" ON public.audit_events FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "System can create audit events" ON public.audit_events FOR INSERT WITH CHECK (true);

-- Indexes for performance
CREATE INDEX idx_sources_project_id ON public.sources(project_id);
CREATE INDEX idx_sources_content_hash ON public.sources(content_hash);
CREATE INDEX idx_artifacts_project_id ON public.artifacts(project_id);
CREATE INDEX idx_artifacts_source_id ON public.artifacts(source_id);
CREATE INDEX idx_artifacts_content_hash ON public.artifacts(content_hash);
CREATE INDEX idx_attributions_artifact_id ON public.attributions(artifact_id);
CREATE INDEX idx_attributions_source_id ON public.attributions(source_id);
CREATE INDEX idx_citations_project_id ON public.citations(project_id);
CREATE INDEX idx_lineage_source_artifact ON public.lineage(source_artifact_id);
CREATE INDEX idx_lineage_target_artifact ON public.lineage(target_artifact_id);
CREATE INDEX idx_reports_project_id ON public.reports(project_id);
CREATE INDEX idx_audit_events_user_id ON public.audit_events(user_id);
CREATE INDEX idx_audit_events_project_id ON public.audit_events(project_id);

-- Triggers for updated_at
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON public.projects FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;