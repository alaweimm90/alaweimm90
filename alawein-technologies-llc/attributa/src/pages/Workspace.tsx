import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { formatDistance } from 'date-fns';
import { History, FileText, Code, BookOpen, Search, Trash2, Eye, Plus, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useSEO } from '@/hooks/useSEO';
import { useAuth } from '@/hooks/useAuth';
import { useCurrentProject } from '@/hooks/useProjects';
import { useProjectArtifacts, useProjectSources, useAttributions } from '@/hooks/useAttribution';
import { useAnalysis } from '@/hooks/useAnalysis';
import ProjectSelector from '@/components/projects/ProjectSelector';
import EnhancedAnalysisPanel from '@/components/attribution/EnhancedAnalysisPanel';
import NeuralBackground from '@/components/dev/NeuralBackground';
import AnimatedGrid from '@/components/dev/AnimatedGrid';
const Workspace = () => {
  const { user } = useAuth();
  const { currentProject } = useCurrentProject();
  const [searchParams] = useSearchParams();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);

  const { data: artifacts = [], isLoading: artifactsLoading } = useProjectArtifacts(currentProject?.id);
  const { data: sources = [], isLoading: sourcesLoading } = useProjectSources(currentProject?.id);
  const { data: attributions = [], isLoading: attributionsLoading } = useAttributions(currentProject?.id);
  const { runAnalysis, isAnalyzing } = useAnalysis();

  useSEO({
    title: 'Workspace — Projects & Reports | Attributa.dev',
    description: 'Manage analyses, compare results, and export reports.'
  });

  const filteredArtifacts = artifacts.filter(artifact => 
    artifact.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    artifact.content.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getTypeIcon = (type: string) => {
    if (type === 'code') return <Code className="h-4 w-4" />;
    if (type === 'latex') return <BookOpen className="h-4 w-4" />;
    return <FileText className="h-4 w-4" />;
  };

  const getAttributionBadge = (artifactId: string) => {
    const artifactAttributions = attributions.filter(attr => attr.artifact_id === artifactId);
    if (artifactAttributions.length === 0) {
      return <Badge variant="outline">No attributions</Badge>;
    }

    const highConfidence = artifactAttributions.filter(attr => attr.confidence_level === 'High').length;
    const mediumConfidence = artifactAttributions.filter(attr => attr.confidence_level === 'Medium').length;
    
    if (highConfidence > 0) {
      return <Badge className="bg-destructive/10 text-destructive border-destructive/20">{highConfidence} High Risk</Badge>;
    } else if (mediumConfidence > 0) {
      return <Badge className="bg-warning/10 text-warning border-warning/20">{mediumConfidence} Medium Risk</Badge>;
    } else {
      return <Badge className="bg-success/10 text-success border-success/20">Low Risk</Badge>;
    }
  };

  // Require authentication
  if (!user) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <Card className="glass-card">
          <CardContent className="text-center py-12">
            <h2 className="text-3xl font-bold tracking-tight mb-4">Workspace</h2>
            <p className="text-muted-foreground mb-6">Sign in to use the workspace for cloud sync and exports. You can still use Quick Scan without an account.</p>
            <Button asChild>
              <a href="/auth">Sign in</a>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!currentProject) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-6">
          <div className="text-center space-y-4">
            <History className="h-12 w-12 text-muted-foreground mx-auto" />
            <h1 className="text-3xl font-bold tracking-tight">Workspace</h1>
            <p className="text-muted-foreground max-w-md mx-auto">
              Select or create a project to start analyzing content and viewing results.
            </p>
          </div>
          <ProjectSelector />
        </div>
      </div>
    );
  }

  if (artifactsLoading || sourcesLoading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <Card className="glass-card">
          <CardContent className="text-center py-12">
            <div className="space-y-2">
              <div className="h-4 w-32 bg-accent animate-pulse rounded mx-auto" />
              <p className="text-muted-foreground">Loading workspace...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (artifacts.length === 0) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-6">
          <ProjectSelector />
          <div className="text-center space-y-4">
            <History className="h-12 w-12 text-muted-foreground mx-auto" />
            <h1 className="text-3xl font-bold tracking-tight">No Content Yet</h1>
            <p className="text-muted-foreground max-w-md mx-auto">
              Start by analyzing some content to see results in your workspace.
            </p>
            <Button asChild>
              <Link to="/scan">Analyze Content</Link>
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <NeuralBackground />
      <AnimatedGrid />
      <div className="space-y-6 animate-fade-in">
        <ProjectSelector />
        
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h1 className="text-3xl font-bold tracking-tight">Workspace</h1>
                <p className="text-muted-foreground">
                  {artifacts.length} artifact{artifacts.length === 1 ? '' : 's'}, {sources.length} source{sources.length === 1 ? '' : 's'}, {attributions.length} attribution{attributions.length === 1 ? '' : 's'}
                </p>
              </div>

              <div className="flex items-center space-x-2">
                <Button 
                  onClick={() => runAnalysis(undefined)} 
                  disabled={isAnalyzing}
                  variant="outline" 
                  size="sm"
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    'Run Analysis'
                  )}
                </Button>
                <Button asChild>
                  <Link to="/scan">
                    <Plus className="h-4 w-4 mr-2" />
                    New Analysis
                  </Link>
                </Button>
              </div>
            </div>

        {/* Search */}
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search analyses..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>

            {/* Artifacts grid */}
            <div className="grid md:grid-cols-2 gap-4">
              {filteredArtifacts.map((artifact) => (
                <Card 
                  key={artifact.id} 
                  className={`glass-card hover:shadow-md transition-shadow cursor-pointer ${
                    selectedArtifactId === artifact.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedArtifactId(artifact.id)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-2">
                        {getTypeIcon(artifact.artifact_type)}
                        <div className="space-y-1">
                          <CardTitle className="text-lg line-clamp-1">
                            {artifact.title || 'Untitled Artifact'}
                          </CardTitle>
                          <CardDescription className="text-xs">
                            {artifact.artifact_type} • {artifact.char_length.toLocaleString()} chars
                          </CardDescription>
                        </div>
                      </div>
                      {getAttributionBadge(artifact.id)}
                    </div>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <div className="text-sm text-muted-foreground">
                      {artifact.content.substring(0, 100)}
                      {artifact.content.length > 100 && '...'}
                    </div>

                    <div className="flex items-center justify-between pt-2">
                      <span className="text-xs text-muted-foreground">
                        {formatDistance(new Date(artifact.created_at), new Date(), { addSuffix: true })}
                      </span>
                      
                      <Button size="sm" variant="outline">
                        <Eye className="h-3 w-3 mr-1" />
                        Analyze
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {filteredArtifacts.length === 0 && searchQuery && (
              <div className="text-center py-12">
                <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No results found</h3>
                <p className="text-muted-foreground">
                  Try adjusting your search query or clear it to see all artifacts.
                </p>
              </div>
            )}
          </div>

          {/* Analysis Panel */}
          <div className="lg:col-span-1">
            <EnhancedAnalysisPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Workspace;