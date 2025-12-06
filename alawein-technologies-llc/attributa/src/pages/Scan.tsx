import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Clipboard, Github, FileText, Code, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { extractTextFromPDF } from '@/lib/pdfExtractor';
import { useToast } from '@/hooks/use-toast';
import { useSEO } from '@/hooks/useSEO';
import { useAuth } from '@/hooks/useAuth';
import { useCurrentProject } from '@/hooks/useProjects';
import { useIngest } from '@/hooks/useAttribution';
import ProjectSelector from '@/components/projects/ProjectSelector';
import { useAppStore } from '@/store';
import { mockIngest } from '@/services/realApi';
import NeuralBackground from '@/components/dev/NeuralBackground';
import AnimatedGrid from '@/components/dev/AnimatedGrid';

const sampleDocs = {
  academic: {
    title: "Sample Academic Prose",
    content: `We present a novel approach to deep learning architectures that significantly improves performance on natural language understanding tasks. Our method leverages transformer-based attention mechanisms combined with graph neural networks to achieve state-of-the-art results across multiple benchmarks.

The fundamental challenge in modern NLP lies in capturing both local linguistic patterns and global semantic relationships. Traditional approaches have focused primarily on sequential modeling, which fails to capture the inherent graph structure of language. Our architecture addresses this limitation by incorporating explicit graph representations into the attention computation.

We evaluate our approach on several standard datasets including GLUE, SuperGLUE, and custom evaluation suites. The results demonstrate consistent improvements of 3-7% over baseline methods across all tasks. Particularly notable are the improvements on reading comprehension and natural language inference tasks, where our model achieves new state-of-the-art performance.

References:
1. Smith, J. et al. (2023). "Advanced Transformer Architectures for NLP." Proceedings of NeurIPS 2023. doi:10.1234/nonexistent.2023.001
2. Johnson, M. and Brown, K. (2022). "Graph-based Language Models." Journal of AI Research, 15(3), 123-145. doi:10.5555/fake.doi.2022.456
3. Davis, R. (2023). "Attention Mechanisms in Deep Learning." Nature Machine Intelligence, 4(2), 78-92. doi:10.1038/fictional.2023.789`,
    type: "prose" as const
  },
  code: {
    title: "Sample Code Repository",
    content: `# app.py - Web application with security issues
import subprocess
import os
from flask import Flask, request

app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    # CWE-78: OS Command Injection vulnerability
    result = subprocess.run("grep " + query + " /var/log/app.log", shell=True, capture_output=True)
    return result.stdout.decode()

@app.route('/upload')  
def upload():
    filename = request.args.get('file', '')
    # CWE-22: Path Traversal vulnerability
    with open('/uploads/' + filename, 'r') as f:
        return f.read()

if __name__ == '__main__':
    # CWE-489: Debug mode in production
    app.run(debug=True, host='0.0.0.0')`,
    type: "code" as const
  }
};

const Scan = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { currentProject } = useCurrentProject();
  const ingestMutation = useIngest();
  const { analysisOptions, updateAnalysisOptions, addReport } = useAppStore();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<'paste' | 'upload' | 'github' | 'samples'>('paste');
  const [content, setContent] = useState('');
  const [githubUrl, setGithubUrl] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);

  useSEO({
    title: 'Analyze Content — Attribution Intelligence | Attributa.dev',
    description: 'Upload text, code, or documents for transparent attribution analysis. Understand collaboration patterns with uncertainty quantification.'
  });

  const handleAnalyze = async () => {
    // Require some content
    if (!content.trim() && selectedFiles.length === 0 && !githubUrl.trim()) {
      toast({ title: 'No content provided', description: 'Please provide content to analyze.', variant: 'destructive' });
      return;
    }

    // Quick Scan (no project selected): fully client-side, ephemeral
      if (!currentProject) {
        if (activeTab === 'github') {
          toast({ title: 'Workspace needed', description: 'GitHub analysis uses a workspace. Sign in (optional) to create one.' });
          return;
        }

      // Limits
      if (activeTab === 'paste' && content.length > 10000) {
        toast({ title: 'Limit exceeded', description: 'Quick Scan supports up to 10,000 characters.', variant: 'destructive' });
        return;
      }
      if (activeTab === 'upload') {
        if (selectedFiles.length === 0) {
          toast({ title: 'No file selected', description: 'Choose a file to analyze.', variant: 'destructive' });
          return;
        }
        if (selectedFiles.length > 1) {
          toast({ title: 'One file at a time', description: 'Quick Scan supports 1 file only.', variant: 'destructive' });
          return;
        }
        const f = selectedFiles[0];
        if (f.size > 2 * 1024 * 1024) {
          toast({ title: 'File too large', description: 'Max file size for Quick Scan is 2MB.', variant: 'destructive' });
          return;
        }
      }

      try {
        // Prepare file content if needed
        let processedFiles: { name: string; content: string }[] | undefined;
        if (activeTab === 'upload' && selectedFiles.length > 0) {
          const file = selectedFiles[0];
          const text = file.type === 'application/pdf' ? await extractTextFromPDF(file) : await file.text();
          processedFiles = [{ name: file.name, content: text }];
        }

        const ingestReq = {
          source: activeTab === 'upload' ? 'file' as const : 'paste' as const,
          content: activeTab === 'paste' ? content.trim() : undefined,
          files: processedFiles,
          options: analysisOptions,
        };

        const result = await mockIngest(ingestReq as Parameters<typeof mockIngest>[0]);

        // Create ephemeral report in local store
        const report = {
          docId: result.docId,
          createdAt: Date.now(),
          summary: result.summary,
          segments: result.segments,
          signals: {},
          scores: {},
        };
        addReport(report as Parameters<typeof addReport>[0]);

        const pdfs = selectedFiles
          .filter(f => f.type === 'application/pdf')
          .map(f => ({ name: f.name, url: URL.createObjectURL(f) }));

        navigate(`/results/${result.docId}`, { state: { ephemeral: true, pdfs } });
      } catch (error: unknown) {
        console.error('Quick Scan failed:', error);
        const message = error instanceof Error ? error.message : 'Please try again.';
        toast({ title: 'Quick Scan failed', description: message, variant: 'destructive' });
      }
      return;
    }

    // Workspace-backed ingestion (saved results)
    setUploadProgress(0);
    try {
      // Step 1: Prepare files (with progress up to 70%)
      let processedFiles: { name: string; content: string }[] | undefined;
      if (selectedFiles.length > 0) {
        processedFiles = [];
        const total = selectedFiles.length;
        let idx = 0;
        for (const file of selectedFiles) {
          let fileContent = '';
          if (file.type === 'application/pdf') {
            fileContent = await extractTextFromPDF(file);
          } else {
            fileContent = await file.text();
          }
          processedFiles.push({
            name: file.name,
            content: fileContent,
          });
          idx++;
          setUploadProgress(Math.min(70, Math.round((idx / total) * 70)));
        }
      }

      const request = {
        projectId: currentProject.id,
        source: activeTab === 'github' ? 'github' as const : activeTab === 'upload' ? 'file' as const : 'paste' as const,
        content: content.trim() || undefined,
        githubUrl: githubUrl.trim() || undefined,
        files: processedFiles,
        title: selectedFiles[0]?.name || 'Uploaded Content',
        metadata: { analysisOptions }
      };

      setUploadProgress(90);
      const result = await ingestMutation.mutateAsync(request);
      setUploadProgress(100);

      toast({ title: 'Content ingested successfully', description: `${result.artifactCount} artifacts created` });
      navigate(`/workspace?project=${currentProject.id}`);
    } catch (error: unknown) {
      console.error('Analysis failed:', error);
      const message = error instanceof Error ? error.message : 'Please try again.';
      toast({ title: 'Analysis failed', description: message, variant: 'destructive' });
    }
  };
  const handleSampleSelect = (sampleKey: keyof typeof sampleDocs) => {
    const sample = sampleDocs[sampleKey];
    setContent(sample.content);
    setActiveTab('paste');
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      const arr = Array.from(files);
      setSelectedFiles(arr);
      const pdfCount = arr.filter(f => f.type === 'application/pdf').length;
      if (arr.length > 0) {
        toast({ title: 'Files selected', description: `${arr.length} file${arr.length > 1 ? 's' : ''} chosen${pdfCount ? ` • ${pdfCount} PDF${pdfCount > 1 ? 's' : ''}` : ''}.` });
      }
    }
  };

  // Demo mode - allow analysis without authentication for showcasing
  // Note: In production, you may want to require authentication

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary engineering-pattern">
      <NeuralBackground />
      <AnimatedGrid />
      
      <div className="relative z-10 container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-8">
          <div className="text-center space-y-4 animate-fade-in">
            <h1 className="text-3xl font-bold tracking-tight display-tight font-mono">Analyze Your Content</h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload text, code, or documents to analyze attribution and detect AI generation signals.
          </p>
        </div>

        <ProjectSelector />

        {!currentProject && (
          <Card className="glass-card border-warning/30">
            <CardHeader>
              <CardTitle>Local Analysis</CardTitle>
              <CardDescription>
                Analyze content locally with full privacy. Max 10k chars • 3 files • 2MB limit.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary" className="bg-success/10 text-success border-success/20">Local-only</Badge>
                <Badge variant="secondary">Max 10k chars</Badge>
                <Badge variant="secondary">1 file ≤ 2MB</Badge>
              </div>
              <div className="mt-4">
                <Button size="sm" onClick={() => navigate('/auth')}>Enable sync (optional)</Button>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="space-y-8">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Input Source</CardTitle>
              <CardDescription>
                Choose how to provide content for analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as typeof activeTab)}>
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="paste" className="flex items-center space-x-2">
                    <Clipboard className="h-4 w-4" />
                    <span>Paste</span>
                  </TabsTrigger>
                  <TabsTrigger value="upload" className="flex items-center space-x-2">
                    <Upload className="h-4 w-4" />
                    <span>Upload</span>
                  </TabsTrigger>
                  <TabsTrigger value="github" className="flex items-center space-x-2">
                    <Github className="h-4 w-4" />
                    <span>GitHub</span>
                  </TabsTrigger>
                  <TabsTrigger value="samples" className="flex items-center space-x-2">
                    <FileText className="h-4 w-4" />
                    <span>Samples</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="paste" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="content">Content to analyze</Label>
                    <Textarea
                      id="content"
                      placeholder="Paste your text, code, or LaTeX content here... (minimum 100 characters for reliable analysis)"
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      rows={12}
                      className="font-mono text-sm"
                    />
                    <p className="text-xs text-muted-foreground">
                      {content.length} characters • Supports up to 200,000 characters
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="upload" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="files">Upload files</Label>
                    <Input
                      id="files"
                      type="file"
                      onChange={handleFileSelect}
                      multiple
                      accept=".txt,.md,.py,.ipynb,.tex,.bib,.pdf"
                    />
                    <p className="text-xs text-muted-foreground">
                      Supported: .txt, .md, .py, .ipynb, .tex, .bib, .pdf
                    </p>
                    {selectedFiles.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-sm font-medium">Selected files:</p>
                        {selectedFiles.map((file, index) => (
                          <Badge key={index} variant="secondary" className="mr-2">
                            {file.name} ({(file.size / 1024).toFixed(1)} KB)
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="github" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="github">GitHub Repository URL</Label>
                    <Input
                      id="github"
                      placeholder="https://github.com/username/repository"
                      value={githubUrl}
                      onChange={(e) => setGithubUrl(e.target.value)}
                    />
                    <Badge variant="outline" className="w-fit">
                      {currentProject ? 'Coming soon: Full repository analysis' : 'Workspace needed: GitHub analysis uses a project (optional sign‑in)'}
                    </Badge>
                  </div>
                </TabsContent>

                <TabsContent value="samples" className="space-y-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <Card className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => handleSampleSelect('academic')}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center space-x-2">
                          <BookOpen className="h-5 w-5 text-primary" />
                          <CardTitle className="text-lg">Academic Prose</CardTitle>
                        </div>
                        <CardDescription>
                          Research paper with non-resolving citations and polished academic writing
                        </CardDescription>
                      </CardHeader>
                    </Card>
                    
                    <Card className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => handleSampleSelect('code')}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center space-x-2">
                          <Code className="h-5 w-5 text-primary" />
                          <CardTitle className="text-lg">Code Repository</CardTitle>
                        </div>
                        <CardDescription>
                          Python Flask app with security vulnerabilities (CWE-78, CWE-22)
                        </CardDescription>
                      </CardHeader>
                    </Card>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Analysis Options</CardTitle>
              <CardDescription>
                Configure how the analysis should be performed
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="local-only">Analyze with local model only</Label>
                  <p className="text-sm text-muted-foreground">
                    Process content locally without external API calls (recommended for privacy)
                  </p>
                </div>
                <Switch
                  id="local-only"
                  checked={analysisOptions.useLocalOnly}
                  onCheckedChange={(checked) => updateAnalysisOptions({ useLocalOnly: checked })}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="watermark">Try watermark test (experimental)</Label>
                  <p className="text-sm text-muted-foreground">
                    Attempt to detect statistical watermarks in the text
                  </p>
                </div>
                <Switch
                  id="watermark"
                  checked={analysisOptions.tryWatermark}
                  onCheckedChange={(checked) => updateAnalysisOptions({ tryWatermark: checked })}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="external-apis">Use external APIs if keys present</Label>
                  <p className="text-sm text-muted-foreground">
                    Enable external model calls when API keys are configured
                  </p>
                </div>
                <Switch
                  id="external-apis"
                  checked={analysisOptions.useExternalApis}
                  onCheckedChange={(checked) => updateAnalysisOptions({ useExternalApis: checked })}
                />
              </div>
            </CardContent>
          </Card>

          {ingestMutation.isPending && (
            <div className="space-y-2 max-w-md mx-auto">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Processing content...</span>
                <span className="text-sm text-muted-foreground">{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
          )}

          <div className="flex justify-center">
            <Button
              onClick={handleAnalyze}
              disabled={
                (!content.trim() && selectedFiles.length === 0 && !githubUrl.trim()) ||
                ingestMutation.isPending
              }
              size="lg"
              className="px-8"
            >
              {ingestMutation.isPending ? 'Analyzing...' : 'Start Analysis'}
            </Button>
          </div>
        </div>
      </div>
    </div>
    </div>
  );
};

export default Scan;