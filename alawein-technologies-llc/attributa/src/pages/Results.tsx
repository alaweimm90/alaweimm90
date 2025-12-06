import { useEffect, useState, useMemo, Suspense, lazy } from 'react';
import { useParams, useLocation, useNavigate, Link } from 'react-router-dom';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Download, AlertTriangle, Shield, Code, BookOpen, FileText, ScrollText, ArrowLeft } from 'lucide-react';
import { useAppStore } from '@/store';
import { mockAnalyzeText, mockAnalyzeWatermark, mockAuditCitations, mockAnalyzeCode, mockComputeScore } from '@/services/realApi';
import OverviewTab from '@/components/results/OverviewTab';
import AnimatedGrid from '@/components/dev/AnimatedGrid';
import NeuralBackground from '@/components/dev/NeuralBackground';
 
 // Lazy load heavy analysis tabs
const SegmentsTab = lazy(() => import('@/components/results/SegmentsTab'));
const CitationsTab = lazy(() => import('@/components/results/CitationsTab'));
const CodeSecurityTab = lazy(() => import('@/components/results/CodeSecurityTab'));
const ExportsTab = lazy(() => import('@/components/results/ExportsTab'));
const DocumentViewer = lazy(() => import('@/components/results/DocumentViewer'));
import { useSEO } from '@/hooks/useSEO';

const Results = () => {
  const { docId } = useParams<{ docId: string }>();
  const { currentReport, setCurrentReport, analysisOptions, reports, updateReport, scoringWeights, pdfsByDocId, clearWorkspace } = useAppStore();
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const location = useLocation();
  const navigate = useNavigate();
  const locationState = location.state as { pdfs?: { name: string; url: string }[]; ephemeral?: boolean } | undefined;
  const navPdfs = locationState?.pdfs;
  const isEphemeral = locationState?.ephemeral === true;
  
  const dataURLtoBlob = (dataUrl: string) => {
    const [header, base64] = dataUrl.split(',');
    const mimeMatch = /data:(.*);base64/.exec(header);
    const mime = mimeMatch ? mimeMatch[1] : 'application/octet-stream';
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return new Blob([bytes], { type: mime });
  };

  const storedPdfs = useMemo(() => {
    const arr = (docId && pdfsByDocId[docId]) || [];
    try {
      return arr.map(({ name, dataUrl }) => ({ name, url: URL.createObjectURL(dataURLtoBlob(dataUrl)) }));
    } catch {
      return [] as { name: string; url: string }[];
    }
  }, [docId, pdfsByDocId]);

  useEffect(() => {
    return () => {
      storedPdfs.forEach(p => URL.revokeObjectURL(p.url));
    };
  }, [storedPdfs]);

  const pdfs = navPdfs && navPdfs.length > 0 ? navPdfs : storedPdfs;
  useSEO({
    title: 'Results — Attribution & Integrity Report | Attributa.dev',
    description: 'Review attribution signals, citations, and integrity risks with explainable evidence.'
  });

  useEffect(() => {
    if (!docId) return;

    // Always reset loader on doc change
    setIsLoading(false);

    // Ensure the currentReport matches the route param
    if (!currentReport || currentReport.docId !== docId) {
      const found = reports.find(r => r.docId === docId);
      if (found) setCurrentReport(found);
      // stop here; another render will run analysis with the correct report
      return;
    }

    // If analysis already present or no segments, ensure loader is off
    if (Object.keys(currentReport.signals || {}).length > 0 || currentReport.segments.length === 0) {
      setIsLoading(false);
      return;
    }

    // Run analysis if not already done
    const runAnalysis = async () => {
      setIsLoading(true);
      try {
        const signals: Record<string, unknown> = {};
        const scores: Record<string, unknown> = {};
        
        // Analyze each segment
        for (const segment of currentReport.segments) {
          // Text analysis
          const textAnalysis = await mockAnalyzeText(segment.segmentId);
          signals[segment.segmentId] = {
            gltr: textAnalysis.gltr,
            detectgpt: textAnalysis.detectgpt
          };
          
          // Watermark analysis if enabled and applicable
          if ((segment.type === 'prose' || segment.type === 'latex') && analysisOptions.tryWatermark) {
            const watermarkAnalysis = await mockAnalyzeWatermark(segment.segmentId);
            signals[segment.segmentId].watermark = watermarkAnalysis.watermark;
          }
          
          // Compute score with store weights
          const score = await mockComputeScore(segment.segmentId, {
            gltrTail: textAnalysis.gltr.tailTokenShare,
            gltrVar: textAnalysis.gltr.rankVariance,
            curvature: textAnalysis.detectgpt.curvature,
            watermarkP: signals[segment.segmentId].watermark?.pValue,
            lengthChars: segment.lengthChars,
            type: segment.type,
            weights: scoringWeights
          });
          scores[segment.segmentId] = score;
        }
        
        // Citation analysis
        let citations: unknown[] = [];
        if (currentReport.segments.some(s => s.type === 'latex' || s.type === 'prose')) {
          const citationAnalysis = await mockAuditCitations(docId!);
          citations = citationAnalysis.citations;
        }

        // Code analysis
        let codeFindings: unknown[] = [];
        if (currentReport.segments.some(s => s.type === 'code')) {
          const codeAnalysis = await mockAnalyzeCode(docId!);
          codeFindings = codeAnalysis.findings;
        }
        
        const updatedReport = {
          ...currentReport,
          signals,
          scores,
          citations,
          codeFindings
        } as typeof currentReport;
        setCurrentReport(updatedReport);
        updateReport(updatedReport);
      } catch (error) {
        console.error('Analysis failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    runAnalysis();
  }, [docId, currentReport, reports, setCurrentReport, analysisOptions, updateReport, scoringWeights]);

  // Auto-clear ephemeral results when leaving the page
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (isEphemeral) {
        clearWorkspace();
      }
    };

    const handlePopState = () => {
      if (isEphemeral) {
        clearWorkspace();
      }
    };

    if (isEphemeral) {
      window.addEventListener('beforeunload', handleBeforeUnload);
      window.addEventListener('popstate', handlePopState);

      return () => {
        window.removeEventListener('beforeunload', handleBeforeUnload);
        window.removeEventListener('popstate', handlePopState);
      };
    }
  }, [isEphemeral, clearWorkspace]);

  if (!currentReport || currentReport.docId !== docId) {
    const exists = reports.some(r => r.docId === docId);
    if (!exists) {
      return (
        <div className="container mx-auto px-4 py-8">
          <Card>
            <CardContent className="flex items-center justify-center py-12">
              <div className="text-center space-y-2">
                <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto" />
                <h3 className="text-lg font-semibold">Document not found</h3>
                <p className="text-muted-foreground">The requested analysis could not be found.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }
    return (
      <div className="container mx-auto px-4 py-8">
        <Card>
          <CardContent className="flex items-center justify-center py-12">
            <div className="text-center space-y-2">
              <div className="h-4 w-32 bg-accent animate-pulse rounded mx-auto" />
              <p className="text-muted-foreground">Loading report…</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const hasCode = currentReport.segments.some(s => s.type === 'code');
  const hasCitations = currentReport.segments.some(s => s.type === 'latex' || s.type === 'prose');

  return (
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary">
      <NeuralBackground />
      <AnimatedGrid />

      <div className="relative z-10 container mx-auto px-4 py-10 max-w-6xl">
        <div className="space-y-6 animate-fade-in">
          {/* Ephemeral results banner */}
          {isEphemeral && (
            <div className="p-4 border bg-warning/10 border-warning/20 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-warning" />
                  <span className="text-sm font-medium text-info">
                    Local Analysis Results
                  </span>
                </div>
                <Button size="sm" asChild>
                  <Link to="/scan">New Analysis</Link>
                </Button>
              </div>
              <p className="text-xs text-info mt-1">
                Privacy-first: analysis performed locally in your browser.
              </p>
            </div>
          )}

          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-3">
                <Button variant="ghost" size="sm" asChild>
                  <Link to="/scan" className="flex items-center gap-2">
                    <ArrowLeft className="h-4 w-4" />
                    Back to Analysis
                  </Link>
                </Button>
                <h1 className="text-3xl font-bold tracking-tight">Analysis Results</h1>
              </div>
              <p className="text-xs text-muted-foreground">
                ID: {currentReport.docId}
              </p>
            </div>
            
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="bg-success/10 text-success border-success/20">
                Local‑first analysis
              </Badge>
              {!isEphemeral && (
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export Report
                </Button>
              )}
            </div>
          </div>

          {/* Loading indicator */}
          {isLoading && (
            <Card className="glass-card animate-fade-in">
              <CardContent className="py-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Running analysis...</span>
                    <span className="text-sm text-muted-foreground">Processing segments</span>
                  </div>
                  <Progress value={33} className="h-2" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Privacy banner */}
          <div className="glass-card rounded-lg p-4">
            <p className="text-sm text-info">
              <Shield className="h-4 w-4 inline mr-2" />
              Analysis is probabilistic; no binary attribution. 
              Confidence reflects signal strength and input length, not authorship certainty.
            </p>
          </div>

          {/* Results tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-6 glass-card p-1 rounded-lg">
              <TabsTrigger value="overview" className="flex items-center space-x-2 rounded-md data-[state=active]:bg-primary/15 data-[state=active]:text-primary">
                <FileText className="h-4 w-4" />
                <span>Overview</span>
              </TabsTrigger>
              <TabsTrigger value="document" className="flex items-center space-x-2 rounded-md data-[state=active]:bg-primary/15 data-[state=active]:text-primary">
                <ScrollText className="h-4 w-4" />
                <span>Document</span>
              </TabsTrigger>
              <TabsTrigger value="segments" className="flex items-center space-x-2 rounded-md data-[state=active]:bg-primary/15 data-[state=active]:text-primary">
                <Shield className="h-4 w-4" />
                <span>Segments</span>
              </TabsTrigger>
              {hasCitations && !isEphemeral && (
                <TabsTrigger value="citations" className="flex items-center space-x-2 rounded-md data-[state=active]:bg-primary/15 data-[state=active]:text-primary">
                  <BookOpen className="h-4 w-4" />
                  <span>Citations</span>
                </TabsTrigger>
              )}
              {hasCode && (
                <TabsTrigger value="code" className="flex items-center space-x-2 rounded-md data-[state=active]:bg-primary/15 data-[state=active]:text-primary">
                  <Code className="h-4 w-4" />
                  <span>Code Security</span>
                </TabsTrigger>
              )}
              {!isEphemeral && (
                <TabsTrigger value="exports" className="flex items-center space-x-2 rounded-md data-[state=active]:bg-primary/15 data-[state=active]:text-primary">
                  <Download className="h-4 w-4" />
                  <span>Exports</span>
                </TabsTrigger>
              )}
            </TabsList>

            <TabsContent value="overview">
              <OverviewTab report={currentReport} />
            </TabsContent>

            <TabsContent value="document">
              <Suspense fallback={<div className="flex justify-center p-8"><div className="animate-spin h-6 w-6 border-2 border-primary/20 border-t-primary rounded-full" /></div>}>
                <DocumentViewer report={currentReport} pdfs={pdfs} />
              </Suspense>
            </TabsContent>

            <TabsContent value="segments">
              <Suspense fallback={<div className="flex justify-center p-8"><div className="animate-spin h-6 w-6 border-2 border-primary/20 border-t-primary rounded-full" /></div>}>
                <SegmentsTab report={currentReport} />
              </Suspense>
            </TabsContent>

            {hasCitations && !isEphemeral && (
              <TabsContent value="citations">
                <Suspense fallback={<div className="flex justify-center p-8"><div className="animate-spin h-6 w-6 border-2 border-primary/20 border-t-primary rounded-full" /></div>}>
                  <CitationsTab report={currentReport} />
                </Suspense>
              </TabsContent>
            )}

            {hasCode && (
              <TabsContent value="code">
                <Suspense fallback={<div className="flex justify-center p-8"><div className="animate-spin h-6 w-6 border-2 border-primary/20 border-t-primary rounded-full" /></div>}>
                  <CodeSecurityTab report={currentReport} />
                </Suspense>
              </TabsContent>
            )}

            {!isEphemeral && (
              <TabsContent value="exports">
                <Suspense fallback={<div className="flex justify-center p-8"><div className="animate-spin h-6 w-6 border-2 border-primary/20 border-t-primary rounded-full" /></div>}>
                  <ExportsTab report={currentReport} />
                </Suspense>
              </TabsContent>
            )}
          </Tabs>
        </div>
      </div>
    </main>
  );
};

export default Results;