import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Download, FileText, Code2, Copy, Check } from 'lucide-react';
import { Report } from '@/types';
import { useToast } from '@/hooks/use-toast';

interface ExportsTabProps {
  report: Report;
}

const ExportsTab = ({ report }: ExportsTabProps) => {
  const { toast } = useToast();
  const [copiedFormat, setCopiedFormat] = useState<string | null>(null);

  const generateMarkdownReport = () => {
    const timestamp = new Date().toISOString();
    const allScores = Object.values(report.scores || {});
    const avgScore = allScores.length ? allScores.reduce((sum, s) => sum + (s?.score || 0), 0) / allScores.length : 0;
    const confidence = avgScore >= 0.65 ? 'High' : avgScore >= 0.45 ? 'Medium' : 'Low';

    return `# Attributa Analysis Report

**Document ID:** ${report.docId}  
**Generated:** ${timestamp}  
**Analysis Version:** MVP (Mocked)

## Executive Summary

- **Total Characters:** ${report.summary.totalChars.toLocaleString()}
- **Segments Analyzed:** ${report.summary.numSegments}
- **Attribution Confidence:** ${(avgScore * 100).toFixed(0)}% (${confidence} confidence)
- **Content Types:** ${Object.entries(report.summary.types).map(([type, count]) => `${count} ${type}`).join(', ')}

## Segment Analysis

${report.segments.map(segment => {
  const score = report.scores?.[segment.segmentId];
  const signals = report.signals?.[segment.segmentId];
  
  return `### ${segment.title || 'Untitled'} (${segment.type})

**Length:** ${segment.lengthChars.toLocaleString()} characters  
**Attribution Score:** ${score ? (score.score * 100).toFixed(0) + '%' : 'Analyzing...'}  
**Confidence:** ${score?.confidence || 'Pending'}

${signals?.gltr ? `**GLTR Analysis:**
- Tail Token Share: ${(signals.gltr.tailTokenShare * 100).toFixed(1)}%
- Rank Variance: ${signals.gltr.rankVariance.toFixed(3)}
- Token Distribution: ${signals.gltr.histogram.map((v, i) => `${['Green', 'Yellow', 'Red', 'Purple'][i]}: ${(v * 100).toFixed(1)}%`).join(', ')}
` : ''}

${signals?.detectgpt ? `**DetectGPT Analysis:**
- Curvature: ${signals.detectgpt.curvature.toFixed(3)}
- Perturbations: ${signals.detectgpt.numPerturbations}
` : ''}

${signals?.watermark?.pValue ? `**Watermark Detection:**
- p-value: ${signals.watermark.pValue.toFixed(4)}
- Status: ${signals.watermark.pValue < 0.01 ? 'Detected' : 'Not Detected'}
` : ''}

${score?.rationale ? `**Key Findings:**
${score.rationale.map(r => `- ${r}`).join('\n')}
` : ''}
`;
}).join('\n')}

${report.citations && report.citations.length > 0 ? `## Citation Analysis

**Validity Rate:** ${((report.citations.filter(c => c.resolves).length / report.citations.length) * 100).toFixed(1)}%

${report.citations.map(citation => `
### ${citation.raw}
- **Status:** ${citation.resolves ? '✅ Valid' : '❌ Invalid'}
${citation.suggestions ? `- **Suggestions:** ${citation.suggestions.map(s => `${s.title} (${(s.confidence * 100).toFixed(0)}% match)`).join(', ')}` : ''}
`).join('')}
` : ''}

${report.codeFindings && report.codeFindings.length > 0 ? `## Security Analysis

**Total Findings:** ${report.codeFindings.length}  
**Findings per KLOC:** ${((report.codeFindings.length / 1000) * 1000).toFixed(1)}

${report.codeFindings.map(finding => `
### ${finding.cwe}: ${finding.rule}
- **File:** ${finding.path}:${finding.line}
- **Severity:** ${finding.severity}
- **Code:** \`${finding.snippet}\`
`).join('')}
` : ''}

## Disclaimer

This tool provides probabilistic signals consistent with AI-generated text. Results may be unreliable on short, paraphrased, or templated content. Do not use as a sole basis for academic or disciplinary decisions.

**Analysis Method:** Local/Mocked analysis (MVP version)  
**Privacy:** No external API calls made, all processing local
`;
  };

  const generateJsonReport = () => {
    return JSON.stringify({
      docId: report.docId,
      timestamp: new Date().toISOString(),
      version: "MVP-mocked",
      summary: report.summary,
      segments: report.segments.map(segment => ({
        ...segment,
        signals: report.signals?.[segment.segmentId],
        score: report.scores?.[segment.segmentId]
      })),
      citations: report.citations,
      codeFindings: report.codeFindings,
      metadata: {
        analysisType: "local-mock",
        disclaimer: "Probabilistic analysis - not definitive attribution"
      }
    }, null, 2);
  };

  const handleCopy = async (content: string, format: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedFormat(format);
      toast({
        title: "Copied to clipboard",
        description: `${format} report copied successfully`,
      });
      setTimeout(() => setCopiedFormat(null), 2000);
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Could not copy to clipboard",
        variant: "destructive",
      });
    }
  };

  const handleDownload = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Download started",
      description: `${filename} has been downloaded`,
    });
  };

  const markdownReport = generateMarkdownReport();
  const jsonReport = generateJsonReport();

  return (
    <div className="space-y-6">
      {/* Export Options */}
      <Card>
        <CardHeader>
          <CardTitle>Export Options</CardTitle>
          <CardDescription>
            Download or copy analysis results in various formats
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Markdown Export */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <FileText className="h-5 w-5" />
                <h3 className="font-medium">Markdown Report</h3>
                <Badge variant="secondary">Human-readable</Badge>
              </div>
              
              <p className="text-sm text-muted-foreground">
                Formatted report suitable for documentation, sharing, or further editing.
              </p>
              
              <div className="flex space-x-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => handleCopy(markdownReport, 'Markdown')}
                  className="flex items-center space-x-2"
                >
                  {copiedFormat === 'Markdown' ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                  <span>Copy</span>
                </Button>
                
                <Button 
                  size="sm"
                  onClick={() => handleDownload(markdownReport, `attributa-report-${report.docId}.md`, 'text/markdown')}
                  className="flex items-center space-x-2"
                >
                  <Download className="h-4 w-4" />
                  <span>Download</span>
                </Button>
              </div>
            </div>

            {/* JSON Export */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Code2 className="h-5 w-5" />
                <h3 className="font-medium">JSON Report</h3>
                <Badge variant="secondary">Machine-readable</Badge>
              </div>
              
              <p className="text-sm text-muted-foreground">
                Structured data format for programmatic analysis or integration with other tools.
              </p>
              
              <div className="flex space-x-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => handleCopy(jsonReport, 'JSON')}
                  className="flex items-center space-x-2"
                >
                  {copiedFormat === 'JSON' ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                  <span>Copy</span>
                </Button>
                
                <Button 
                  size="sm"
                  onClick={() => handleDownload(jsonReport, `attributa-report-${report.docId}.json`, 'application/json')}
                  className="flex items-center space-x-2"
                >
                  <Download className="h-4 w-4" />
                  <span>Download</span>
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Preview */}
      <Card>
        <CardHeader>
          <CardTitle>Markdown Preview</CardTitle>
          <CardDescription>
            Preview of the exported markdown report
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            value={markdownReport}
            readOnly
            rows={20}
            className="font-mono text-xs resize-none"
          />
        </CardContent>
      </Card>

      {/* JSON Preview */}
      <Card>
        <CardHeader>
          <CardTitle>JSON Preview</CardTitle>
          <CardDescription>
            Preview of the exported JSON data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            value={jsonReport}
            readOnly
            rows={15}
            className="font-mono text-xs resize-none"
          />
        </CardContent>
      </Card>

      {/* Export Metadata */}
      <Card>
        <CardHeader>
          <CardTitle>Export Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <h4 className="font-medium">Report Information</h4>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Document ID: {report.docId}</li>
                <li>• Analysis Version: MVP (Mocked)</li>
                <li>• Timestamp: {new Date().toISOString()}</li>
                <li>• Format: Markdown & JSON</li>
              </ul>
            </div>
            
            <div className="space-y-2">
              <h4 className="font-medium">Privacy & Disclaimer</h4>
              <ul className="space-y-1 text-muted-foreground">
                <li>• No external API calls made</li>
                <li>• Analysis performed locally</li>
                <li>• Results are probabilistic</li>
                <li>• Not for definitive attribution</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ExportsTab;