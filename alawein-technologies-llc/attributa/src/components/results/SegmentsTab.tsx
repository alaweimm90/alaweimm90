import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { FileText, Code, BookOpen, AlertTriangle, BarChart3, Wand2, Info } from 'lucide-react';
import { Report, Segment, Finding } from '@/types';
import { shouldShowReliabilityWarning, getLowConfidenceExplanation } from '@/lib/segmentation';
import { hasCounterEvidence } from '@/lib/scoring';
import { Modal, ModalContent, ModalDescription, ModalHeader, ModalTitle, ModalTrigger } from '@/components/ui/modal';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface SegmentsTabProps {
  report: Report;
}

const SegmentsTab = ({ report }: SegmentsTabProps) => {
  const [selectedSegment, setSelectedSegment] = useState<string | null>(
    report.segments[0]?.segmentId || null
  );

  const selectedSegmentData = report.segments.find(s => s.segmentId === selectedSegment);
  const selectedSignals = selectedSegment ? report.signals?.[selectedSegment] : null;
  const selectedScore = selectedSegment ? report.scores?.[selectedSegment] : null;

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'code': return <Code className="h-4 w-4" />;
      case 'latex': return <BookOpen className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getConfidenceBadge = (segment: Segment) => {
    const score = report.scores?.[segment.segmentId];
    if (!score) return <Badge variant="outline">Analyzing...</Badge>;
    
    const className = score.confidence === 'High' 
      ? 'bg-ai-high/10 text-ai-high border-ai-high/20'
      : score.confidence === 'Medium'
      ? 'bg-ai-medium/10 text-ai-medium border-ai-medium/20'
      : 'bg-ai-low/10 text-ai-low border-ai-low/20';
    
    // Add warning indicator for short segments
    const isShortSegment = shouldShowReliabilityWarning(segment);
    const badgeContent = (
      <span className="flex items-center gap-1">
        {score.confidence}
        {isShortSegment && score.confidence === 'Low' && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <AlertTriangle className="h-3 w-3 text-warning" />
              </TooltipTrigger>
              <TooltipContent>
                <p>{getLowConfidenceExplanation(segment, score.confidence)}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </span>
    );
    
    return <Badge className={className}>{badgeContent}</Badge>;
  };

  const generateFindings = (segment: Segment): Finding[] => {
    const signals = report.signals?.[segment.segmentId];
    const score = report.scores?.[segment.segmentId];
    
    if (!signals || !score) return [];
    
    const findings: Finding[] = [];
    
    if (signals.gltr && signals.gltr.tailTokenShare < 0.08) {
      findings.push({
        clue: "Low tail-token frequency",
        evidence: `${(signals.gltr.tailTokenShare * 100).toFixed(1)}% tail tokens (< 8% expected)`,
        attributionReason: "Human writers typically use more low-probability tokens than language models",
        suggestedAction: "Review for formulaic or template-like language patterns",
        confidence: score.confidence
      });
    }
    
    if (signals.detectgpt && signals.detectgpt.curvature < -0.3) {
      findings.push({
        clue: "Negative probability curvature",
        evidence: `Curvature: ${signals.detectgpt.curvature.toFixed(3)} (${signals.detectgpt.numPerturbations} perturbations)`,
        attributionReason: "Text shows characteristic smoothness of neural language model outputs",
        suggestedAction: "Check for paraphrasing or significant editing of AI-generated content",
        confidence: score.confidence
      });
    }
    
    if (signals.watermark && signals.watermark.pValue && signals.watermark.pValue < 0.01) {
      findings.push({
        clue: "Statistical watermark detected",
        evidence: `p-value: ${signals.watermark.pValue.toFixed(4)} (< 0.01 threshold)`,
        attributionReason: "Text contains patterns consistent with AI watermarking schemes",
        suggestedAction: "Verify source and generation method with author",
        confidence: "High"
      });
    }
    
    if (signals.refValidityRate !== undefined && signals.refValidityRate < 0.7) {
      findings.push({
        clue: "Low reference validity",
        evidence: `${(signals.refValidityRate * 100).toFixed(0)}% of references resolve correctly`,
        attributionReason: "Academic AI models often generate plausible but non-existent citations",
        suggestedAction: "Manually verify all citations and references",
        confidence: score.confidence
      });
    }
    
    return findings;
  };

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      {/* Segment List */}
      <Card>
        <CardHeader>
          <CardTitle>Segments ({report.segments.length})</CardTitle>
          <CardDescription>Click to view detailed analysis</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 max-h-96 overflow-y-auto">
          {report.segments.map((segment) => (
            <div
              key={segment.segmentId}
              className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                selectedSegment === segment.segmentId 
                  ? 'border-primary bg-primary/5' 
                  : 'border-border hover:border-primary/50'
              }`}
              onClick={() => setSelectedSegment(segment.segmentId)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getTypeIcon(segment.type)}
                  <span className="font-medium text-sm line-clamp-1">
                    {segment.title || 'Untitled'}
                  </span>
                </div>
                {getConfidenceBadge(segment)}
              </div>
              
              {/* Compact signal display */}
              {report.signals?.[segment.segmentId] && (
                <div className="text-xs text-muted-foreground mb-2 flex flex-wrap gap-2">
                  {report.signals[segment.segmentId].gltr && (
                    <span>GLTR tail = {(report.signals[segment.segmentId].gltr!.tailTokenShare * 100).toFixed(1)}%</span>
                  )}
                  {report.signals[segment.segmentId].detectgpt && (
                    <>
                      <span>•</span>
                      <span>Curvature = {report.signals[segment.segmentId].detectgpt!.curvature.toFixed(2)}</span>
                    </>
                  )}
                  {report.scores?.[segment.segmentId] && (
                    <>
                      <span>•</span>
                      <span>Confidence: {report.scores[segment.segmentId].confidence}</span>
                    </>
                  )}
                </div>
              )}
              
              <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                {segment.preview}
              </p>
              
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{segment.lengthChars.toLocaleString()} chars</span>
                <span className="capitalize">{segment.type}</span>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Segment Detail */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            {selectedSegmentData && getTypeIcon(selectedSegmentData.type)}
            <span>{selectedSegmentData?.title || 'Select a segment'}</span>
          </CardTitle>
          {selectedSegmentData && (
            <CardDescription>
              {selectedSegmentData.lengthChars.toLocaleString()} characters • {selectedSegmentData.type}
            </CardDescription>
          )}
        </CardHeader>
        
        {selectedSegmentData && selectedSignals ? (
          <CardContent className="space-y-6">
            {/* GLTR Histogram */}
            {selectedSignals.gltr && (
              <div className="space-y-3">
                <h4 className="font-medium flex items-center space-x-2">
                  <BarChart3 className="h-4 w-4" />
                  <span>GLTR Token Distribution</span>
                </h4>
                <div className="grid grid-cols-4 gap-2">
                  {selectedSignals.gltr.histogram.map((value, index) => {
                    const colors = ['bg-gltr-green', 'bg-gltr-yellow', 'bg-gltr-red', 'bg-gltr-purple'];
                    const labels = ['Green (top-10)', 'Yellow (top-100)', 'Red (top-1000)', 'Purple (tail)'];
                    
                    return (
                      <div key={index} className="space-y-2">
                        <div className="text-center">
                           <div className={`h-20 ${colors[index]} rounded flex items-end justify-center`}>
                             <div 
                               className="w-full bg-foreground/20 rounded-t"
                               style={{ height: `${value * 100}%` }}
                             />
                          </div>
                        </div>
                        <div className="text-center">
                          <p className="text-xs font-medium">{(value * 100).toFixed(1)}%</p>
                          <p className="text-xs text-muted-foreground">{labels[index]}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="text-sm space-y-1">
                  <p><strong>Tail Token Share:</strong> {(selectedSignals.gltr.tailTokenShare * 100).toFixed(1)}%</p>
                  <p><strong>Rank Variance:</strong> {selectedSignals.gltr.rankVariance.toFixed(3)}</p>
                </div>
              </div>
            )}

            {/* DetectGPT Curvature */}
            {selectedSignals.detectgpt && (
              <div className="space-y-3">
                <h4 className="font-medium flex items-center gap-2">
                  DetectGPT Curvature
                  {hasCounterEvidence(selectedSignals.detectgpt.curvature) && (
                    <Badge variant="outline" className="text-success border-success/30">
                      Counter-evidence
                    </Badge>
                  )}
                </h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Curvature</span>
                    <span className="font-mono">{selectedSignals.detectgpt.curvature.toFixed(3)}</span>
                  </div>
                  <Progress 
                    value={((selectedSignals.detectgpt.curvature + 1) / 2) * 100} 
                    className="h-2"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>-1.0 (AI-like)</span>
                    <span>+1.0 (Human-like)</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Based on {selectedSignals.detectgpt.numPerturbations} perturbations
                    {hasCounterEvidence(selectedSignals.detectgpt.curvature) && 
                      ' • Positive curvature suggests human authorship'
                    }
                  </p>
                </div>
              </div>
            )}

            {/* Watermark */}
            {selectedSignals.watermark && selectedSignals.watermark.pValue && (
              <div className="space-y-3">
                <h4 className="font-medium">Watermark Detection</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>p-value</span>
                    <span className="font-mono">{selectedSignals.watermark.pValue.toFixed(4)}</span>
                  </div>
                  <Badge className={selectedSignals.watermark.pValue < 0.01 ? 'bg-ai-high/10 text-ai-high border-ai-high/20' : 'bg-ai-low/10 text-ai-low border-ai-low/20'}>
                    {selectedSignals.watermark.pValue < 0.01 ? 'Watermark Detected' : 'No Watermark'}
                  </Badge>
                </div>
              </div>
            )}

            {/* Score */}
            {selectedScore && (
              <div className="space-y-3">
                <h4 className="font-medium">Composite Score</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Attribution Score</span>
                    <span className="font-mono">{(selectedScore.score * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={selectedScore.score * 100} className="h-2" />
                  <Badge className={
                    selectedScore.confidence === 'High' 
                      ? 'bg-ai-high/10 text-ai-high border-ai-high/20'
                      : selectedScore.confidence === 'Medium'
                      ? 'bg-ai-medium/10 text-ai-medium border-ai-medium/20'
                      : 'bg-ai-low/10 text-ai-low border-ai-low/20'
                  }>
                    {selectedScore.confidence} Confidence
                  </Badge>
                </div>
              </div>
            )}

            {/* Apply Rewrite Button */}
            <div className="pt-4 border-t">
              <Modal>
                <ModalTrigger asChild>
                  <Button className="w-full flex items-center gap-2">
                    <Wand2 className="h-4 w-4" />
                    Apply Rewrite
                  </Button>
                </ModalTrigger>
                <ModalContent>
                  <ModalHeader>
                    <ModalTitle>Superprompt Preview</ModalTitle>
                    <ModalDescription>
                      This is the prompt that will be sent to Claude to rewrite the selected segment.
                    </ModalDescription>
                  </ModalHeader>
                  <div className="space-y-4">
                    <div className="p-4 bg-muted rounded-lg">
                      <p className="text-sm font-mono">
                        You are an expert academic writer. Please rewrite the following text to:
                        <br />• Reduce AI detection signals (vary sentence structure, use more diverse vocabulary)
                        <br />• Maintain original meaning and factual accuracy  
                        <br />• Improve natural language flow and readability
                        <br />• Keep the same approximate length
                        <br /><br />
                        Original text: "{selectedSegmentData?.preview.substring(0, 100)}..."
                      </p>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Coming soon: Real-time rewriting with Claude integration
                    </p>
                  </div>
                </ModalContent>
              </Modal>
            </div>
          </CardContent>
        ) : (
          <CardContent>
            <div className="flex items-center justify-center py-12 text-muted-foreground">
              <div className="text-center space-y-2">
                <AlertTriangle className="h-12 w-12 mx-auto" />
                <p>Select a segment to view detailed analysis</p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Findings Table */}
      {selectedSegmentData && (
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle>Findings</CardTitle>
            <CardDescription>
              Detailed breakdown of attribution signals in {selectedSegmentData.title}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Clue</TableHead>
                  <TableHead>Evidence</TableHead>
                  <TableHead>Attribution Reason</TableHead>
                  <TableHead>Suggested Action</TableHead>
                  <TableHead>Confidence</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {generateFindings(selectedSegmentData).map((finding, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{finding.clue}</TableCell>
                    <TableCell className="font-mono text-sm">{finding.evidence}</TableCell>
                    <TableCell className="text-sm">{finding.attributionReason}</TableCell>
                    <TableCell className="text-sm">{finding.suggestedAction}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className={
                        finding.confidence === 'High' 
                          ? 'border-ai-high/20 text-ai-high'
                          : finding.confidence === 'Medium'
                          ? 'border-ai-medium/20 text-ai-medium'
                          : 'border-ai-low/20 text-ai-low'
                      }>
                        {finding.confidence}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
                {generateFindings(selectedSegmentData).length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                      No significant findings detected in this segment
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default SegmentsTab;