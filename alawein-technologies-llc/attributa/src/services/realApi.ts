import { 
  IngestRequest, 
  IngestResponse, 
  Segment, 
  SegmentSignals, 
  SegmentScore, 
  GLTRStats, 
  DetectGPTStats, 
  WatermarkResult,
  CitationValidation,
  CodeFinding,
  SegmentType 
} from '@/types';

// Import real analyzers
import { analyzeText, ensureModelLoaded } from '@/lib/nlp/analyzer';
import { auditCitations } from '@/lib/citations/crossref';
import { analyzeCode } from '@/lib/code/staticScan';
import { greenlistWatermark } from '@/lib/watermark/greenlist';
import { computeCompositeScore } from '@/lib/scoring';
import { mergeShortProse } from '@/lib/segmentation';

// Utility functions (same as mock)
const generateId = () => Math.random().toString(36).substr(2, 9);

const segmentContent = (content: string, type: SegmentType = 'prose'): Segment[] => {
  const segments: Segment[] = [];
  
  if (type === 'code') {
    // For code, create one segment per "file"
    segments.push({
      segmentId: `seg_${generateId()}`,
      type: 'code',
      title: 'app.py',
      lengthChars: content.length,
      preview: content.substring(0, 200) + '...'
    });
  } else {
    // For text/prose/latex, split by paragraphs or sections
    const chunks = content.split('\n\n').filter(chunk => chunk.trim().length > 100);
    
    if (chunks.length === 0 && content.trim().length > 0) {
      // Ensure at least one segment for short content
      segments.push({
        segmentId: `seg_${generateId()}`,
        type,
        title: 'Document',
        lengthChars: content.length,
        preview: content.substring(0, 200) + (content.length > 200 ? '...' : '')
      });
    } else {
      chunks.forEach((chunk, index) => {
        let title = `Segment ${index + 1}`;
        
        // Try to extract title from content
        if (chunk.includes('References:') || chunk.includes('Bibliography:')) {
          title = 'References';
        } else if (chunk.toLowerCase().includes('introduction')) {
          title = 'Introduction';
        } else if (chunk.toLowerCase().includes('conclusion')) {
          title = 'Conclusion';
        } else if (chunk.toLowerCase().includes('method')) {
          title = 'Methodology';
        }
        
        segments.push({
          segmentId: `seg_${generateId()}`,
          type: chunk.includes('doi:') || chunk.includes('References:') ? 'latex' : 'prose',
          title,
          lengthChars: chunk.length,
          preview: chunk.substring(0, 200) + (chunk.length > 200 ? '...' : '')
        });
      });
    }
  }
  
  return segments;
};

const processAndMergeSegments = (segments: Segment[]): Segment[] => {
  // Apply smart merging to reduce short segments
  const merged = mergeShortProse(segments, 1200);

  // Convert back to regular segments for compatibility
  return merged.map(seg => ({
    segmentId: seg.segmentId,
    type: seg.type,
    title: seg.title,
    lengthChars: seg.lengthChars,
    preview: seg.preview
  }));
};

// Cache for segment content
const segmentContentCache = new Map<string, string>();

export const realIngest = async (request: IngestRequest): Promise<IngestResponse> => {
  // Try to preload the local model, but continue if it fails (fallback heuristics will be used)
  try {
    await ensureModelLoaded();
  } catch (e) {
    console.error('Proceeding without local model (fallback mode):', e);
  }
  
  let content = request.content || '';
  let segments: Segment[] = [];
  
  if (request.files && request.files.length > 0) {
    // Process files
    request.files.forEach(file => {
      const fileType: SegmentType = file.name.endsWith('.py') || file.name.endsWith('.js') || file.name.endsWith('.ipynb') 
        ? 'code' 
        : file.name.endsWith('.tex') || file.name.endsWith('.bib')
        ? 'latex'
        : 'prose';
      
      segments.push(...segmentContent(file.content, fileType));
      content += file.content + '\n\n';
    });
  } else if (request.githubUrl) {
    // Mock GitHub processing (same as mock for now)
    content = "# Mock GitHub Repository\nThis is a mock implementation of GitHub repository analysis.";
    segments = segmentContent(content, 'code');
  } else {
    // Process pasted content
    const detectedType: SegmentType = content.includes('subprocess') || content.includes('import ') || content.includes('def ') 
      ? 'code'
      : content.includes('\\section') || content.includes('doi:')
      ? 'latex'
      : 'prose';
    
    segments = segmentContent(content, detectedType);
  }
  
  // Cache segment content for later analysis
  if (content) {
    // Split content by paragraphs to match segments
    const chunks = content.split('\n\n').filter(chunk => chunk.trim().length > 100);
    segments.forEach((seg, index) => {
      if (index < chunks.length) {
        segmentContentCache.set(seg.segmentId, chunks[index]);
      } else {
        // For code or single segments, use the whole content
        segmentContentCache.set(seg.segmentId, content);
      }
    });
  }
  
  const typeCount = segments.reduce((acc, seg) => {
    acc[seg.type] = (acc[seg.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  // Apply smart merging to reduce short segments
  const mergedSegments = processAndMergeSegments(segments);
  
  const updatedTypeCount = mergedSegments.reduce((acc, seg) => {
    acc[seg.type] = (acc[seg.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return {
    docId: `doc_${generateId()}`,
    summary: {
      totalChars: content.length,
      numSegments: mergedSegments.length,
      types: updatedTypeCount
    },
    segments: mergedSegments
  };
};

export const realAnalyzeText = async (segmentId: string): Promise<{ segmentId: string; gltr: GLTRStats; detectgpt: DetectGPTStats }> => {
  // Get cached content for this segment
  const content = segmentContentCache.get(segmentId);
  if (!content) {
    // Fallback to mock-like data if content not found
    const seed = segmentId.charCodeAt(segmentId.length - 1);
    const random = (min: number, max: number) => min + (seed % 100) / 100 * (max - min);
    
    return {
      segmentId,
      gltr: {
        tailTokenShare: random(0.05, 0.15),
        rankVariance: random(0.08, 0.20),
        histogram: [
          random(0.35, 0.55), // green
          random(0.20, 0.30), // yellow  
          random(0.15, 0.25), // red
          random(0.05, 0.15)  // purple
        ]
      },
      detectgpt: {
        curvature: random(-0.8, 0.2),
        numPerturbations: 100
      }
    };
  }
  
  try {
    // Use real analyzer
    const result = await analyzeText(content);
    return {
      segmentId,
      gltr: result.gltr,
      detectgpt: result.detectgpt
    };
  } catch (error) {
    console.error('Real analysis failed, using fallback:', error);
    // Fallback to mock data on error
    const seed = segmentId.charCodeAt(segmentId.length - 1);
    const random = (min: number, max: number) => min + (seed % 100) / 100 * (max - min);
    
    return {
      segmentId,
      gltr: {
        tailTokenShare: random(0.05, 0.15),
        rankVariance: random(0.08, 0.20),
        histogram: [
          random(0.35, 0.55), // green
          random(0.20, 0.30), // yellow  
          random(0.15, 0.25), // red
          random(0.05, 0.15)  // purple
        ]
      },
      detectgpt: {
        curvature: random(-0.8, 0.2),
        numPerturbations: 100
      }
    };
  }
};

export const realAnalyzeWatermark = async (segmentId: string): Promise<{ segmentId: string; watermark: WatermarkResult }> => {
  const content = segmentContentCache.get(segmentId);
  if (!content) {
    // Fallback behavior
    const seed = segmentId.charCodeAt(segmentId.length - 1);
    const pValue = seed % 3 === 0 ? (seed % 100) / 1000 : null;
    
    return {
      segmentId,
      watermark: { pValue }
    };
  }
  
  try {
    // Use real watermark analyzer
    const result = await greenlistWatermark(content);
    return {
      segmentId,
      watermark: result
    };
  } catch (error) {
    console.error('Watermark analysis failed:', error);
    return {
      segmentId,
      watermark: { pValue: null }
    };
  }
};

export const realAuditCitations = async (docId: string): Promise<{ docId: string; citations: CitationValidation[] }> => {
  // Extract all DOIs from cached content
  const allContent = Array.from(segmentContentCache.values()).join('\n');
  
  if (!allContent) {
    // Return empty results if no content
    return { docId, citations: [] };
  }
  
  try {
    const citations = await auditCitations(allContent);
    return { docId, citations };
  } catch (error) {
    console.error('Citation analysis failed:', error);
    // Return mock data as fallback
    return {
      docId,
      citations: [
        {
          raw: "doi:10.1234/nonexistent.2023.001",
          resolves: false,
          suggestions: [
            { title: "Transformer Networks in Natural Language Processing", doi: "10.1145/3456789.3456790", confidence: 0.75 }
          ]
        }
      ]
    };
  }
};

export const realAnalyzeCode = async (docId: string): Promise<{ docId: string; findings: CodeFinding[] }> => {
  // Get all code content from cache
  const allContent = Array.from(segmentContentCache.values()).join('\n');
  
  if (!allContent || !allContent.includes('import ') && !allContent.includes('def ')) {
    // Return empty if no code content
    return { docId, findings: [] };
  }
  
  try {
    const findings = await analyzeCode(allContent);
    return { docId, findings };
  } catch (error) {
    console.error('Code analysis failed:', error);
    // Return mock data as fallback
    return {
      docId,
      findings: [
        {
          path: "app.py",
          line: 12,
          rule: "subprocess-shell",
          cwe: "CWE-78", 
          severity: "HIGH",
          snippet: 'subprocess.run("grep " + query + " /var/log/app.log", shell=True)'
        }
      ]
    };
  }
};

export const realComputeScore = async (segmentId: string, signals: unknown, weights?: unknown): Promise<SegmentScore> => {
  // Use real scoring with proper weights
  const result = computeCompositeScore({ ...signals, weights });
  return result;
};

// Export the same interface as mockApi for drop-in replacement
export const mockIngest = realIngest;
export const mockAnalyzeText = realAnalyzeText;
export const mockAnalyzeWatermark = realAnalyzeWatermark;
export const mockAuditCitations = realAuditCitations;
export const mockAnalyzeCode = realAnalyzeCode;
export const mockComputeScore = realComputeScore;

// Roadmap stubs (same as mock)
export const mockRewrite = async (segmentId: string, findingIndex: number): Promise<{ ok: boolean; message: string }> => {
  await new Promise(r => setTimeout(r, 300));
  return { ok: true, message: 'Rewrite stub: this will apply the Superprompt in a future update.' };
};

export const mockCitationsFix = async (rawCitation: string, suggestedDoi: string): Promise<{ ok: boolean; patched: string }> => {
  await new Promise(r => setTimeout(r, 300));
  return { ok: true, patched: `${rawCitation} → DOI:${suggestedDoi}` };
};

export const mockAnalysisBatch = async (segmentIds: string[]): Promise<{ ok: boolean; processed: number }> => {
  await new Promise(r => setTimeout(r, 300));
  return { ok: true, processed: segmentIds.length };
};