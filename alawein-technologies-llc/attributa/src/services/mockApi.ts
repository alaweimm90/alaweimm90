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
import { computeCompositeScore } from '@/lib/scoring';
import { mergeShortProse } from '@/lib/segmentation';


// Mock data and utilities
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

export const mockIngest = async (request: IngestRequest): Promise<IngestResponse> => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
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
    // Mock GitHub processing
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

export const mockAnalyzeText = async (segmentId: string): Promise<{ segmentId: string; gltr: GLTRStats; detectgpt: DetectGPTStats }> => {
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // Generate realistic but varied mock data
  const seed = segmentId.charCodeAt(segmentId.length - 1);
  const random = (min: number, max: number) => min + (seed % 100) / 100 * (max - min);
  
  return {
    segmentId,
    gltr: {
      tailTokenShare: random(0.05, 0.15), // Lower = more suspicious
      rankVariance: random(0.08, 0.20),
      histogram: [
        random(0.35, 0.55), // green
        random(0.20, 0.30), // yellow  
        random(0.15, 0.25), // red
        random(0.05, 0.15)  // purple
      ]
    },
    detectgpt: {
      curvature: random(-0.8, 0.2), // More negative = more suspicious
      numPerturbations: 100
    }
  };
};

export const mockAnalyzeWatermark = async (segmentId: string): Promise<{ segmentId: string; watermark: WatermarkResult }> => {
  await new Promise(resolve => setTimeout(resolve, 300));
  
  const seed = segmentId.charCodeAt(segmentId.length - 1);
  const pValue = seed % 3 === 0 ? (seed % 100) / 1000 : null; // Sometimes no watermark detected
  
  return {
    segmentId,
    watermark: { pValue }
  };
};

export const mockAuditCitations = async (docId: string): Promise<{ docId: string; citations: CitationValidation[] }> => {
  await new Promise(resolve => setTimeout(resolve, 700));
  
  const mockCitations: CitationValidation[] = [
    {
      raw: "doi:10.1234/nonexistent.2023.001",
      resolves: false,
      suggestions: [
        { title: "Transformer Networks in Natural Language Processing", doi: "10.1145/3456789.3456790", confidence: 0.75 },
        { title: "Deep Learning Architectures for Text Analysis", confidence: 0.62 }
      ]
    },
    {
      raw: "doi:10.5555/fake.doi.2022.456", 
      resolves: false,
      suggestions: [
        { title: "Graph Neural Networks for Language Modeling", doi: "10.1038/s41586-022-04567-8", confidence: 0.68 }
      ]
    },
    {
      raw: "doi:10.1038/fictional.2023.789",
      resolves: false,
      suggestions: [
        { title: "Attention Mechanisms in Machine Learning", doi: "10.1126/science.abc1234", confidence: 0.71 }
      ]
    }
  ];
  
  return {
    docId,
    citations: mockCitations
  };
};

export const mockAnalyzeCode = async (docId: string): Promise<{ docId: string; findings: CodeFinding[] }> => {
  await new Promise(resolve => setTimeout(resolve, 600));
  
  const mockFindings: CodeFinding[] = [
    {
      path: "app.py",
      line: 12,
      rule: "subprocess-shell",
      cwe: "CWE-78", 
      severity: "HIGH",
      snippet: 'subprocess.run("grep " + query + " /var/log/app.log", shell=True)'
    },
    {
      path: "app.py",
      line: 18,
      rule: "path-traversal",
      cwe: "CWE-22",
      severity: "HIGH", 
      snippet: "with open('/uploads/' + filename, 'r') as f:"
    },
    {
      path: "app.py",
      line: 22,
      rule: "debug-mode",
      cwe: "CWE-489",
      severity: "MEDIUM",
      snippet: "app.run(debug=True, host='0.0.0.0')"
    }
  ];
  
  return {
    docId,
    findings: mockFindings
  };
};

export const mockComputeScore = async (segmentId: string, signals: unknown, weights?: unknown): Promise<SegmentScore> => {
  await new Promise(resolve => setTimeout(resolve, 200));
  const result = computeCompositeScore({ ...signals, weights });
  return result;
};

// Roadmap stubs
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
