import { 
  ingest, 
  analyzeText, 
  analyzeWatermark, 
  auditCitations, 
  analyzeCode, 
  computeScore,
  clearSegmentCache 
} from '../src/services/api';
import { IngestRequest } from '../src/types';

describe('Integration Tests', () => {
  beforeEach(() => {
    clearSegmentCache();
  });

  describe('End-to-end text analysis workflow', () => {
    it('should process academic text with citations', async () => {
      const request: IngestRequest = {
        source: 'paste',
        content: `
We present a novel approach to deep learning that improves performance. Our method leverages transformer architectures to achieve state-of-the-art results.

The fundamental challenge in NLP lies in capturing semantic relationships. Our architecture addresses this limitation.

References:
1. Smith, J. (2023). "Advanced Transformers". doi:10.1234/fake.2023.001
2. Brown, M. (2022). "Deep Learning Methods". Journal of AI, 15(3), 123-145.
        `.trim(),
        options: {
          useLocalOnly: true,
          tryWatermark: true,
          useExternalApis: false
        }
      };

      // Step 1: Ingest
      const ingestResult = await ingest(request);
      
      expect(ingestResult.docId).toBeDefined();
      expect(ingestResult.segments.length).toBeGreaterThan(0);
      expect(ingestResult.summary.totalChars).toBeGreaterThan(0);

      // Step 2: Analyze first segment
      const segment = ingestResult.segments[0];
      const textAnalysis = await analyzeText(segment.segmentId);
      
      expect(textAnalysis.gltr).toBeDefined();
      expect(textAnalysis.detectgpt).toBeDefined();
      expect(textAnalysis.gltr.histogram).toHaveLength(4);

      // Step 3: Watermark analysis
      if (segment.lengthChars > 1000) {
        const watermarkResult = await analyzeWatermark(segment.segmentId);
        expect(watermarkResult.pValue).toBeDefined();
      }

      // Step 4: Citation analysis
      const citationResult = await auditCitations(ingestResult.docId);
      expect(citationResult.validated).toBeDefined();
      expect(Array.isArray(citationResult.validated)).toBe(true);
      expect(citationResult.metrics).toBeDefined();

      // Step 5: Compute scores
      const signals = {
        gltr: textAnalysis.gltr,
        detectgpt: textAnalysis.detectgpt
      };
      
      const score = await computeScore(segment.segmentId, signals);
      expect(score.score).toBeGreaterThanOrEqual(0);
      expect(score.score).toBeLessThanOrEqual(1);
      expect(score.confidence).toMatch(/^(Low|Medium|High)$/);
      expect(Array.isArray(score.rationale)).toBe(true);
    });

    it('should process code with security analysis', async () => {
      const request: IngestRequest = {
        source: 'paste',
        content: `
import subprocess
import os

def search_logs(query):
    # CWE-78: Command injection vulnerability
    result = subprocess.run("grep " + query + " /var/log/app.log", shell=True)
    return result.stdout

def upload_file(filename):
    # CWE-22: Path traversal vulnerability  
    with open('/uploads/' + filename, 'r') as f:
        return f.read()

# CWE-798: Hardcoded credentials
API_KEY = "sk-1234567890abcdef"
        `.trim(),
        options: {
          useLocalOnly: true,
          tryWatermark: false,
          useExternalApis: false
        }
      };

      // Step 1: Ingest
      const ingestResult = await ingest(request);
      
      expect(ingestResult.segments.length).toBeGreaterThan(0);
      
      const codeSegment = ingestResult.segments.find(s => s.type === 'code');
      expect(codeSegment).toBeDefined();

      // Step 2: Code analysis
      const codeAnalysis = await analyzeCode(ingestResult.docId, []);
      
      expect(codeAnalysis.findings.length).toBeGreaterThan(0);
      expect(codeAnalysis.metrics.filesScanned).toBeGreaterThan(0);
      
      // Should detect multiple CWE categories
      const cweTypes = new Set(codeAnalysis.findings.map(f => f.cwe));
      expect(cweTypes.size).toBeGreaterThan(1);
      
      // Should detect high severity issues
      const highSeverityFindings = codeAnalysis.findings.filter(f => f.severity === 'HIGH');
      expect(highSeverityFindings.length).toBeGreaterThan(0);

      // Step 3: Text analysis of code
      if (codeSegment) {
        const textAnalysis = await analyzeText(codeSegment.segmentId);
        expect(textAnalysis.gltr).toBeDefined();
        expect(textAnalysis.detectgpt).toBeDefined();

        // Step 4: Score computation with CWE data
        const signals = {
          gltr: textAnalysis.gltr,
          detectgpt: textAnalysis.detectgpt,
          cwePerKloc: codeAnalysis.metrics.findingsPerKloc
        };
        
        const score = await computeScore(codeSegment.segmentId, signals);
        
        expect(score.score).toBeGreaterThan(0);
        if (codeAnalysis.metrics.findingsPerKloc > 5) {
          expect(score.rationale).toContain('CWE findings');
        }
      }
    });

    it('should handle multi-file upload scenario', async () => {
      const files = [
        {
          name: 'paper.tex',
          content: `
\\section{Introduction}
Machine learning has revolutionized artificial intelligence. Furthermore, deep learning approaches have shown remarkable success.

\\section{Methods}  
We propose a novel architecture that combines attention mechanisms with graph neural networks.

\\bibliographystyle{plain}
\\begin{thebibliography}{1}
\\bibitem{smith2023} J. Smith, "AI Advances", Journal of ML, 2023.
\\end{thebibliography}
          `
        },
        {
          name: 'analysis.py',
          content: `
import pandas as pd
import numpy as np

def analyze_data(data_path):
    # Load and process data
    df = pd.read_csv(data_path)
    return df.describe()

# Configuration
DATABASE_PASSWORD = "hardcoded_secret_123"
          `
        }
      ];

      const request: IngestRequest = {
        source: 'file',
        files,
        options: {
          useLocalOnly: true,
          tryWatermark: false,
          useExternalApis: false
        }
      };

      // Step 1: Ingest multiple files
      const ingestResult = await ingest(request);
      
      expect(ingestResult.segments.length).toBeGreaterThan(1);
      expect(ingestResult.summary.types).toBeDefined();
      
      // Should have both latex and code segments
      expect(ingestResult.summary.types.latex).toBeGreaterThan(0);
      expect(ingestResult.summary.types.code).toBeGreaterThan(0);

      // Step 2: Analyze all segments
      const analyses = await Promise.all(
        ingestResult.segments.map(segment => analyzeText(segment.segmentId))
      );
      
      expect(analyses).toHaveLength(ingestResult.segments.length);
      
      // Step 3: Citation and code analysis
      const [citationResult, codeResult] = await Promise.all([
        auditCitations(ingestResult.docId),
        analyzeCode(ingestResult.docId, [])
      ]);
      
      expect(citationResult.validated.length).toBeGreaterThanOrEqual(0);
      expect(codeResult.findings.length).toBeGreaterThan(0); // Should find hardcoded password
      
      // Step 4: Score all segments
      const scores = await Promise.all(
        ingestResult.segments.map((segment, i) => {
          const signals = {
            gltr: analyses[i].gltr,
            detectgpt: analyses[i].detectgpt,
            refValidityRate: segment.type === 'latex' ? 
              citationResult.metrics.refValidityRate : undefined,
            cwePerKloc: segment.type === 'code' ?
              codeResult.metrics.findingsPerKloc : undefined
          };
          
          return computeScore(segment.segmentId, signals);
        })
      );
      
      expect(scores).toHaveLength(ingestResult.segments.length);
      scores.forEach(score => {
        expect(score.score).toBeGreaterThanOrEqual(0);
        expect(score.score).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('Error handling and edge cases', () => {
    it('should handle empty content gracefully', async () => {
      const request: IngestRequest = {
        source: 'paste',
        content: '',
        options: {
          useLocalOnly: true,
          tryWatermark: false,
          useExternalApis: false
        }
      };

      const result = await ingest(request);
      expect(result.segments).toHaveLength(0);
      expect(result.summary.totalChars).toBe(0);
    });

    it('should handle analysis of non-existent segment gracefully', async () => {
      const result = await analyzeText('non-existent-segment-id');
      
      // Should return mock data instead of throwing
      expect(result.segmentId).toBe('non-existent-segment-id');
      expect(result.gltr).toBeDefined();
      expect(result.detectgpt).toBeDefined();
    });

    it('should handle short text segments', async () => {
      const request: IngestRequest = {
        source: 'paste',
        content: 'Short text.',
        options: {
          useLocalOnly: true,
          tryWatermark: false,
          useExternalApis: false
        }
      };

      const ingestResult = await ingest(request);
      
      if (ingestResult.segments.length > 0) {
        const segment = ingestResult.segments[0];
        
        // Watermark should return null for short text
        const watermarkResult = await analyzeWatermark(segment.segmentId);
        expect(watermarkResult.pValue).toBeNull();
        
        // Analysis should still work
        const textResult = await analyzeText(segment.segmentId);
        expect(textResult.gltr).toBeDefined();
        
        // DetectGPT should return 0 curvature for very short text
        if (segment.lengthChars < 300) {
          expect(textResult.detectgpt.curvature).toBe(0);
        }
      }
    });
  });
});