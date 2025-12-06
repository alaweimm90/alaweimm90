export type SegmentType = "prose" | "latex" | "code" | "mixed";

export interface Segment {
  segmentId: string;
  type: SegmentType;
  title?: string;
  lengthChars: number;
  preview: string;
}

export interface GLTRStats {
  tailTokenShare: number;
  rankVariance: number;
  histogram: number[]; // [green, yellow, red, purple]
}

export interface DetectGPTStats {
  curvature: number; // ~[-1, +1]
  numPerturbations: number;
}

export interface WatermarkResult { 
  pValue: number | null; 
}

export interface CitationValidation {
  raw: string;
  resolves: boolean;
  suggestions?: { 
    title: string; 
    doi?: string; 
    confidence: number 
  }[];
}

export interface CodeFinding {
  path: string;
  line: number;
  rule: string;
  cwe: string;
  severity: "LOW" | "MEDIUM" | "HIGH";
  snippet: string;
}

export interface SegmentSignals {
  gltr?: GLTRStats;
  detectgpt?: DetectGPTStats;
  watermark?: WatermarkResult;
  refValidityRate?: number; // 0..1
  cwePerKloc?: number;      // numeric
}

export interface SegmentScore {
  score: number;            // 0..1
  confidence: "Low" | "Medium" | "High";
  rationale: string[];
}

export interface Report {
  docId: string;
  createdAt?: number;
  summary: { 
    totalChars: number; 
    numSegments: number; 
    types: Record<string, number>; 
  };
  segments: Segment[];
  signals: Record<string, SegmentSignals>; // keyed by segmentId
  scores: Record<string, SegmentScore>;
  citations?: CitationValidation[];
  codeFindings?: CodeFinding[];
}

export interface AnalysisOptions {
  useLocalOnly: boolean;
  tryWatermark: boolean;
  useExternalApis: boolean;
}

export interface IngestRequest {
  source: "paste" | "file" | "github";
  content?: string;
  files?: { name: string; content: string }[];
  githubUrl?: string;
  options: AnalysisOptions;
}

export interface IngestResponse {
  docId: string;
  summary: {
    totalChars: number;
    numSegments: number;
    types: Record<string, number>;
  };
  segments: Segment[];
}

export type ConfidenceLevel = "Low" | "Medium" | "High";

export interface Finding {
  clue: string;
  evidence: string;
  whySuspicious: string;
  suggestedAction: string;
  confidence: ConfidenceLevel;
}