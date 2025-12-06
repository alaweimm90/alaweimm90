# Analysis Methods API

## GLTR (Giant Language model Test Room)

### Overview
GLTR analyzes text by examining token likelihood predictions from GPT-2. It categorizes tokens into probability bins to identify patterns typical of AI generation.

### Usage
```typescript
import { analyzeGLTR } from '@/lib/nlp/gltr';

const result = await analyzeGLTR(text);
```

### Response Format
```typescript
interface GLTRResult {
  tailTokenShare: number;      // 0.0 to 1.0
  confidenceInterval: [number, number];
  confidence: number;          // 0.0 to 1.0
  bins: {
    top10: number;            // Tokens in top 10 predictions
    top100: number;           // Tokens in top 100 predictions  
    top1000: number;          // Tokens in top 1000 predictions
    tail: number;             // Tokens beyond top 1000
  };
  metadata: {
    totalTokens: number;
    processingTime: number;
    modelVersion: string;
  };
}
```

### Interpretation
- **tailTokenShare < 0.1**: Likely human-written
- **tailTokenShare 0.1-0.3**: Mixed or edited content
- **tailTokenShare > 0.3**: Likely AI-generated

## DetectGPT

### Overview
DetectGPT uses probability curvature analysis by slightly perturbing text and measuring how the probability changes.

### Usage
```typescript
import { analyzeDetectGPT } from '@/lib/nlp/detectgpt';

const result = await analyzeDetectGPT(text, {
  perturbations: 10,
  apiKey: process.env.OPENAI_API_KEY
});
```

### Response Format
```typescript
interface DetectGPTResult {
  curvature: number;          // Negative values suggest AI generation
  confidenceInterval: [number, number];
  confidence: number;
  perturbations: {
    original: number;         // Original text probability
    perturbed: number[];      // Perturbed text probabilities
    mean: number;            // Mean perturbed probability
    variance: number;        // Variance in probabilities
  };
  metadata: {
    perturbationCount: number;
    model: string;
    processingTime: number;
  };
}
```

### Interpretation
- **curvature < -0.1**: Likely AI-generated
- **curvature -0.1 to 0.1**: Uncertain
- **curvature > 0.1**: Likely human-written

## Citation Validation

### Overview
Validates academic citations by cross-referencing with DOI databases and checking format consistency.

### Usage
```typescript
import { validateCitations } from '@/lib/citations/validator';

const result = await validateCitations(text);
```

### Response Format
```typescript
interface CitationResult {
  validReferences: number;
  totalReferences: number;
  validityRatio: number;      // 0.0 to 1.0
  confidence: number;
  issues: {
    invalidDOIs: string[];
    missingDOIs: string[];
    formatErrors: string[];
    duplicates: string[];
  };
  citations: Array<{
    text: string;
    doi?: string;
    valid: boolean;
    type: 'journal' | 'conference' | 'book' | 'web';
    confidence: number;
  }>;
}
```

### Interpretation
- **validityRatio > 0.9**: High-quality references
- **validityRatio 0.7-0.9**: Moderate quality
- **validityRatio < 0.7**: Poor reference quality

## Watermark Detection

### Overview
Detects cryptographic watermarks embedded in AI-generated text using statistical analysis.

### Usage
```typescript
import { detectWatermark } from '@/lib/watermark/detector';

const result = await detectWatermark(text);
```

### Response Format
```typescript
interface WatermarkResult {
  detected: boolean;
  strength: number;          // 0.0 to 1.0
  confidence: number;
  pattern: {
    type: 'greenlist' | 'redlist' | 'statistical';
    markers: number;
    distribution: number[];
  };
  metadata: {
    method: string;
    vocabulary: number;
    gamma: number;           // Detection threshold
  };
}
```

### Interpretation
- **detected = true, strength > 0.8**: Strong watermark present
- **detected = true, strength < 0.5**: Weak or degraded watermark
- **detected = false**: No watermark found

## Combined Analysis

### Usage
```typescript
import { analyzeText } from '@/lib/nlp/analyzer';

const result = await analyzeText(text, {
  methods: ['gltr', 'detectgpt', 'citations', 'watermark']
});
```

### Response Format
```typescript
interface CombinedResult {
  overallScore: number;      // Weighted combination
  confidence: number;        // Overall confidence
  agreement: number;         // Method agreement (0.0 to 1.0)
  
  gltr?: GLTRResult;
  detectgpt?: DetectGPTResult;
  citations?: CitationResult;
  watermark?: WatermarkResult;
  
  segments: Array<{
    text: string;
    startIndex: number;
    endIndex: number;
    results: Partial<CombinedResult>;
  }>;
  
  metadata: {
    totalTime: number;
    wordsAnalyzed: number;
    methodsUsed: string[];
  };
}
```

## Error Handling

### Common Error Codes
- `CONTENT_TOO_SHORT`: Text under minimum length (50 characters)
- `CONTENT_TOO_LONG`: Text exceeds maximum length (50,000 characters)  
- `API_KEY_MISSING`: Required API key not provided
- `RATE_LIMIT_EXCEEDED`: API rate limit reached
- `NETWORK_ERROR`: Connection issues
- `PARSING_ERROR`: Unable to process content format

### Example Error Handling
```typescript
try {
  const result = await analyzeText(content);
} catch (error) {
  switch (error.code) {
    case 'CONTENT_TOO_SHORT':
      console.log('Minimum 50 characters required');
      break;
    case 'API_KEY_MISSING':
      console.log('API key required for this analysis');
      break;
    default:
      console.error('Analysis failed:', error.message);
  }
}
```