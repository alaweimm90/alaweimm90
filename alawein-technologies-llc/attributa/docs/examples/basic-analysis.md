# Basic Analysis Examples

## Simple Text Analysis

```typescript
import { analyzeText } from '@/lib/nlp/analyzer';

// Basic analysis with default settings
const result = await analyzeText("This is a sample text to analyze.");

console.log('GLTR Score:', result.gltr.tailTokenShare);
console.log('Confidence:', result.gltr.confidence);
```

## PDF Analysis

```typescript
import { extractPDF } from '@/lib/pdf/extractor';
import { analyzeText } from '@/lib/nlp/analyzer';

// Extract text from PDF
const pdfText = await extractPDF(file);

// Analyze extracted text
const result = await analyzeText(pdfText, {
  methods: ['gltr', 'citations'],
  localOnly: true
});

console.log('Citation validity:', result.citations?.validReferences);
```

## Batch Analysis

```typescript
const documents = [
  "First document content...",
  "Second document content...",
  "Third document content..."
];

const results = await Promise.all(
  documents.map(doc => analyzeText(doc))
);

// Calculate average scores
const avgScore = results.reduce((sum, r) => sum + r.gltr.tailTokenShare, 0) / results.length;
console.log('Average GLTR score:', avgScore);
```

## Custom Configuration

```typescript
const customOptions = {
  methods: ['gltr', 'detectgpt'] as const,
  confidence: 0.95,
  localOnly: true,
  segmentLength: 500,
  overlapRatio: 0.1
};

const result = await analyzeText(longDocument, customOptions);

// Access segment-level results
result.segments.forEach((segment, index) => {
  console.log(`Segment ${index + 1}:`, segment.gltr.tailTokenShare);
});
```

## Error Handling

```typescript
try {
  const result = await analyzeText(content);
  
  if (result.errors?.length) {
    console.warn('Analysis warnings:', result.errors);
  }
  
  // Check confidence levels
  if (result.gltr.confidence < 0.7) {
    console.log('Low confidence result - interpret carefully');
  }
  
} catch (error) {
  console.error('Analysis failed:', error.message);
  
  // Handle specific error types
  if (error.code === 'CONTENT_TOO_SHORT') {
    console.log('Try with longer content');
  }
}
```

## Export Results

```typescript
import { exportToPDF, exportToJSON } from '@/lib/export';

const result = await analyzeText(content);

// Export to PDF report
const pdfBlob = await exportToPDF(result, {
  includeConfidenceIntervals: true,
  showMethodDetails: true
});

// Export raw data
const jsonData = exportToJSON(result);
console.log(JSON.stringify(jsonData, null, 2));
```