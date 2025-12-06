# Attributa Documentation

## Overview

Attributa is an open-source AI attribution analysis tool that helps identify AI-generated content patterns in text documents. It uses statistical methods to provide probability estimates with explicit uncertainty bounds.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/alawein/Attributa.git
cd Attributa

# Install dependencies
npm install

# Start development server
npm run dev
```

### Basic Usage

1. **Upload Content**: Paste text or upload a PDF document
2. **Select Methods**: Choose which analysis techniques to apply
3. **Run Analysis**: Process the content locally in your browser
4. **Review Results**: Examine probability scores and confidence intervals

## Analysis Methods

### 1. Statistical Analysis (GLTR)
- **What it does**: Analyzes token likelihood patterns using GPT-2
- **Strengths**: Fast, well-researched, good baseline
- **Limitations**: Only trained on GPT-2, may not detect newer models
- **Confidence**: Medium to High

### 2. Perturbation Analysis (DetectGPT)
- **What it does**: Examines probability curvature via text modifications
- **Strengths**: Model-agnostic, theoretically grounded
- **Limitations**: Slower, requires external API calls
- **Confidence**: Medium

### 3. Citation Validation
- **What it does**: Cross-references citations against academic databases
- **Strengths**: Objective, verifiable results
- **Limitations**: Only works for academic content
- **Confidence**: High

### 4. Watermark Detection
- **What it does**: Looks for cryptographic markers in text
- **Strengths**: Extremely accurate when present
- **Limitations**: Rarely used, requires model cooperation
- **Confidence**: High (when detected)

## API Reference

### Core Functions

#### `analyzeText(content: string, options?: AnalysisOptions)`

Analyzes text content using selected methods.

```typescript
import { analyzeText } from '@/lib/nlp/analyzer';

const result = await analyzeText("Sample text...", {
  methods: ['gltr', 'detectgpt'],
  confidence: 0.95,
  localOnly: true
});

console.log(result.gltr.tailTokenShare); // 0.123
console.log(result.detectgpt.curvature); // -0.045
```

#### `extractPDF(file: File)`

Extracts text content from PDF files.

```typescript
import { extractPDF } from '@/lib/pdf/extractor';

const text = await extractPDF(pdfFile);
console.log(text); // "Extracted PDF content..."
```

### Configuration

#### Analysis Options

```typescript
interface AnalysisOptions {
  methods: ('gltr' | 'detectgpt' | 'citations' | 'watermark')[];
  confidence: number; // 0.0 to 1.0
  localOnly: boolean;
  segmentLength: number; // characters
  overlapRatio: number; // 0.0 to 1.0
}
```

## Development

### Project Structure

```
src/
├── components/        # React components
│   ├── ui/           # Reusable UI components
│   ├── attribution/  # Analysis-specific components
│   └── dev/          # Development utilities
├── lib/              # Core analysis libraries
│   ├── nlp/          # Natural language processing
│   ├── citations/    # Citation validation
│   └── watermark/    # Watermark detection
├── pages/            # Application routes
├── hooks/            # Custom React hooks
├── store/            # Global state management
└── styles/           # CSS and styling
```

### Adding New Analysis Methods

1. Create method implementation in `src/lib/nlp/`
2. Add to analyzer interface
3. Update UI components
4. Add tests
5. Update documentation

Example:

```typescript
// src/lib/nlp/newmethod.ts
export async function analyzeNewMethod(text: string): Promise<MethodResult> {
  return {
    score: 0.75,
    confidence: 0.85,
    metadata: { /* method-specific data */ }
  };
}
```

### Testing

```bash
# Run unit tests
npm test

# Run E2E tests
npm run e2e

# Run accessibility tests
npm run test:a11y

# Run performance tests
npm run lighthouse
```

## Deployment

### Production Build

```bash
npm run build
npm run preview
```

### Environment Variables

```env
VITE_OPENAI_API_KEY=your_key_here
VITE_ANTHROPIC_API_KEY=your_key_here
VITE_SUPABASE_URL=your_url_here
VITE_SUPABASE_KEY=your_key_here
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 4173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0"]
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Support

- GitHub Issues: [Report bugs](https://github.com/alawein/Attributa/issues)
- Documentation: [docs/](.)
- Examples: [examples/](./examples/)

## Research & Citations

If you use Attributa in academic work, please cite:

```bibtex
@software{attributa2024,
  title = {Attributa: Open-Source AI Attribution Analysis},
  author = {Alawein, Meshal},
  year = {2024},
  url = {https://github.com/alawein/Attributa}
}
```

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for version history.