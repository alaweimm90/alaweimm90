# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Attributa is a privacy-first attribution intelligence tool for ethical AI development. It analyzes text using local ML models to detect AI-generated content patterns, validate citations, and scan code for security vulnerabilities.

## Development Commands

```bash
# Install dependencies
npm install

# Start development server (port 8080)
npm run dev

# Build for production
npm run build

# Build for development
npm run build:dev

# Run linting
npm run lint

# Run tests
npm run test

# Run E2E tests (Playwright-based)
npm run e2e

# Preview production build
npm run preview
```

## Testing Commands

```bash
# Run all unit tests
npm run test

# Run specific test file
npx jest tests/scoring.test.ts

# Run tests with coverage
npx jest --coverage

# Run E2E tests with custom base URL
BASE_URL=http://localhost:3000 npm run e2e

# E2E test gates (environment variables)
GATES_REQUIRE_MODEL_NAME=true       # Require model name display
GATES_REQUIRE_GLTR_VISIBLE=true     # Require GLTR visualization
GATES_REQUIRE_WORKER_NO_ERRORS=true # No worker errors allowed
GATES_REQUIRE_ALL_BROWSERS=true     # Test all browser engines
GATES_WARM_RUN_MS_MAX=5000         # Max warm run time (ms)
```

## Architecture

### Frontend Stack
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite (development server on port 8080, IPv6 support)
- **UI Components**: shadcn-ui with Radix UI primitives
- **Styling**: Tailwind CSS with custom animations
- **State Management**: Zustand with persistence middleware
- **Routing**: React Router v6
- **Data Fetching**: TanStack Query (React Query)
- **PDF Processing**: pdfjs-dist for rendering, mammoth for DOCX conversion

### Key Directories

- `src/pages/` - Main application routes (Index, Scan, Results, Workspace, Settings, Auth, Documentation, NotFound)
- `src/components/` - Reusable UI components organized by feature
  - `ui/` - Core UI primitives (buttons, dialogs, forms, etc.)
  - `dev/` - Development-specific components (AISecurityIcon, NeuralBackground, PixelAccents)
  - `attribution/` - Attribution analysis components
  - `results/` - Result display components
- `src/lib/` - Core analysis algorithms and utilities
  - `nlp/` - GLTR, DetectGPT, tokenizers, and watermark detection
  - `citations/` - DOI validation and CrossRef integration
  - `code/` - Static security scanning for CWEs
  - `watermark/` - Greenlist-based watermark detection
- `src/services/` - API integration layer (mock and real implementations)
- `src/store/` - Zustand stores for global state management
- `src/hooks/` - Custom React hooks
- `supabase/` - Backend functions and database migrations
- `tests/` - Unit and integration test suite
- `scripts/` - Build and QA automation scripts

### Core Analysis Flow

1. **Document Ingestion**: PDFs/text uploaded and processed client-side via `pdfExtractor.ts`
2. **Segmentation**: Text split into analyzable chunks via `segmentation.ts`
3. **Analysis Pipeline**:
   - GLTR analysis using GPT-2 tokenizer (WebAssembly)
   - DetectGPT probability curvature detection
   - Optional watermark detection
   - Citation validation against CrossRef
   - Code security scanning for common vulnerabilities
4. **Scoring**: Weighted combination of signals with segment-length penalties
5. **Export**: Results available as JSON, PDF reports, or BibTeX citations

### Key Technical Details

- **Local-First Privacy**: All primary analysis runs in-browser using WebAssembly models
- **External APIs**: Optional and require explicit user consent (OpenAI, Anthropic, CrossRef)
- **Model Loading**: Uses @huggingface/transformers for client-side GPT-2
- **PDF Processing**: pdfjs-dist for rendering, mammoth for DOCX conversion
- **Authentication**: Supabase Auth integration (optional)
- **Persistence**: Zustand middleware for local storage of reports and settings
- **Performance**: Lazy loading, code splitting, and performance monitoring components

### Testing Approach

- **Unit Tests**: Jest with ts-jest for core algorithms (scoring, citations, watermark detection)
- **Integration Tests**: API service testing with mocked responses
- **E2E Tests**: Playwright-based tests via `scripts/qa.mjs` for full workflow validation
- **Test Location**: All test files in `tests/` directory
- **Coverage**: Excludes index files, main.tsx, and App.tsx from coverage reports

## Important Configuration

- **TypeScript**: Relaxed strictness (`noImplicitAny: false`, `strictNullChecks: false`, `noUnusedParameters: false`)
- **Path Alias**: `@/` maps to `src/` directory
- **ESLint**: Configured with React hooks and refresh plugins
- **Vite**: Development server on port 8080, supports IPv6, React deduplication
- **Jest**: Configured with ts-jest preset, ESM support, and path alias mapping

## Performance Considerations

- **New UI Components**: Enhanced components for performance monitoring, error boundaries, focus management, and responsive grids
- **Micro-interactions**: Custom animation components for improved UX
- **Lazy Loading**: LazyImage component for optimized image loading
- **Code Splitting**: Automatic via Vite for optimal bundle sizes

## Recent Updates

The codebase recently underwent comprehensive SEO, performance, and accessibility improvements including:
- Enhanced error handling with ErrorBoundary component
- Improved focus management for accessibility
- Performance monitoring utilities
- Responsive grid layouts
- Micro-interaction animations
- Pixel accent decorations for visual interest