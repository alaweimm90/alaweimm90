## v2.1.0 — 2025-01-12

### Added
- Comprehensive accessibility features with A11yToolbar
- Advanced performance optimizations with code splitting
- Service worker for offline caching
- Live region announcements for screen readers
- High contrast mode and reduced motion support
- Enhanced focus management and keyboard navigation
- Web vitals monitoring and performance tracking
- Comprehensive documentation and API reference
- Examples for common use cases

### Changed
- Redesigned workflow visualization with cleaner layout
- Simplified header navigation removing clutter
- Improved footer organization with proper social links
- Updated contact information with correct LinkedIn profile
- Changed "Research-Grade Methods" to "Statistical Analysis"
- Removed generic marketing copy for direct language

### Fixed
- GitHub links now point to profile instead of private repo
- "Learn Methods" button works with smooth scrolling
- Fixed typo "Try in Browser" → "Analyze"
- Corrected web-vitals API imports (FID → INP)
- TypeScript errors with prefer-const and explicit-any
- Bundle size optimization with manual chunking

### Removed
- Generic "Transparency First" principles section
- Duplicate GitHub button and badge clutter in header
- AI-generated buzzwords and hedging language

## v0.1.1 — 2025-08-09

### Polish Pack Features
- **Export**: JSON/CSV download from Overview tab with complete report data
- **Settings**: User-configurable local model (Hugging Face ID) with fallback handling
- **Documentation**: Added comprehensive README caveats section explaining limitations and privacy
- **CI**: Enhanced STATUS workflow with performance aggregates and browser success rates
- **Resilience**: Improved demo error handling for offline/model load failures
- **UX**: Added accessibility labels and responsive GLTR histogram styling
- **Security**: Added SECURITY.md and CONTRIBUTING.md documentation

## v0.1.0 — 2025-08-09

### Features
- PR #1: Workerized scorer, GLTR normalization, "Run all demos", model loader progress/cancel (https://github.com/alawein/Attributa/pull/1)
  - Offloaded GLTR + DetectGPT computation to web workers for responsive UI
  - Implemented step-wise GPT-2 token analysis with computeRanks and avgLogProb
  - Fixed GLTR histogram normalization to sum to ~100%
  - Added one-click "Run all demos" for Human/AI prose, LaTeX, and Python code
  - Enhanced model loader with progress tracking and cancel functionality

### CI/CD
- PR #6: Add Playwright E2E workflow + npm e2e wrapper (https://github.com/alawein/Attributa/pull/6)
  - Cross-browser E2E testing (Chromium, Firefox, WebKit)
  - Label-tunable quality gates (ci-relaxed/ci-strict)
  - Automated STATUS.md updates and draft releases
  - PR comment integration with test results
