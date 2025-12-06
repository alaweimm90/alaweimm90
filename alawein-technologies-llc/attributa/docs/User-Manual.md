# Attributa.dev User Manual

Table of Contents
- 1. Introduction
- 2. Key Concepts
- 3. Getting Started
  - 3.1 Quick Start (Lovable)
  - 3.2 Local Development
- 4. Using Attributa
  - 4.1 Scan
  - 4.2 Results
  - 4.3 Exports
  - 4.4 Settings
  - 4.5 Workspace
- 5. How-To Guides
- 6. Troubleshooting
- 7. Privacy & Security
- 8. FAQ
- 9. Glossary
- 10. Appendix

1. Introduction
Attributa.dev is a privacy-first attribution intelligence web app that analyzes content to surface evidence about possible AI-generation and source attribution. It runs primarily in your browser using local models, with optional external APIs (OpenAI/Anthropic) and optional Supabase integration for persistence and edge functions.

Supported inputs: plain text, code, PDFs/LaTeX. Outputs: per-segment signals, overall scores, citation validity, code security findings, and exportable reports.

[Placeholder: Screenshot – Home / Scan flow]

2. Key Concepts
- GLTR: Token rank distribution histogram indicating how “predictable” the text is to a language model.
- DetectGPT Curvature: Curvature score (often negative for AI text) derived from perturbation-based likelihood curvature; simplified variant in this app.
- Watermark p-value: Probability that text was generated with a watermarking scheme (greenlist), reported as a p-value.
- Citation Validation: DOI extraction, Crossref lookup, BibTeX parsing; measures citation consistency and validity.
- Code Security: Static scan against built-in rules with CWE mappings; no dynamic code execution.
- Composite Scoring: Weighted combination of signals with confidence bands and rationale.

3. Getting Started
3.1 Quick Start (Lovable)
- Open the project in Lovable and press Run to start the dev server.
- Navigate to /scan to upload or paste content.
- Results appear automatically on /results.

3.2 Local Development
Prerequisites: Node.js 18+, npm, modern browser, Git.
- git clone <your-fork-url>
- cd attributa
- npm install
- npm run dev
Optional: .env.local for external APIs
- VITE_OPENAI_API_KEY=...
- VITE_ANTHROPIC_API_KEY=...
Optional Supabase: follow supabase/config.toml and functions/ setup.
Run tests: npm test

4. Using Attributa
4.1 Scan (/scan)
- Upload text/code/PDF (processed client-side where possible).
- Toggle which analyzers to run (Settings > Analyzers).
- [Screenshot: Scan page with inputs]

4.2 Results (/results)
Tabs and panels:
- Overview: composite score, confidence band, rationale, weights explanation.
- Segments: per-segment GLTR, curvature, watermark p-values.
- Citations: DOI resolution, Crossref matches, validation metrics.
- Code Security: CWE findings, counts, KLOC metrics.
- Exports: generate PDF/DOCX reports.
[Screenshots: Overview, Segments, Citations, Code Security]

4.3 Exports
- Choose format (PDF/DOCX) and options (include rationale, weights, charts).
- [Screenshot: Export dialog]

4.4 Settings (/settings)
- Local vs external models; API keys; rate limits.
- Scoring weights; feature toggles; privacy controls.
- [Screenshot: Settings]

4.5 Workspace (/workspace)
- Manage projects, artifacts, segments.
- Re-run analyses; compare versions.

5. How-To Guides
- End-to-end analysis: Upload → Analyze → Interpret → Export.
- Validate citations in LaTeX: paste .tex or PDF; review DOI matches and suggestions.
- Run code security scan: paste code or upload files; inspect CWE mappings.
- Adjust scoring weights: tune per-signal weights; observe normalized percentages.

6. Troubleshooting
- Model loading failures: Ensure WebGPU available; reload; fall back to CPU; reduce input size.
- CORS on citations: Try again later; rate limits may apply.
- Large PDFs: Use smaller chunks; close other heavy tabs.
- Worker timeouts: Split very large inputs; check console for hints.
- Supabase connectivity: Verify project URL and anon key; check RLS policies.

7. Privacy & Security
- Local-first: default analysis runs in-browser; no telemetry.
- External APIs: opt-in only; disclose implications.
- Limits: size/timeouts to prevent browser lockups.
- Storage: optional Supabase with RLS; never stores secrets in client code.

8. FAQ
- Is this a detector? It provides probabilistic signals; interpret in context.
- Does it work for non-English? Limited; optimized for English.
- How accurate is watermark detection? Simplified; indicative, not definitive.

9. Glossary
- GLTR, curvature, p-value, DOI, Crossref, BibTeX, CWE, RLS, KLOC, Rationale.

10. Appendix
- Keyboard: standard browser shortcuts; app-specific shortcuts not required.
- File size limits: Prefer <10MB total code; text segments ~1.5k–3k chars per run.
- Links: README.md, README_ANALYZERS.md, CONTRIBUTING.md.
