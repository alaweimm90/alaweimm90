/**
 * AI Development Tools Suite
 *
 * A comprehensive suite of AI-powered development tools that leverage
 * the ATLAS multiagent orchestration platform.
 */

export { AITools } from './core/ai-tools.js';
export { ATLASIntegration } from './core/atlas-integration.js';

// Tool exports
export { TestGenerator } from './tools/test-generator/index.js';
export { DocGenerator } from './tools/doc-generator/index.js';
export { ReviewAssistant } from './tools/review-assistant/index.js';
export { ArchitectureAnalyzer } from './tools/architecture-analyzer/index.js';
export { PerformanceProfiler } from './tools/performance-profiler/index.js';
export { SecurityScanner } from './tools/security-scanner/index.js';

// Types
export type {
  ToolConfig,
  ToolResult,
  AIToolsOptions,
  ATLASConfig
} from './core/types.js';

export type {
  TestGenerationOptions,
  TestSuite
} from './tools/test-generator/types.js';

export type {
  DocGenerationOptions,
  Documentation
} from './tools/doc-generator/types.js';

export type {
  ReviewOptions,
  ReviewResult
} from './tools/review-assistant/types.js';

export type {
  ArchitectureOptions,
  ArchitectureReport
} from './tools/architecture-analyzer/types.js';

export type {
  PerformanceOptions,
  PerformanceReport
} from './tools/performance-profiler/types.js';

export type {
  SecurityOptions,
  SecurityReport
} from './tools/security-scanner/types.js';

// CLI
export { createCLI } from './cli/index.js';

// Default export for convenience
import { AITools } from './core/ai-tools.js';
export default AITools;
