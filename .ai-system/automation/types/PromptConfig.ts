/**
 * Configuration for AI prompt templates with metadata
 */
export interface PromptConfig {
  name: string;
  version: string;
  description?: string;
  tags?: string[];
  parameters?: Record<string, unknown>;
  template: string;
  created?: string;
  modified?: string;
  author?: string;
  category?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Collection of prompt configurations
 */
export interface PromptCollection {
  prompts: PromptConfig[];
  metadata?: {
    version?: string;
    description?: string;
    lastUpdated?: string;
  };
}
