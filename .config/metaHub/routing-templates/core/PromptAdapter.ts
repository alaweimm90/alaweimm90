/**
 * TRAE Prompt Adapter Template
 * Model-Specific Prompt Optimization and Token Efficiency
 *
 * This template provides a complete prompt adaptation system that optimizes
 * prompts for different model families, manages token usage, and ensures
 * model-specific formatting for optimal performance.
 *
 * Features:
 * - Provider-specific templates (OpenAI, Anthropic, Gemini, Local)
 * - Token usage optimization (70% compression target)
 * - Model-specific prompt formatting
 * - Validation and constraint enforcement
 * - Quality enhancement algorithms
 * - Framework and language agnostic design
 */

// ============================================================================
// TYPE DEFINITIONS (Language Agnostic)
// ============================================================================

export interface ModelCapability {
  name: string;
  provider: string;
  model: string;
  tier: RoutingTier;
  region: GeographicRegion;
  maxTokens: number;
  costPerToken: number;
  qualityScore: number;
  latency: number;
  reliability: number;
  specializations: string[];
  limitations: string[];
  compliance: ComplianceFramework[];
}

export enum RoutingTier {
  TIER_1 = 'tier_1',
  TIER_2 = 'tier_2',
  TIER_3 = 'tier_3',
}

export enum GeographicRegion {
  NORTH_AMERICA = 'north_america',
  EUROPE = 'europe',
  ASIA_PACIFIC = 'asia_pacific',
  SOUTH_AMERICA = 'south_america',
  AFRICA = 'africa',
  GLOBAL = 'global',
}

export enum ComplianceFramework {
  HIPAA = 'hipaa',
  GDPR = 'gdpr',
  SOC2 = 'soc2',
  PCI_DSS = 'pci_dss',
}

export interface TaskAnalysis {
  complexity: TaskComplexity;
  estimatedTokens: number;
  requiredCapabilities: string[];
  timeSensitivity: TimeSensitivity;
  costSensitivity: CostSensitivity;
  geographicPreference?: GeographicRegion;
  contextLength: number;
  domain: string[];
  complianceRequirements?: ComplianceFramework[];
}

export enum TaskComplexity {
  SIMPLE = 'simple',
  MODERATE = 'moderate',
  COMPLEX = 'complex',
  CRITICAL = 'critical',
}

export enum TimeSensitivity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export enum CostSensitivity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
}

export interface RoutingContext {
  userId?: string;
  sessionId: string;
  requestId: string;
  timestamp: Date;
  clientRegion?: GeographicRegion;
  priority: RequestPriority;
  tags: string[];
  complianceContext?: ComplianceContext;
}

export enum RequestPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface ComplianceContext {
  dataClassification: DataClassification;
  retentionPolicy: RetentionPolicy;
  auditRequirements: boolean;
  geographicRestrictions: GeographicRegion[];
}

export enum DataClassification {
  PUBLIC = 'public',
  INTERNAL = 'internal',
  CONFIDENTIAL = 'confidential',
  RESTRICTED = 'restricted',
}

export enum RetentionPolicy {
  EPHEMERAL = 'ephemeral',
  SHORT_TERM = 'short_term',
  LONG_TERM = 'long_term',
  PERMANENT = 'permanent',
}

// ============================================================================
// PROMPT ADAPTER IMPLEMENTATION
// ============================================================================

export class PromptAdapter {
  private templates: Map<string, PromptTemplate> = new Map();
  private modelOptimizations: Map<string, ModelOptimization> = new Map();

  // Token optimization settings
  private readonly MAX_PROMPT_LENGTH = 0.8; // Use 80% of context window
  private readonly TOKEN_BUFFER = 1000; // Reserve tokens for response
  private readonly COMPRESSION_RATIO = 0.7; // Target compression ratio

  constructor() {
    this.initializeTemplates();
    this.initializeModelOptimizations();
  }

  /**
   * Adapt prompt for specific model
   */
  adaptPrompt(
    originalPrompt: string,
    model: ModelCapability,
    taskAnalysis: TaskAnalysis,
    context?: RoutingContext
  ): AdaptedPrompt {
    const template = this.getTemplateForModel(model);
    const optimizations = this.getOptimizationsForModel(model);

    let adaptedPrompt = originalPrompt;

    // Apply model-specific formatting
    adaptedPrompt = this.applyModelFormatting(adaptedPrompt, model, template);

    // Optimize for token efficiency
    adaptedPrompt = this.optimizeTokenUsage(adaptedPrompt, model, taskAnalysis);

    // Apply model-specific constraints
    adaptedPrompt = this.applyConstraints(adaptedPrompt, template);

    // Add context-aware enhancements
    adaptedPrompt = this.addContextEnhancements(adaptedPrompt, context, model);

    // Apply optimizations
    adaptedPrompt = this.applyOptimizations(adaptedPrompt, optimizations, taskAnalysis);

    const metrics = this.calculateAdaptationMetrics(originalPrompt, adaptedPrompt, model);

    return {
      originalPrompt,
      adaptedPrompt,
      model,
      ...(template?.template && { template: template.template }),
      optimizations: optimizations.map(opt => opt.type),
      metrics,
    };
  }

  /**
   * Get prompt template for model family
   */
  getTemplateForModel(model: ModelCapability): PromptTemplate | null {
    // Try exact model match first
    const exactKey = `${model.provider}:${model.model}`;
    if (this.templates.has(exactKey)) {
      return this.templates.get(exactKey)!;
    }

    // Try provider match
    if (this.templates.has(model.provider)) {
      return this.templates.get(model.provider)!;
    }

    // Try model family match (e.g., 'gpt-4', 'claude-3')
    const modelParts = model.model.split('-');
    const familyKey = modelParts.length > 0 ? modelParts[0] : '';
    if (familyKey && this.templates.has(familyKey)) {
      return this.templates.get(familyKey)!;
    }

    return null;
  }

  /**
   * Estimate token count for adapted prompt
   */
  estimateTokenCount(prompt: string, model: ModelCapability): number {
    // Rough estimation: 1 token ≈ 4 characters for most models
    // This would be replaced with actual tokenizer in production
    const baseTokens = Math.ceil(prompt.length / 4);

    // Apply model-specific multipliers
    const multiplier = this.getTokenMultiplier(model);
    return Math.ceil(baseTokens * multiplier);
  }

  /**
   * Validate prompt against model constraints
   */
  validatePrompt(prompt: string, model: ModelCapability): PromptValidation {
    const issues: PromptIssue[] = [];
    const tokenCount = this.estimateTokenCount(prompt, model);

    // Check token limits
    if (tokenCount > model.maxTokens * this.MAX_PROMPT_LENGTH) {
      issues.push({
        type: 'token_limit',
        severity: 'error',
        message: `Prompt too long: ${tokenCount} tokens (max: ${Math.floor(
          model.maxTokens * this.MAX_PROMPT_LENGTH
        )})`,
        suggestion: 'Compress or truncate prompt',
      });
    }

    // Check for model-specific limitations
    const limitations = this.checkModelLimitations(prompt, model);
    issues.push(...limitations);

    // Check prompt quality
    const qualityIssues = this.checkPromptQuality(prompt, model);
    issues.push(...qualityIssues);

    return {
      valid: issues.filter(i => i.severity === 'error').length === 0,
      tokenCount,
      issues,
      warnings: issues.filter(i => i.severity === 'warning'),
      errors: issues.filter(i => i.severity === 'error'),
    };
  }

  /**
   * Compress prompt while maintaining meaning
   */
  compressPrompt(prompt: string, targetRatio: number = this.COMPRESSION_RATIO): string {
    // Remove redundant whitespace
    let compressed = prompt.replace(/\s+/g, ' ').trim();

    // Remove unnecessary phrases
    compressed = this.removeRedundantPhrases(compressed);

    // Shorten examples while keeping structure
    compressed = this.compressExamples(compressed);

    // Use more concise language
    compressed = this.useConciseLanguage(compressed);

    // Ensure we don't exceed target ratio
    const originalLength = prompt.length;
    const targetLength = originalLength * targetRatio;

    if (compressed.length > targetLength) {
      compressed = this.truncateToLength(compressed, targetLength);
    }

    return compressed;
  }

  // Private methods

  private initializeTemplates(): void {
    // OpenAI GPT templates
    this.templates.set('openai', {
      modelFamily: 'openai',
      template: `You are a helpful AI assistant. {prompt}`,
      variables: ['prompt'],
      optimizations: [
        {
          type: 'token_reduction',
          description: 'Remove unnecessary formatting',
          impact: 0.1,
        },
      ],
      constraints: [
        {
          type: 'max_length',
          value: 0.8,
          enforcement: 'strict',
        },
      ],
    });

    // Anthropic Claude templates
    this.templates.set('anthropic', {
      modelFamily: 'anthropic',
      template: `Human: {prompt}\n\nAssistant:`,
      variables: ['prompt'],
      optimizations: [
        {
          type: 'quality_enhancement',
          description: 'Use structured format',
          impact: 0.15,
        },
      ],
      constraints: [
        {
          type: 'format',
          value: 'conversational',
          enforcement: 'strict',
        },
      ],
    });

    // Google Gemini templates
    this.templates.set('gemini', {
      modelFamily: 'gemini',
      template: `{prompt}`,
      variables: ['prompt'],
      optimizations: [
        {
          type: 'latency_optimization',
          description: 'Simplified format',
          impact: 0.05,
        },
      ],
      constraints: [
        {
          type: 'content_filter',
          value: 'strict',
          enforcement: 'flexible',
        },
      ],
    });

    // Local LLM templates
    this.templates.set('local', {
      modelFamily: 'local',
      template: `Task: {prompt}\nResponse:`,
      variables: ['prompt'],
      optimizations: [
        {
          type: 'token_reduction',
          description: 'Minimal formatting',
          impact: 0.2,
        },
      ],
      constraints: [
        {
          type: 'max_length',
          value: 0.6,
          enforcement: 'strict',
        },
      ],
    });
  }

  private initializeModelOptimizations(): void {
    // GPT-4 optimizations
    this.modelOptimizations.set('gpt-4', {
      model: 'gpt-4',
      preferredFormat: 'structured',
      tokenEfficiency: 1.0,
      qualityEnhancements: ['step_by_step', 'examples'],
      limitations: ['verbose_responses'],
      optimizations: ['chain_of_thought', 'few_shot_learning'],
    });

    // Claude-3 optimizations
    this.modelOptimizations.set('claude-3', {
      model: 'claude-3',
      preferredFormat: 'conversational',
      tokenEfficiency: 0.9,
      qualityEnhancements: ['constitutional_ai', 'long_context'],
      limitations: ['shorter_responses'],
      optimizations: ['xml_tags', 'structured_output'],
    });

    // GPT-3.5 optimizations
    this.modelOptimizations.set('gpt-3.5-turbo', {
      model: 'gpt-3.5-turbo',
      preferredFormat: 'direct',
      tokenEfficiency: 1.1,
      qualityEnhancements: ['fast_responses'],
      limitations: ['older_training', 'shorter_context'],
      optimizations: ['prompt_engineering', 'temperature_tuning'],
    });

    // Local LLM optimizations
    this.modelOptimizations.set('local-llm', {
      model: 'local-llm',
      preferredFormat: 'simple',
      tokenEfficiency: 0.8,
      qualityEnhancements: ['cost_effective'],
      limitations: ['limited_knowledge', 'variable_quality'],
      optimizations: ['minimal_prompts', 'clear_instructions'],
    });
  }

  private applyModelFormatting(
    prompt: string,
    model: ModelCapability,
    template: PromptTemplate | null
  ): string {
    if (!template) {
      return prompt;
    }

    // Replace variables in template
    let formatted = template.template;
    formatted = formatted.replace('{prompt}', prompt);

    // Add model-specific prefixes/suffixes
    const optimization = this.modelOptimizations.get(model.model);
    if (optimization) {
      switch (optimization.preferredFormat) {
        case 'structured':
          formatted = `Please provide a structured response to: ${formatted}`;
          break;
        case 'conversational':
          formatted = `Let's work through this together: ${formatted}`;
          break;
        case 'simple':
          formatted = `Task: ${formatted}`;
          break;
      }
    }

    return formatted;
  }

  private optimizeTokenUsage(
    prompt: string,
    model: ModelCapability,
    taskAnalysis: TaskAnalysis
  ): string {
    let optimized = prompt;

    // Calculate available tokens
    const availableTokens =
      Math.floor(model.maxTokens * this.MAX_PROMPT_LENGTH) - this.TOKEN_BUFFER;
    const currentTokens = this.estimateTokenCount(prompt, model);

    if (currentTokens > availableTokens) {
      // Need to compress
      const targetRatio = availableTokens / currentTokens;
      optimized = this.compressPrompt(prompt, Math.min(targetRatio, this.COMPRESSION_RATIO));
    }

    // Apply task-specific optimizations
    optimized = this.applyTaskOptimizations(optimized, taskAnalysis, model);

    return optimized;
  }

  private applyConstraints(prompt: string, template: PromptTemplate | null): string {
    if (!template) return prompt;

    let constrained = prompt;

    for (const constraint of template.constraints) {
      switch (constraint.type) {
        case 'max_length':
          const maxLength =
            typeof constraint.value === 'number'
              ? constraint.value
              : parseFloat(constraint.value as string);
          if (constrained.length > maxLength) {
            constrained = constrained.substring(0, maxLength);
          }
          break;

        case 'format':
          // Apply format constraints
          break;

        case 'content_filter':
          constrained = this.applyContentFilter(constrained, constraint);
          break;
      }
    }

    return constrained;
  }

  private addContextEnhancements(
    prompt: string,
    context: RoutingContext | undefined,
    model: ModelCapability
  ): string {
    if (!context) return prompt;

    let enhanced = prompt;

    // Add priority information
    if (context.priority === 'high') {
      enhanced = `URGENT: ${enhanced}`;
    }

    // Add geographic context if relevant
    if (context.clientRegion && context.clientRegion !== model.region) {
      enhanced = `Note: Request from ${context.clientRegion}. ${enhanced}`;
    }

    // Add session context
    if (context.sessionId) {
      enhanced = `Session ${context.sessionId}: ${enhanced}`;
    }

    return enhanced;
  }

  private applyOptimizations(
    prompt: string,
    optimizations: PromptOptimization[],
    taskAnalysis: TaskAnalysis
  ): string {
    let optimized = prompt;

    for (const optimization of optimizations) {
      switch (optimization.type) {
        case 'token_reduction':
          optimized = this.compressPrompt(optimized, 0.9);
          break;

        case 'quality_enhancement':
          optimized = this.enhanceQuality(optimized, taskAnalysis);
          break;

        case 'latency_optimization':
          optimized = this.optimizeForLatency(optimized);
          break;
      }
    }

    return optimized;
  }

  private calculateAdaptationMetrics(
    original: string,
    adapted: string,
    model: ModelCapability
  ): PromptMetrics {
    const originalTokens = this.estimateTokenCount(original, model);
    const adaptedTokens = this.estimateTokenCount(adapted, model);
    const compressionRatio = adaptedTokens / originalTokens;

    return {
      originalLength: original.length,
      adaptedLength: adapted.length,
      originalTokens,
      adaptedTokens,
      compressionRatio,
      tokenSavings: originalTokens - adaptedTokens,
      estimatedCost: adaptedTokens * model.costPerToken,
    };
  }

  private getOptimizationsForModel(model: ModelCapability): PromptOptimization[] {
    const template = this.getTemplateForModel(model);
    return template?.optimizations || [];
  }

  private getTokenMultiplier(model: ModelCapability): number {
    // Different models have different tokenization schemes
    const multipliers: Record<string, number> = {
      'gpt-4': 1.0,
      'gpt-3.5-turbo': 1.1,
      'claude-3': 0.9,
      gemini: 1.0,
      local: 0.8,
    };

    const modelPrefix = model.model.split('-')[0];
    return modelPrefix ? multipliers[modelPrefix] || 1.0 : 1.0;
  }

  private checkModelLimitations(prompt: string, model: ModelCapability): PromptIssue[] {
    const issues: PromptIssue[] = [];

    // Check for known model limitations
    if (model.limitations.includes('high_cost') && prompt.length > 1000) {
      issues.push({
        type: 'cost_warning',
        severity: 'warning',
        message: 'Long prompt may increase costs significantly',
        suggestion: 'Consider compressing the prompt',
      });
    }

    if (model.limitations.includes('shorter_context') && prompt.length > 2000) {
      issues.push({
        type: 'context_limit',
        severity: 'error',
        message: 'Prompt may exceed context window',
        suggestion: 'Shorten prompt or use model with larger context',
      });
    }

    return issues;
  }

  private checkPromptQuality(prompt: string, model: ModelCapability): PromptIssue[] {
    const issues: PromptIssue[] = [];

    // Check for clarity
    if (prompt.length < 10) {
      issues.push({
        type: 'too_short',
        severity: 'warning',
        message: 'Prompt is very short',
        suggestion: 'Add more context and details',
      });
    }

    // Check for ambiguous language
    const ambiguousWords = ['maybe', 'perhaps', 'might', 'could'];
    const hasAmbiguous = ambiguousWords.some(word => prompt.toLowerCase().includes(word));
    if (hasAmbiguous) {
      issues.push({
        type: 'ambiguous',
        severity: 'info',
        message: 'Prompt contains ambiguous language',
        suggestion: 'Use more specific and direct language',
      });
    }

    return issues;
  }

  private removeRedundantPhrases(prompt: string): string {
    const redundancies = [/\bplease\b/gi, /\bkindly\b/gi, /\bif you would\b/gi, /\bcan you\b/gi];

    let cleaned = prompt;
    redundancies.forEach(pattern => {
      cleaned = cleaned.replace(pattern, '');
    });

    return cleaned.trim();
  }

  private compressExamples(prompt: string): string {
    // Find code examples and compress them
    const codeBlockRegex = /```[\s\S]*?```/g;
    return prompt.replace(codeBlockRegex, match => {
      // Keep first and last few lines of code blocks
      const lines = match.split('\n');
      if (lines.length > 10) {
        const firstLines = lines.slice(0, 3);
        const lastLines = lines.slice(-3);
        return [...firstLines, '// ... code compressed ...', ...lastLines].join('\n');
      }
      return match;
    });
  }

  private useConciseLanguage(prompt: string): string {
    const replacements: Record<string, string> = {
      'in order to': 'to',
      'due to the fact that': 'because',
      'in the event that': 'if',
      'for the purpose of': 'to',
      'with regard to': 'about',
      'in accordance with': 'per',
    };

    let concise = prompt;
    Object.entries(replacements).forEach(([long, short]) => {
      concise = concise.replace(new RegExp(long, 'gi'), short);
    });

    return concise;
  }

  private truncateToLength(text: string, maxLength: number): string {
    if (text.length <= maxLength) return text;

    // Try to truncate at sentence boundary
    const sentences = text.split(/[.!?]+/);
    let result = '';

    for (const sentence of sentences) {
      if ((result + sentence).length > maxLength * 0.8) break;
      result += sentence + '. ';
    }

    if (result.length === 0) {
      // Fallback to hard truncation
      result = text.substring(0, maxLength - 3) + '...';
    }

    return result.trim();
  }

  private applyTaskOptimizations(
    prompt: string,
    taskAnalysis: TaskAnalysis,
    model: ModelCapability
  ): string {
    let optimized = prompt;

    // Add task-specific instructions
    if (taskAnalysis.complexity === 'simple') {
      optimized = `Keep response brief and direct: ${optimized}`;
    } else if (taskAnalysis.complexity === 'complex') {
      optimized = `Provide detailed, step-by-step analysis: ${optimized}`;
    }

    // Add domain context
    if (taskAnalysis.domain.length > 0) {
      optimized = `Domain: ${taskAnalysis.domain.join(', ')}. ${optimized}`;
    }

    return optimized;
  }

  private applyContentFilter(prompt: string, constraint: PromptConstraint): string {
    // Basic content filtering - in production would be more sophisticated
    const filteredWords = ['inappropriate', 'offensive']; // Placeholder
    let filtered = prompt;

    filteredWords.forEach(word => {
      filtered = filtered.replace(new RegExp(word, 'gi'), '[FILTERED]');
    });

    return filtered;
  }

  private enhanceQuality(prompt: string, taskAnalysis: TaskAnalysis): string {
    // Add quality enhancement instructions
    return `Provide a high-quality, well-structured response: ${prompt}`;
  }

  private optimizeForLatency(prompt: string): string {
    // Remove complex formatting that might slow processing
    return prompt.replace(/\n\n+/g, '\n').trim();
  }
}

// ============================================================================
// SUPPORTING INTERFACES AND TYPES
// ============================================================================

export interface AdaptedPrompt {
  originalPrompt: string;
  adaptedPrompt: string;
  model: ModelCapability;
  template?: string;
  optimizations: string[];
  metrics: PromptMetrics;
}

export interface PromptMetrics {
  originalLength: number;
  adaptedLength: number;
  originalTokens: number;
  adaptedTokens: number;
  compressionRatio: number;
  tokenSavings: number;
  estimatedCost: number;
}

export interface PromptValidation {
  valid: boolean;
  tokenCount: number;
  issues: PromptIssue[];
  warnings: PromptIssue[];
  errors: PromptIssue[];
}

export interface PromptIssue {
  type: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
  suggestion?: string;
}

export interface PromptTemplate {
  modelFamily: string;
  template: string;
  variables: string[];
  optimizations: PromptOptimization[];
  constraints: PromptConstraint[];
}

export interface PromptOptimization {
  type: 'token_reduction' | 'quality_enhancement' | 'latency_optimization';
  description: string;
  impact: number; // percentage improvement
}

export interface PromptConstraint {
  type: 'max_length' | 'format' | 'content_filter';
  value: any;
  enforcement: 'strict' | 'flexible';
}

export interface ModelOptimization {
  model: string;
  preferredFormat: 'structured' | 'conversational' | 'direct' | 'simple';
  tokenEfficiency: number;
  qualityEnhancements: string[];
  limitations: string[];
  optimizations: string[];
}

// ============================================================================
// USAGE EXAMPLES AND INTEGRATION PATTERNS
// ============================================================================

/**
 * Example: Basic Prompt Adaptation
 *
 * const adapter = new PromptAdapter();
 * const adapted = adapter.adaptPrompt(
 *   "Generate a React component",
 *   modelCapability,
 *   taskAnalysis,
 *   routingContext
 * );
 * console.log(`Compression: ${(adapted.metrics.compressionRatio * 100).toFixed(1)}%`);
 */

/**
 * Example: Prompt Validation
 *
 * const validation = adapter.validatePrompt(prompt, model);
 * if (!validation.valid) {
 *   console.error('Validation errors:', validation.errors);
 *   // Handle validation issues
 * }
 */

/**
 * Example: Token Optimization
 *
 * const compressed = adapter.compressPrompt(longPrompt, 0.7);
 * console.log(`Original: ${longPrompt.length}, Compressed: ${compressed.length}`);
 */

/**
 * Example: Model-Specific Formatting
 *
 * // For Claude models
 * const claudePrompt = adapter.adaptPrompt(
 *   prompt,
 *   claudeModel,
 *   taskAnalysis
 * );
 * // Result: "Human: {prompt}\n\nAssistant:"
 *
 * // For GPT models
 * const gptPrompt = adapter.adaptPrompt(
 *   prompt,
 *   gptModel,
 *   taskAnalysis
 * );
 * // Result: "You are a helpful AI assistant. {prompt}"
 */

export default PromptAdapter;
