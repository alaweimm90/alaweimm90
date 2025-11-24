/**
 * TRAE Model Selector Template
 * Intelligent Model Selection Engine with Complexity Analysis
 *
 * This template provides a complete model selection system that analyzes task
 * complexity and selects optimal models for cost-quality balance.
 *
 * Features:
 * - 8-indicator task complexity analysis
 * - 4-tier complexity classification
 * - Multi-criteria model ranking (quality, capability, cost, reliability)
 * - Automatic fallback chain building
 * - Compliance-aware model selection
 * - Framework and language agnostic design
 */

// ============================================================================
// TYPE DEFINITIONS (Language Agnostic)
// ============================================================================

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

export interface RoutingDecision {
  selectedModel: ModelCapability;
  fallbackChain: ModelCapability[];
  estimatedCost: number;
  estimatedLatency: number;
  confidence: number;
  reasoning: string[];
  costSavings: number;
  riskAssessment: RiskLevel;
  complianceValidation: ComplianceValidation;
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface ComplianceValidation {
  framework: ComplianceFramework;
  validated: boolean;
  requirements: string[];
  violations: string[];
  complianceSurcharge?: number;
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

export enum CostOptimizationMode {
  AGGRESSIVE = 'aggressive',
  BALANCED = 'balanced',
  QUALITY_FIRST = 'quality_first',
}

// ============================================================================
// DEPENDENCY INTERFACES
// ============================================================================

export interface AIEngineIntegration {
  generateText(prompt: string, options?: any): Promise<any>;
  getAvailableModels(): ModelCapability[];
  estimateCost(model: string, tokens: number): number;
  getModelCapabilities(model: string): ModelCapability | null;
  validateCompliance(model: ModelCapability, context: ComplianceContext): ComplianceValidation;
}

// ============================================================================
// MODEL SELECTOR IMPLEMENTATION
// ============================================================================

export class ModelSelector {
  private availableModels: ModelCapability[] = [];
  private aiEngine: AIEngineIntegration;
  private complexityThresholds = {
    [TaskComplexity.SIMPLE]: {
      maxTokens: 1000,
      requiredCapabilities: ['basic_reasoning'],
      qualityThreshold: 70,
    },
    [TaskComplexity.MODERATE]: {
      maxTokens: 4000,
      requiredCapabilities: ['code_generation', 'analysis'],
      qualityThreshold: 80,
    },
    [TaskComplexity.COMPLEX]: {
      maxTokens: 8000,
      requiredCapabilities: ['advanced_reasoning', 'complex_analysis'],
      qualityThreshold: 90,
    },
    [TaskComplexity.CRITICAL]: {
      maxTokens: 16000,
      requiredCapabilities: ['expert_reasoning', 'safety_critical'],
      qualityThreshold: 95,
    },
  };

  constructor(aiEngine: AIEngineIntegration) {
    this.aiEngine = aiEngine;
    this.initializeModels();
  }

  /**
   * Analyze task complexity and requirements
   */
  async analyzeTask(
    prompt: string,
    context?: any,
    userPreferences?: any,
    complianceContext?: ComplianceContext
  ): Promise<TaskAnalysis> {
    const analysis = await this.performComplexityAnalysis(prompt, context);

    // Adjust based on user preferences
    if (userPreferences?.priority === 'cost') {
      analysis.costSensitivity = CostSensitivity.HIGH;
    } else if (userPreferences?.priority === 'quality') {
      analysis.costSensitivity = CostSensitivity.LOW;
    }

    // Consider context length
    analysis.contextLength = this.estimateContextLength(prompt, context);

    // Determine domain expertise needed
    analysis.domain = this.identifyDomain(prompt, context);

    // Add compliance requirements
    if (complianceContext) {
      analysis.complianceRequirements = this.determineComplianceRequirements(
        prompt,
        complianceContext
      );
    }

    return analysis;
  }

  /**
   * Select optimal model based on task analysis
   */
  async selectModel(
    taskAnalysis: TaskAnalysis,
    costOptimizationMode: CostOptimizationMode,
    geographicPreference?: GeographicRegion
  ): Promise<RoutingDecision> {
    const candidates = this.filterCandidateModels(taskAnalysis);
    const rankedModels = this.rankModelsByScore(candidates, taskAnalysis, costOptimizationMode);

    if (rankedModels.length === 0) {
      throw new Error('No suitable models found for task requirements');
    }

    const selectedModel = rankedModels[0];

    if (!selectedModel) {
      throw new Error('No suitable models found for task requirements');
    }

    const fallbackChain = this.buildFallbackChain(selectedModel, rankedModels);

    const decision: RoutingDecision = {
      selectedModel,
      fallbackChain,
      estimatedCost: this.calculateEstimatedCost(selectedModel, taskAnalysis),
      estimatedLatency: this.calculateEstimatedLatency(selectedModel, geographicPreference),
      confidence: this.calculateConfidence(selectedModel, taskAnalysis),
      reasoning: this.generateReasoning(selectedModel, taskAnalysis, costOptimizationMode),
      costSavings: this.calculateCostSavings(selectedModel, rankedModels),
      riskAssessment: this.assessRisk(selectedModel, taskAnalysis),
      complianceValidation: this.validateModelCompliance(selectedModel, taskAnalysis),
    };

    return decision;
  }

  /**
   * Get all available models
   */
  getAvailableModels(): ModelCapability[] {
    return [...this.availableModels];
  }

  /**
   * Update model capabilities (for dynamic model management)
   */
  updateModelCapabilities(models: ModelCapability[]): void {
    this.availableModels = models;
  }

  // Private methods

  private async performComplexityAnalysis(prompt: string, context?: any): Promise<TaskAnalysis> {
    const promptLength = prompt.length;
    const estimatedTokens = Math.ceil(promptLength / 4); // Rough token estimation

    // Analyze prompt complexity indicators
    const complexityIndicators = this.analyzeComplexityIndicators(prompt, context);

    // Determine complexity level
    const complexity = this.determineComplexityLevel(
      estimatedTokens,
      complexityIndicators,
      context
    );

    // Identify required capabilities
    const requiredCapabilities = this.identifyRequiredCapabilities(
      complexity,
      complexityIndicators
    );

    // Assess time sensitivity
    const timeSensitivity = this.assessTimeSensitivity(prompt, context);

    return {
      complexity,
      estimatedTokens,
      requiredCapabilities,
      timeSensitivity,
      costSensitivity: CostSensitivity.MEDIUM, // Default, adjusted by caller
      contextLength: estimatedTokens,
      domain: [], // Will be set by caller
    };
  }

  private analyzeComplexityIndicators(prompt: string, context?: any): ComplexityIndicators {
    const indicators = {
      hasCode: /```[\s\S]*?```|function|class|interface|import|export/.test(prompt),
      hasMath: /[+\-*/=<>≠≈∑∫√]|\b(math|calculate|compute|algorithm)\b/i.test(prompt),
      hasReasoning: /\b(analyze|explain|reason|therefore|because|however)\b/i.test(prompt),
      hasMultipleSteps:
        /\b(first|second|third|then|next|finally|step)\b.*\b(first|second|third|then|next|finally|step)\b/i.test(
          prompt
        ),
      hasSafetyCritical: /\b(security|safe|critical|emergency|danger)\b/i.test(prompt),
      hasComplexLogic: /\b(if|else|switch|loop|recursion|optimization|architecture)\b/i.test(
        prompt
      ),
      contextSize: context ? JSON.stringify(context).length : 0,
    };

    return indicators;
  }

  private determineComplexityLevel(
    estimatedTokens: number,
    indicators: ComplexityIndicators,
    context?: any
  ): TaskComplexity {
    let score = 0;

    // Token-based scoring
    if (estimatedTokens > 12000) score += 3;
    else if (estimatedTokens > 6000) score += 2;
    else if (estimatedTokens > 2000) score += 1;

    // Indicator-based scoring
    if (indicators.hasCode) score += 2;
    if (indicators.hasMath) score += 2;
    if (indicators.hasReasoning) score += 1;
    if (indicators.hasMultipleSteps) score += 2;
    if (indicators.hasSafetyCritical) score += 3;
    if (indicators.hasComplexLogic) score += 2;
    if (indicators.contextSize > 10000) score += 1;

    // Context-based adjustments
    if (context?.priority === 'critical') score += 2;
    if (context?.complexity === 'high') score += 1;

    if (score >= 8) return TaskComplexity.CRITICAL;
    if (score >= 5) return TaskComplexity.COMPLEX;
    if (score >= 3) return TaskComplexity.MODERATE;
    return TaskComplexity.SIMPLE;
  }

  private identifyRequiredCapabilities(
    complexity: TaskComplexity,
    indicators: ComplexityIndicators
  ): string[] {
    const capabilities = new Set<string>();

    // Base capabilities by complexity
    const baseCapabilities = this.complexityThresholds[complexity].requiredCapabilities;
    baseCapabilities.forEach(cap => capabilities.add(cap));

    // Indicator-based capabilities
    if (indicators.hasCode) {
      capabilities.add('code_generation');
      capabilities.add('syntax_analysis');
    }
    if (indicators.hasMath) {
      capabilities.add('mathematical_reasoning');
    }
    if (indicators.hasSafetyCritical) {
      capabilities.add('safety_analysis');
      capabilities.add('risk_assessment');
    }
    if (indicators.hasComplexLogic) {
      capabilities.add('logical_reasoning');
      capabilities.add('system_design');
    }

    return Array.from(capabilities);
  }

  private assessTimeSensitivity(prompt: string, context?: any): TimeSensitivity {
    if (context?.priority === 'critical' || context?.urgency === 'high') {
      return TimeSensitivity.CRITICAL;
    }
    if (/\b(urgent|immediate|asap|emergency)\b/i.test(prompt)) {
      return TimeSensitivity.HIGH;
    }
    if (/\b(soon|quickly|fast)\b/i.test(prompt)) {
      return TimeSensitivity.MEDIUM;
    }
    return TimeSensitivity.LOW;
  }

  private identifyDomain(prompt: string, context?: any): string[] {
    const domains = new Set<string>();

    // Code-related domains
    if (/\b(javascript|typescript|python|java|rust|go|react|node|npm)\b/i.test(prompt)) {
      domains.add('programming');
    }

    // Business domains
    if (/\b(business|marketing|sales|finance|strategy|management)\b/i.test(prompt)) {
      domains.add('business');
    }

    // Technical domains
    if (/\b(security|encryption|authentication|database|api|infrastructure)\b/i.test(prompt)) {
      domains.add('technical');
    }

    // Scientific domains
    if (/\b(science|research|analysis|data|statistics|machine.learning)\b/i.test(prompt)) {
      domains.add('scientific');
    }

    return Array.from(domains);
  }

  private determineComplianceRequirements(
    prompt: string,
    context: ComplianceContext
  ): ComplianceFramework[] {
    const requirements = new Set<ComplianceFramework>();

    // Data classification requirements
    if (
      context.dataClassification === DataClassification.CONFIDENTIAL ||
      context.dataClassification === DataClassification.RESTRICTED
    ) {
      requirements.add(ComplianceFramework.GDPR);
      requirements.add(ComplianceFramework.HIPAA);
    }

    // Geographic restrictions
    if (context.geographicRestrictions.length > 0) {
      requirements.add(ComplianceFramework.GDPR);
    }

    // Audit requirements
    if (context.auditRequirements) {
      requirements.add(ComplianceFramework.SOC2);
    }

    // Financial data
    if (/\b(financial|payment|credit|transaction)\b/i.test(prompt)) {
      requirements.add(ComplianceFramework.PCI_DSS);
    }

    // Health data
    if (/\b(health|medical|patient|diagnosis|treatment)\b/i.test(prompt)) {
      requirements.add(ComplianceFramework.HIPAA);
    }

    return Array.from(requirements);
  }

  private estimateContextLength(prompt: string, context?: any): number {
    const promptTokens = Math.ceil(prompt.length / 4);
    const contextTokens = context ? Math.ceil(JSON.stringify(context).length / 4) : 0;
    return promptTokens + contextTokens;
  }

  private filterCandidateModels(taskAnalysis: TaskAnalysis): ModelCapability[] {
    return this.availableModels.filter(model => {
      // Check token capacity
      if (model.maxTokens < taskAnalysis.estimatedTokens) {
        return false;
      }

      // Check required capabilities
      const hasRequiredCapabilities = taskAnalysis.requiredCapabilities.every(
        cap => model.specializations.includes(cap) || model.specializations.includes('general')
      );

      if (!hasRequiredCapabilities) {
        return false;
      }

      // Check quality threshold
      const requiredQuality = this.complexityThresholds[taskAnalysis.complexity].qualityThreshold;
      if (model.qualityScore < requiredQuality) {
        return false;
      }

      // Check compliance requirements
      if (taskAnalysis.complianceRequirements) {
        const hasComplianceSupport = taskAnalysis.complianceRequirements.every(framework =>
          model.compliance.includes(framework)
        );
        if (!hasComplianceSupport) {
          return false;
        }
      }

      return true;
    });
  }

  private rankModelsByScore(
    candidates: ModelCapability[],
    taskAnalysis: TaskAnalysis,
    costOptimizationMode: CostOptimizationMode
  ): ModelCapability[] {
    return candidates.sort((a, b) => {
      const scoreA = this.calculateModelScore(a, taskAnalysis, costOptimizationMode);
      const scoreB = this.calculateModelScore(b, taskAnalysis, costOptimizationMode);

      return scoreB - scoreA; // Higher score first
    });
  }

  private calculateModelScore(
    model: ModelCapability,
    taskAnalysis: TaskAnalysis,
    costOptimizationMode: CostOptimizationMode
  ): number {
    let score = 0;

    // Quality score (0-40 points)
    score += (model.qualityScore / 100) * 40;

    // Capability match (0-30 points)
    const capabilityMatch = this.calculateCapabilityMatch(model, taskAnalysis);
    score += capabilityMatch * 30;

    // Cost efficiency (0-20 points) - varies by optimization mode
    const costEfficiency = this.calculateCostEfficiency(model, costOptimizationMode);
    score += costEfficiency * 20;

    // Reliability bonus (0-10 points)
    score += (model.reliability / 100) * 10;

    return score;
  }

  private calculateCapabilityMatch(model: ModelCapability, taskAnalysis: TaskAnalysis): number {
    const requiredCaps = taskAnalysis.requiredCapabilities;
    const modelCaps = model.specializations;

    const matches = requiredCaps.filter(
      cap => modelCaps.includes(cap) || modelCaps.includes('general')
    ).length;

    return matches / requiredCaps.length;
  }

  private calculateCostEfficiency(
    model: ModelCapability,
    costOptimizationMode: CostOptimizationMode
  ): number {
    // Lower cost per token = higher efficiency score
    const baseEfficiency = Math.max(0, 1 - model.costPerToken / 0.01); // Normalize against $0.01/token

    // Adjust based on optimization mode
    switch (costOptimizationMode) {
      case CostOptimizationMode.AGGRESSIVE:
        return baseEfficiency * 1.5; // Prioritize cost more heavily
      case CostOptimizationMode.BALANCED:
        return baseEfficiency;
      case CostOptimizationMode.QUALITY_FIRST:
        return baseEfficiency * 0.5; // De-prioritize cost
      default:
        return baseEfficiency;
    }
  }

  private buildFallbackChain(
    selectedModel: ModelCapability,
    rankedModels: ModelCapability[]
  ): ModelCapability[] {
    // Take top 3 models as fallback chain, excluding the selected one
    return rankedModels.filter(model => model.name !== selectedModel.name).slice(0, 3);
  }

  private calculateEstimatedCost(model: ModelCapability, taskAnalysis: TaskAnalysis): number {
    const inputTokens = taskAnalysis.estimatedTokens;
    const outputTokens = Math.ceil(inputTokens * 0.3); // Estimate 30% output tokens
    const totalTokens = inputTokens + outputTokens;

    return totalTokens * model.costPerToken;
  }

  private calculateEstimatedLatency(
    model: ModelCapability,
    geographicPreference?: GeographicRegion
  ): number {
    let baseLatency = model.latency;

    // Adjust for geographic distance if preference specified
    if (geographicPreference && model.region !== geographicPreference) {
      baseLatency += 100; // Add 100ms for cross-region calls
    }

    return baseLatency;
  }

  private calculateConfidence(model: ModelCapability, taskAnalysis: TaskAnalysis): number {
    const qualityMatch = model.qualityScore / 100;
    const capabilityMatch = this.calculateCapabilityMatch(model, taskAnalysis);
    const reliabilityFactor = model.reliability / 100;

    // Weighted average
    return Math.round((qualityMatch * 0.4 + capabilityMatch * 0.4 + reliabilityFactor * 0.2) * 100);
  }

  private generateReasoning(
    model: ModelCapability,
    taskAnalysis: TaskAnalysis,
    costOptimizationMode: CostOptimizationMode
  ): string[] {
    const reasoning = [];

    reasoning.push(
      `Selected ${model.tier} tier model ${model.name} for ${taskAnalysis.complexity} complexity task`
    );

    if (costOptimizationMode === CostOptimizationMode.AGGRESSIVE) {
      reasoning.push('Cost optimization mode: prioritized efficiency over premium features');
    }

    const capabilityMatch = this.calculateCapabilityMatch(model, taskAnalysis);
    reasoning.push(`Capability match: ${(capabilityMatch * 100).toFixed(0)}%`);

    reasoning.push(
      `Estimated cost: $${this.calculateEstimatedCost(model, taskAnalysis).toFixed(4)}`
    );

    return reasoning;
  }

  private calculateCostSavings(
    selectedModel: ModelCapability,
    rankedModels: ModelCapability[]
  ): number {
    if (rankedModels.length < 2) return 0;

    // Compare against the most expensive alternative
    const mostExpensive = rankedModels.reduce((max, model) =>
      model.costPerToken > max.costPerToken ? model : max
    );

    const savings =
      (mostExpensive.costPerToken - selectedModel.costPerToken) / mostExpensive.costPerToken;
    return Math.round(savings * 100);
  }

  private assessRisk(model: ModelCapability, taskAnalysis: TaskAnalysis): RiskLevel {
    if (taskAnalysis.complexity === TaskComplexity.CRITICAL && model.tier === RoutingTier.TIER_3) {
      return RiskLevel.HIGH;
    }

    if (model.reliability < 90) {
      return RiskLevel.MEDIUM;
    }

    if (taskAnalysis.timeSensitivity === TimeSensitivity.CRITICAL && model.latency > 5000) {
      return RiskLevel.MEDIUM;
    }

    return RiskLevel.LOW;
  }

  private validateModelCompliance(
    model: ModelCapability,
    taskAnalysis: TaskAnalysis
  ): ComplianceValidation {
    if (!taskAnalysis.complianceRequirements || taskAnalysis.complianceRequirements.length === 0) {
      return {
        framework: ComplianceFramework.GDPR,
        validated: true,
        requirements: [],
        violations: [],
      };
    }

    const violations: string[] = [];
    const requirements: string[] = [];

    for (const framework of taskAnalysis.complianceRequirements) {
      if (!model.compliance.includes(framework)) {
        violations.push(`${framework} compliance not supported by model ${model.name}`);
      } else {
        requirements.push(`${framework} compliance verified`);
      }
    }

    return {
      framework: taskAnalysis.complianceRequirements[0],
      validated: violations.length === 0,
      requirements,
      violations,
    };
  }

  private initializeModels(): void {
    // Initialize with common model capabilities
    // In production, this would be loaded from configuration or API
    this.availableModels = [
      {
        name: 'gpt-4',
        provider: 'openai',
        model: 'gpt-4',
        tier: RoutingTier.TIER_1,
        region: GeographicRegion.NORTH_AMERICA,
        maxTokens: 8192,
        costPerToken: 0.03,
        qualityScore: 95,
        latency: 2000,
        reliability: 98,
        specializations: ['advanced_reasoning', 'code_generation', 'analysis', 'creative_writing'],
        limitations: ['high_cost', 'rate_limits'],
        compliance: [ComplianceFramework.GDPR, ComplianceFramework.SOC2],
      },
      {
        name: 'claude-3-opus',
        provider: 'anthropic',
        model: 'claude-3-opus-20240229',
        tier: RoutingTier.TIER_1,
        region: GeographicRegion.NORTH_AMERICA,
        maxTokens: 200000,
        costPerToken: 0.015,
        qualityScore: 96,
        latency: 1800,
        reliability: 99,
        specializations: ['advanced_reasoning', 'safety_analysis', 'long_context', 'analysis'],
        limitations: ['new_model', 'limited_fine_tuning'],
        compliance: [ComplianceFramework.GDPR, ComplianceFramework.SOC2, ComplianceFramework.HIPAA],
      },
      {
        name: 'gpt-3.5-turbo',
        provider: 'openai',
        model: 'gpt-3.5-turbo',
        tier: RoutingTier.TIER_2,
        region: GeographicRegion.GLOBAL,
        maxTokens: 4096,
        costPerToken: 0.002,
        qualityScore: 85,
        latency: 800,
        reliability: 95,
        specializations: ['general_reasoning', 'code_generation', 'conversation'],
        limitations: ['older_model', 'shorter_context'],
        compliance: [ComplianceFramework.GDPR, ComplianceFramework.SOC2],
      },
      {
        name: 'claude-3-sonnet',
        provider: 'anthropic',
        model: 'claude-3-sonnet-20240229',
        tier: RoutingTier.TIER_2,
        region: GeographicRegion.GLOBAL,
        maxTokens: 200000,
        costPerToken: 0.003,
        qualityScore: 90,
        latency: 1200,
        reliability: 97,
        specializations: ['balanced_reasoning', 'code_generation', 'analysis', 'long_context'],
        limitations: ['new_model'],
        compliance: [ComplianceFramework.GDPR, ComplianceFramework.SOC2, ComplianceFramework.HIPAA],
      },
      {
        name: 'local-llm',
        provider: 'local',
        model: 'llama-2-7b',
        tier: RoutingTier.TIER_3,
        region: GeographicRegion.GLOBAL,
        maxTokens: 4096,
        costPerToken: 0.0001,
        qualityScore: 75,
        latency: 500,
        reliability: 90,
        specializations: ['basic_reasoning', 'general_tasks'],
        limitations: ['limited_capabilities', 'local_resources'],
        compliance: [ComplianceFramework.GDPR], // Local models can be compliant if properly configured
      },
    ];
  }
}

// ============================================================================
// SUPPORTING INTERFACES AND TYPES
// ============================================================================

export interface ComplexityIndicators {
  hasCode: boolean;
  hasMath: boolean;
  hasReasoning: boolean;
  hasMultipleSteps: boolean;
  hasSafetyCritical: boolean;
  hasComplexLogic: boolean;
  contextSize: number;
}

// ============================================================================
// USAGE EXAMPLES AND INTEGRATION PATTERNS
// ============================================================================

/**
 * Example: Basic Model Selection
 *
 * const selector = new ModelSelector(aiEngine);
 * const analysis = await selector.analyzeTask("Generate a React component");
 * const decision = await selector.selectModel(analysis, CostOptimizationMode.BALANCED);
 */

/**
 * Example: Compliance-Aware Selection
 *
 * const complianceContext = {
 *   dataClassification: DataClassification.CONFIDENTIAL,
 *   retentionPolicy: RetentionPolicy.LONG_TERM,
 *   auditRequirements: true,
 *   geographicRestrictions: [GeographicRegion.CHINA]
 * };
 *
 * const analysis = await selector.analyzeTask(prompt, context, preferences, complianceContext);
 * const decision = await selector.selectModel(analysis, CostOptimizationMode.BALANCED);
 */

/**
 * Example: Custom Model Configuration
 *
 * const customModels = [
 *   {
 *     name: 'enterprise-gpt-4',
 *     provider: 'openai',
 *     model: 'gpt-4-enterprise',
 *     tier: RoutingTier.TIER_1,
 *     region: GeographicRegion.EUROPE,
 *     maxTokens: 32768,
 *     costPerToken: 0.06,
 *     qualityScore: 98,
 *     latency: 1500,
 *     reliability: 99,
 *     specializations: ['enterprise', 'security', 'compliance'],
 *     limitations: ['premium_pricing'],
 *     compliance: [ComplianceFramework.GDPR, ComplianceFramework.SOC2, ComplianceFramework.HIPAA]
 *   }
 * ];
 *
 * selector.updateModelCapabilities(customModels);
 */

export default ModelSelector;
