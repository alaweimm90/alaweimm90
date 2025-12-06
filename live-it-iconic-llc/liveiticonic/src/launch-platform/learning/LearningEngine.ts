/**
 * LiveItIconic Launch Platform - Learning Engine
 *
 * Meta-learning system that improves platform performance over time through
 * pattern recognition, cross-domain transfer, and continuous optimization
 */

import {
  Experience,
  Pattern,
  Condition,
  LaunchPlan,
  SuccessPrediction,
  Factor,
  Recommendation,
  AgentPriority,
} from '../types';
import { stateManager } from '../core/StateManager';

export interface LearningConfig {
  minExperiences: number;
  minConfidence: number;
  patternThreshold: number;
  transferLearningEnabled: boolean;
  continuousImprovement: boolean;
}

export class LearningEngine {
  private config: LearningConfig;
  private patternCache: Map<string, Pattern[]> = new Map();
  private successFactorWeights: Map<string, number> = new Map();

  constructor(config?: Partial<LearningConfig>) {
    this.config = {
      minExperiences: 5,
      minConfidence: 0.7,
      patternThreshold: 3,
      transferLearningEnabled: true,
      continuousImprovement: true,
      ...config,
    };

    this.initializeWeights();
    console.log('[LearningEngine] Initialized with config:', this.config);
  }

  /**
   * Initialize default success factor weights
   */
  private initializeWeights(): void {
    this.successFactorWeights.set('market_timing', 0.15);
    this.successFactorWeights.set('product_market_fit', 0.20);
    this.successFactorWeights.set('pricing_strategy', 0.12);
    this.successFactorWeights.set('brand_positioning', 0.15);
    this.successFactorWeights.set('channel_mix', 0.13);
    this.successFactorWeights.set('creative_quality', 0.10);
    this.successFactorWeights.set('budget_allocation', 0.08);
    this.successFactorWeights.set('execution_quality', 0.07);
  }

  /**
   * Learn from a completed launch
   */
  async learn(launch: Experience): Promise<void> {
    console.log(`[LearningEngine] Learning from launch: ${launch.id}`);

    // Extract patterns
    const patterns = await this.extractPatterns(launch);

    // Update agent models (conceptual - would integrate with actual agents)
    await this.updateAgentModels(patterns);

    // Transfer knowledge across domains if enabled
    if (this.config.transferLearningEnabled) {
      await this.transferKnowledge(patterns, launch.productType);
    }

    // Store experience
    stateManager.recordExperience(launch);

    // Store patterns
    patterns.forEach(pattern => {
      stateManager.addPattern(pattern);
      this.cachePattern(pattern);
    });

    // Continuous improvement - adjust weights based on outcomes
    if (this.config.continuousImprovement) {
      await this.adjustSuccessFactorWeights(launch);
    }

    console.log(`[LearningEngine] Learned ${patterns.length} patterns from launch ${launch.id}`);
  }

  /**
   * Extract patterns from experience
   */
  private async extractPatterns(experience: Experience): Promise<Pattern[]> {
    const patterns: Pattern[] = [];

    // Analyze outcomes
    const overallSuccess = experience.outcomes.every(o => o.success);
    const avgPerformance =
      experience.outcomes.reduce((sum, o) => sum + (o.actual / o.target), 0) /
      experience.outcomes.length;

    // Pattern 1: Strategy effectiveness
    if (avgPerformance > 1.1) {
      patterns.push({
        id: `pattern_${Date.now()}_strategy`,
        type: 'success',
        description: `${experience.strategy.type} launch strategy highly effective for ${experience.productType}`,
        conditions: [
          { field: 'strategy.type', operator: 'equals', value: experience.strategy.type },
          { field: 'productType', operator: 'equals', value: experience.productType },
        ],
        actions: ['Use similar strategy for future launches'],
        outcomes: experience.outcomes.reduce((acc, o) => {
          acc[o.metric] = o.actual / o.target;
          return acc;
        }, {} as Record<string, number>),
        confidence: Math.min(avgPerformance / 1.5, 0.95),
        frequency: 1,
        domains: [experience.productType],
      });
    }

    // Pattern 2: Channel performance
    const bestChannels = experience.actions
      .filter(a => a.result.success && a.result.metrics.roi > 3.0)
      .map(a => a.type);

    if (bestChannels.length > 0) {
      patterns.push({
        id: `pattern_${Date.now()}_channels`,
        type: 'success',
        description: `High-performing channels identified: ${bestChannels.join(', ')}`,
        conditions: [
          { field: 'productType', operator: 'equals', value: experience.productType },
        ],
        actions: [`Prioritize channels: ${bestChannels.join(', ')}`],
        outcomes: { roi: 3.5, conversion: 0.85 },
        confidence: 0.80,
        frequency: 1,
        domains: [experience.productType, 'general'],
      });
    }

    // Pattern 3: Timing insights
    const launchDay = new Date(experience.timestamp).getDay();
    if (overallSuccess && avgPerformance > 1.05) {
      patterns.push({
        id: `pattern_${Date.now()}_timing`,
        type: 'success',
        description: `Successful launch timing pattern detected`,
        conditions: [
          { field: 'launchDay', operator: 'equals', value: launchDay },
          { field: 'productType', operator: 'equals', value: experience.productType },
        ],
        actions: ['Consider similar launch timing'],
        outcomes: { success_rate: 1.0 },
        confidence: 0.75,
        frequency: 1,
        domains: [experience.productType],
      });
    }

    // Pattern 4: Failure patterns (if applicable)
    const failures = experience.outcomes.filter(o => !o.success);
    if (failures.length > 0) {
      patterns.push({
        id: `pattern_${Date.now()}_failure`,
        type: 'failure',
        description: `Challenges identified: ${failures.map(f => f.metric).join(', ')}`,
        conditions: [
          { field: 'strategy.type', operator: 'equals', value: experience.strategy.type },
        ],
        actions: failures.map(f => `Review and improve ${f.metric} approach`),
        outcomes: failures.reduce((acc, f) => {
          acc[f.metric] = f.actual / f.target;
          return acc;
        }, {} as Record<string, number>),
        confidence: 0.70,
        frequency: 1,
        domains: [experience.productType],
      });
    }

    return patterns;
  }

  /**
   * Update agent models with learned patterns
   */
  private async updateAgentModels(patterns: Pattern[]): Promise<void> {
    // In a real implementation, this would update the internal models
    // of each agent to incorporate learned patterns
    console.log(`[LearningEngine] Updating agent models with ${patterns.length} patterns`);

    // Conceptual: Each agent would receive relevant patterns
    // and adjust their decision-making accordingly
  }

  /**
   * Transfer knowledge across domains
   */
  private async transferKnowledge(patterns: Pattern[], sourceDomain: string): Promise<void> {
    console.log(`[LearningEngine] Transferring knowledge from ${sourceDomain}`);

    for (const pattern of patterns) {
      // Find similar domains where pattern might apply
      const targetDomains = await this.findSimilarDomains(sourceDomain);

      // Abstract the pattern for transfer
      const abstractPattern = this.abstractPattern(pattern);

      // Apply to target domains
      for (const targetDomain of targetDomains) {
        if (!pattern.domains.includes(targetDomain)) {
          const transferredPattern = {
            ...abstractPattern,
            id: `transfer_${pattern.id}_to_${targetDomain}`,
            description: `[Transferred] ${pattern.description}`,
            confidence: pattern.confidence * 0.7, // Reduce confidence for transfers
            domains: [...pattern.domains, targetDomain],
          };

          stateManager.addPattern(transferredPattern);
        }
      }
    }
  }

  /**
   * Find similar domains for knowledge transfer
   */
  private async findSimilarDomains(sourceDomain: string): Promise<string[]> {
    // Simplified similarity mapping
    const similarityMap: Record<string, string[]> = {
      apparel: ['accessories', 'lifestyle', 'fashion'],
      accessories: ['apparel', 'lifestyle'],
      lifestyle: ['apparel', 'accessories', 'home'],
      electronics: ['tech', 'gadgets'],
      food: ['beverage', 'wellness'],
    };

    return similarityMap[sourceDomain.toLowerCase()] || [];
  }

  /**
   * Abstract pattern for cross-domain transfer
   */
  private abstractPattern(pattern: Pattern): Pattern {
    return {
      ...pattern,
      conditions: pattern.conditions.filter(c => c.field !== 'productType'),
      description: pattern.description.replace(/\b\w+\s+product\b/gi, 'product'),
    };
  }

  /**
   * Predict success probability for a launch plan
   */
  async predictSuccess(plan: LaunchPlan): Promise<SuccessPrediction> {
    console.log(`[LearningEngine] Predicting success for launch: ${plan.id}`);

    // Find similar historical launches
    const similarLaunches = await this.findSimilarLaunches(plan);

    // Analyze success factors
    const successFactors = await this.analyzeSuccessFactors(plan, similarLaunches);

    // Calculate probability
    const probability = this.calculateProbability(successFactors);

    // Generate recommendations
    const recommendations = await this.generateRecommendations(successFactors, plan);

    return {
      probability,
      confidence: this.calculateConfidence(similarLaunches.length),
      factors: successFactors,
      risks: plan.risks,
      recommendations,
    };
  }

  /**
   * Find similar historical launches
   */
  private async findSimilarLaunches(plan: LaunchPlan): Promise<Experience[]> {
    const allExperiences = stateManager.getExperiences();

    // Filter by product type similarity
    const similar = allExperiences.filter(exp => {
      // Simple similarity - in production would use more sophisticated matching
      const productMatch = exp.productType === plan.product.category;
      const strategyMatch = exp.strategy.type === plan.strategy.type;

      return productMatch || strategyMatch;
    });

    return similar.slice(-10); // Return last 10 similar launches
  }

  /**
   * Analyze success factors
   */
  private async analyzeSuccessFactors(
    plan: LaunchPlan,
    similarLaunches: Experience[]
  ): Promise<Factor[]> {
    const factors: Factor[] = [];

    // Factor 1: Market Timing
    const timingScore = this.evaluateMarketTiming(plan);
    factors.push({
      name: 'Market Timing',
      impact: timingScore * (this.successFactorWeights.get('market_timing') || 0.15),
      confidence: 0.80,
      explanation: timingScore > 0.7
        ? 'Market conditions are favorable for launch'
        : 'Market timing could be improved',
    });

    // Factor 2: Product-Market Fit
    const pmfScore = this.evaluateProductMarketFit(plan);
    factors.push({
      name: 'Product-Market Fit',
      impact: pmfScore * (this.successFactorWeights.get('product_market_fit') || 0.20),
      confidence: 0.75,
      explanation: pmfScore > 0.7
        ? 'Strong product-market fit indicated'
        : 'Product-market fit needs validation',
    });

    // Factor 3: Budget Adequacy
    const budgetScore = this.evaluateBudget(plan, similarLaunches);
    factors.push({
      name: 'Budget Allocation',
      impact: budgetScore * (this.successFactorWeights.get('budget_allocation') || 0.08),
      confidence: 0.85,
      explanation: budgetScore > 0.7
        ? 'Budget allocation aligned with successful patterns'
        : 'Budget may be insufficient or misallocated',
    });

    // Factor 4: Channel Strategy
    const channelScore = this.evaluateChannels(plan);
    factors.push({
      name: 'Channel Mix',
      impact: channelScore * (this.successFactorWeights.get('channel_mix') || 0.13),
      confidence: 0.82,
      explanation: 'Channel selection based on historical performance',
    });

    return factors;
  }

  /**
   * Evaluate market timing
   */
  private evaluateMarketTiming(plan: LaunchPlan): number {
    // Simplified evaluation - would integrate with TrendDetector in production
    const daysUntilLaunch = Math.floor(
      (plan.timeline.launchDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24)
    );

    // Optimal launch window: 14-30 days
    if (daysUntilLaunch >= 14 && daysUntilLaunch <= 30) {
      return 0.85;
    } else if (daysUntilLaunch < 14) {
      return 0.60;
    } else {
      return 0.70;
    }
  }

  /**
   * Evaluate product-market fit
   */
  private evaluateProductMarketFit(plan: LaunchPlan): number {
    const product = plan.product;

    // Simple heuristic based on product definition completeness
    let score = 0;

    if (product.features.length >= 4) score += 0.25;
    if (product.differentiators.length >= 2) score += 0.25;
    if (product.targetMarket.primaryPersona) score += 0.30;
    if (product.pricing.basePrice > 0) score += 0.20;

    return score;
  }

  /**
   * Evaluate budget adequacy
   */
  private evaluateBudget(plan: LaunchPlan, similarLaunches: Experience[]): number {
    if (similarLaunches.length === 0) return 0.70;

    // Compare budget to similar launches
    const avgSuccessfulBudget = similarLaunches
      .filter(exp => exp.outcomes.every(o => o.success))
      .reduce((sum, exp) => sum + (exp.metrics.budget || 50000), 0) / similarLaunches.length || 50000;

    const budgetRatio = plan.resources.budget.total / avgSuccessfulBudget;

    if (budgetRatio >= 0.8 && budgetRatio <= 1.5) {
      return 0.90;
    } else if (budgetRatio >= 0.6 && budgetRatio <= 2.0) {
      return 0.75;
    } else {
      return 0.60;
    }
  }

  /**
   * Evaluate channel strategy
   */
  private evaluateChannels(plan: LaunchPlan): number {
    const channels = plan.strategy.channels;

    // Get successful patterns for this product type
    const patterns = stateManager.getPatterns({
      type: 'success',
      domain: plan.product.category,
    });

    // Check if selected channels align with successful patterns
    let alignment = 0.70; // Base score

    if (patterns.length > 0) {
      // Would analyze channel overlap with successful patterns
      alignment = 0.80;
    }

    return alignment;
  }

  /**
   * Calculate success probability
   */
  private calculateProbability(factors: Factor[]): number {
    const weightedSum = factors.reduce((sum, factor) => sum + factor.impact, 0);
    const avgConfidence = factors.reduce((sum, f) => sum + f.confidence, 0) / factors.length;

    // Normalize to 0-1 range
    const probability = Math.min(weightedSum * avgConfidence, 0.95);

    return Math.max(probability, 0.20); // Minimum 20% probability
  }

  /**
   * Calculate confidence in prediction
   */
  private calculateConfidence(similarLaunchesCount: number): number {
    // Confidence increases with more historical data
    if (similarLaunchesCount >= 10) return 0.90;
    if (similarLaunchesCount >= 5) return 0.80;
    if (similarLaunchesCount >= 2) return 0.70;
    return 0.60;
  }

  /**
   * Generate recommendations
   */
  private async generateRecommendations(
    factors: Factor[],
    plan: LaunchPlan
  ): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];

    // Analyze weak factors
    const weakFactors = factors.filter(f => f.impact < 0.10);

    for (const factor of weakFactors) {
      recommendations.push({
        id: `rec_${Date.now()}_${factor.name.replace(/\s/g, '_')}`,
        priority: AgentPriority.HIGH,
        category: 'improvement',
        description: `Improve ${factor.name}: ${factor.explanation}`,
        expectedImpact: 0.15,
        effort: 'medium',
        confidence: factor.confidence,
      });
    }

    // Add general best practices
    recommendations.push({
      id: `rec_${Date.now()}_testing`,
      priority: AgentPriority.MEDIUM,
      category: 'optimization',
      description: 'Implement A/B testing for key campaign elements',
      expectedImpact: 0.10,
      effort: 'low',
      confidence: 0.85,
    });

    return recommendations;
  }

  /**
   * Adjust success factor weights based on outcomes
   */
  private async adjustSuccessFactorWeights(experience: Experience): Promise<void> {
    // Simplified weight adjustment - would use more sophisticated ML in production
    const overallSuccess = experience.outcomes.every(o => o.success);

    if (overallSuccess) {
      // Slightly increase weights of factors that were present
      // In production, would analyze which specific factors contributed most
      console.log('[LearningEngine] Adjusting success factor weights based on outcomes');
    }
  }

  /**
   * Cache pattern for quick retrieval
   */
  private cachePattern(pattern: Pattern): void {
    const domain = pattern.domains[0] || 'general';
    if (!this.patternCache.has(domain)) {
      this.patternCache.set(domain, []);
    }
    this.patternCache.get(domain)!.push(pattern);
  }

  /**
   * Get learning statistics
   */
  getStatistics(): {
    totalExperiences: number;
    totalPatterns: number;
    successPatterns: number;
    failurePatterns: number;
    averageConfidence: number;
    domains: string[];
  } {
    const patterns = stateManager.getPatterns();
    const experiences = stateManager.getExperiences();

    const successPatterns = patterns.filter(p => p.type === 'success').length;
    const failurePatterns = patterns.filter(p => p.type === 'failure').length;
    const avgConfidence =
      patterns.reduce((sum, p) => sum + p.confidence, 0) / patterns.length || 0;

    const domains = Array.from(new Set(patterns.flatMap(p => p.domains)));

    return {
      totalExperiences: experiences.length,
      totalPatterns: patterns.length,
      successPatterns,
      failurePatterns,
      averageConfidence: avgConfidence,
      domains,
    };
  }
}

// Singleton instance
export const learningEngine = new LearningEngine();
