// ATLAS Enterprise - Predictive Analytics Engine

import { CodeMetrics } from './types.js';
import { QualityPredictor } from './predictor.js';
import { MetricsCollector } from './collector.js';

export * from './types.js';
export * from './predictor.js';
export * from './collector.js';

/**
 * Predictive Analytics Engine for code quality prediction
 */
export class PredictiveAnalyticsEngine {
  private predictor: QualityPredictor;
  private collector: MetricsCollector;

  constructor() {
    this.predictor = new QualityPredictor();
    this.collector = new MetricsCollector();
  }

  /**
   * Analyze code and predict quality issues
   */
  async analyzeCode(code: string, language: string): Promise<CodeMetrics> {
    const metrics = await this.collector.collectMetrics(code, language);
    const predictions = await this.predictor.predictQuality(metrics);

    return {
      ...metrics,
      predictions,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Train predictive models with historical data
   */
  async trainModel(trainingData: CodeMetrics[]): Promise<void> {
    await this.predictor.train(trainingData);
  }

  /**
   * Get optimization recommendations
   */
  async getRecommendations(metrics: CodeMetrics): Promise<string[]> {
    return this.predictor.generateRecommendations(metrics);
  }
}

/**
 * Initialize predictive analytics
 */
export async function initializePredictiveAnalytics(): Promise<PredictiveAnalyticsEngine> {
  const engine = new PredictiveAnalyticsEngine();
  // Load pre-trained models
  await engine.predictor.loadModels();
  return engine;
}</content>
</edit_file>