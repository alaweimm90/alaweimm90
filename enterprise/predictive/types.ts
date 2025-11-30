// ATLAS Enterprise - Predictive Analytics Types

export interface CodeMetrics {
  // Basic code metrics
  linesOfCode: number;
  cyclomaticComplexity: number;
  maintainabilityIndex: number;
  halsteadVolume: number;
  commentRatio: number;

  // Quality indicators
  duplicateLines: number;
  testCoverage?: number;
  technicalDebt: number;

  // Language-specific metrics
  language: string;
  imports: number;
  functions: number;
  classes: number;

  // Predictions and analysis
  predictions?: QualityPrediction;
  timestamp: string;
}

export interface QualityPrediction {
  qualityScore: number; // 0-100
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  issues: PredictedIssue[];
  recommendations: string[];
  confidence: number; // 0-1
}

export interface PredictedIssue {
  type: IssueType;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  location?: CodeLocation;
  probability: number; // 0-1
  impact: string;
}

export type IssueType =
  | 'security-vulnerability'
  | 'performance-issue'
  | 'maintainability-problem'
  | 'reliability-concern'
  | 'scalability-issue'
  | 'code-smell'
  | 'technical-debt';

export interface CodeLocation {
  file: string;
  line: number;
  column?: number;
}

export interface TrainingData {
  metrics: CodeMetrics;
  actualQuality: number; // 0-100
  issues: IssueType[];
  labels: string[];
}

export interface ModelConfig {
  algorithm: 'random-forest' | 'gradient-boosting' | 'neural-network';
  features: string[];
  hyperparameters: Record<string, any>;
  trainingConfig: {
    testSplit: number;
    crossValidationFolds: number;
    maxIterations: number;
  };
}</content>
</edit_file>