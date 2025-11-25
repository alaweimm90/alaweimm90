// Core Agent Interfaces
export interface SelfLearningAgent {
  learn(experience: Experience): Promise<void>;
  adapt(parameters: Partial<AgentParameters>): void;
  predict(input: unknown): Promise<PredictionResult>;
  getMetrics(): LearningMetrics;
}

export interface AgentParameters {
  learningRate: number;
  explorationRate: number;
  memorySize: number;
  confidenceThreshold: number;
}

export interface Experience {
  input: unknown;
  output: unknown;
  reward: number;
  context: ContextualMemory;
  timestamp: Date;
}

export interface PredictionResult {
  prediction: unknown;
  confidence: number;
  alternatives?: unknown[];
}

// Memory System
export interface ContextualMemory {
  shortTerm: MemoryEntry[];
  longTerm: MemoryEntry[];
  episodic: MemoryEntry[];
  add(entry: MemoryEntry): void;
  retrieve(query: string): MemoryEntry[];
  consolidate(): void;
  getStats(): MemoryStats;
}

export interface MemoryEntry {
  id: string;
  timestamp: Date;
  type: string;
  content: unknown;
  confidence: number;
  tags: string[];
}

export interface MemoryStats {
  shortTermCount: number;
  longTermCount: number;
  episodicCount: number;
  totalMemories: number;
}

// SME Validation
export interface DomainSMEValidator {
  validate(data: Record<string, unknown>): ValidationResult;
  addRule(rule: ValidationRule): void;
  removeRule(ruleId: string): void;
  getRules(): ValidationRule[];
}

export interface ValidationRule {
  id: string;
  name: string;
  condition: string;
  severity: 'error' | 'warning';
  message: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  domain: string;
}

// Hallucination Detection
export interface HallucinationDetector {
  detect(output: unknown, context: ContextualMemory): HallucinationResult;
}

export interface HallucinationResult {
  isHallucination: boolean;
  confidence: number;
  reasons?: string[];
}

// Learning Metrics
export interface LearningMetrics {
  totalExperiences: number;
  averageReward: number;
  adaptationCount: number;
  lastUpdate: Date;
}

// Pipeline System
export interface AutomationPipeline {
  addStage(stage: PipelineStage): void;
  execute(input: unknown): Promise<PipelineResult>;
  getStages(): PipelineStage[];
}

export interface PipelineStage {
  id: string;
  name: string;
  execute(input: unknown): Promise<unknown>;
}

export interface PipelineResult {
  success: boolean;
  output: unknown;
  errors: string[];
  executionTime: number;
}

// Framework Configuration
export interface FrameworkConfig {
  enableLogging: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  memoryConfig: {
    maxShortTermSize: number;
    maxLongTermSize: number;
  };
  validationConfig: {
    defaultConfidenceThreshold: number;
  };
}

// Domain Adapters
export interface DomainAdapter {
  initialize(config: unknown): Promise<void>;
  process(input: unknown): Promise<unknown>;
  getCapabilities(): string[];
}

// CLI Adapter
export interface CLIAdapter {
  processCommand(command: string, args?: string[]): Promise<string>;
  getCommandHistory(): string[];
}

// Web Adapter
export interface WebAdapter {
  middleware(): Function;
  getRoutes(): string[];
}