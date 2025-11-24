// Main exports for the AI Agent Framework
export * from './interfaces';
export * from './memory';
export * from './sme';
export * from './hallucination';
export * from './pipeline';

// Framework initialization and utilities
import { InMemoryContextualMemory } from './memory';
import { ConfigurableSMEValidator } from './sme';
import { ConfidenceHallucinationDetector } from './hallucination';
import { SequentialAutomationPipeline } from './pipeline';
import { FrameworkConfig } from './interfaces';

export class AIAgentFramework {
  private config: FrameworkConfig;
  private memory: InMemoryContextualMemory;
  private hallucinationDetector: ConfidenceHallucinationDetector;

  constructor(config: FrameworkConfig) {
    this.config = config;
    this.memory = new InMemoryContextualMemory();
    this.hallucinationDetector = new ConfidenceHallucinationDetector(
      config.validationConfig.defaultConfidenceThreshold
    );

    if (config.enableLogging) {
      console.log(`🤖 AI Agent Framework initialized with log level: ${config.logLevel}`);
    }
  }

  // Core components
  getMemory(): InMemoryContextualMemory {
    return this.memory;
  }

  getHallucinationDetector(): ConfidenceHallucinationDetector {
    return this.hallucinationDetector;
  }

  // Factory methods for common components
  createSMEValidator(domain: string, rules: any[] = []): ConfigurableSMEValidator {
    return new ConfigurableSMEValidator(domain, rules);
  }

  createPipeline(): SequentialAutomationPipeline {
    return new SequentialAutomationPipeline();
  }

  // Utility methods
  async validateWithSME(domain: string, data: Record<string, unknown>, rules: any[] = []): Promise<any> {
    const validator = this.createSMEValidator(domain, rules);
    return validator.validate(data);
  }

  async detectHallucination(output: unknown): Promise<any> {
    return this.hallucinationDetector.detect(output, this.memory);
  }

  // Framework health check
  getHealthStatus() {
    return {
      status: 'healthy',
      memoryStats: this.memory.getStats(),
      config: this.config,
      timestamp: new Date().toISOString()
    };
  }
}

// Convenience initialization function
export function initializeFramework(config: FrameworkConfig): AIAgentFramework {
  return new AIAgentFramework(config);
}

// Default configuration
export const defaultConfig: FrameworkConfig = {
  enableLogging: true,
  logLevel: 'info',
  memoryConfig: {
    maxShortTermSize: 100,
    maxLongTermSize: 1000
  },
  validationConfig: {
    defaultConfidenceThreshold: 0.8
  }
};

// Quick start function
export function createDefaultFramework(): AIAgentFramework {
  return initializeFramework(defaultConfig);
}