import { AutomationPipeline, PipelineStage, PipelineResult } from './interfaces';

export class SequentialAutomationPipeline implements AutomationPipeline {
  private stages: PipelineStage[] = [];
  private results: Map<string, unknown> = new Map();

  addStage(stage: PipelineStage): void {
    this.stages.push(stage);
  }

  async execute(input: unknown): Promise<PipelineResult> {
    const startTime = Date.now();
    const errors: string[] = [];
    let currentInput = input;

    try {
      for (const stage of this.stages) {
        try {
          this.results.set(`input_${stage.id}`, currentInput);
          currentInput = await stage.execute(currentInput);
          this.results.set(`output_${stage.id}`, currentInput);
        } catch (error) {
          const errorMessage = `Stage '${stage.name}' failed: ${error}`;
          errors.push(errorMessage);

          // Continue with next stage or stop on critical errors
          if (this.isCriticalError(error)) {
            break;
          }
        }
      }

      const executionTime = Date.now() - startTime;

      return {
        success: errors.length === 0,
        output: currentInput,
        errors,
        executionTime
      };
    } catch (error) {
      return {
        success: false,
        output: null,
        errors: [`Pipeline execution failed: ${error}`],
        executionTime: Date.now() - startTime
      };
    }
  }

  getStages(): PipelineStage[] {
    return [...this.stages];
  }

  getStageResult(stageId: string): unknown {
    return this.results.get(`output_${stageId}`) || null;
  }

  clearResults(): void {
    this.results.clear();
  }

  private isCriticalError(error: unknown): boolean {
    // Define what constitutes a critical error that should stop the pipeline
    if (error instanceof Error) {
      return error.message.includes('CRITICAL') ||
             error.message.includes('FATAL') ||
             error.message.includes('UNRECOVERABLE');
    }
    return false;
  }
}

// Pre-built pipeline stages
export class ValidationStage implements PipelineStage {
  constructor(
    public id: string,
    public name: string,
    private validator: (input: unknown) => Promise<boolean>
  ) {}

  async execute(input: unknown): Promise<unknown> {
    const isValid = await this.validator(input);
    if (!isValid) {
      throw new Error(`Validation failed in stage: ${this.name}`);
    }
    return input;
  }
}

export class ProcessingStage implements PipelineStage {
  constructor(
    public id: string,
    public name: string,
    private processor: (input: unknown) => Promise<unknown>
  ) {}

  async execute(input: unknown): Promise<unknown> {
    return await this.processor(input);
  }
}

export class DeploymentStage implements PipelineStage {
  constructor(
    public id: string,
    public name: string,
    private deployer: (input: unknown) => Promise<unknown>
  ) {}

  async execute(input: unknown): Promise<unknown> {
    return await this.deployer(input);
  }
}

// Utility functions for common pipeline operations
export class PipelineUtils {
  static createValidationStage(
    id: string,
    name: string,
    validationFn: (input: unknown) => boolean
  ): ValidationStage {
    return new ValidationStage(id, name, async (input) => validationFn(input));
  }

  static createProcessingStage(
    id: string,
    name: string,
    processFn: (input: unknown) => unknown
  ): ProcessingStage {
    return new ProcessingStage(id, name, async (input) => processFn(input));
  }

  static createDeploymentStage(
    id: string,
    name: string,
    deployFn: (input: unknown) => unknown
  ): DeploymentStage {
    return new DeploymentStage(id, name, async (input) => deployFn(input));
  }
}