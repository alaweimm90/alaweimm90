import { describe, it, expect } from 'vitest';
import {
  PromptConfig,
  ExecutionContext,
  ExecutionStatus,
  AgentTemplate,
  ValidationRule,
  ValidationSeverity,
  DeploymentTarget,
} from '../types';

describe('Unified TypeScript Types', () => {
  describe('PromptConfig', () => {
    it('should create a valid PromptConfig', () => {
      const config: PromptConfig = {
        name: 'test-prompt',
        version: '1.0.0',
        description: 'Test prompt configuration',
        template: 'Hello {{name}}',
        tags: ['test', 'greeting'],
        parameters: { name: 'World' },
      };

      expect(config.name).toBe('test-prompt');
      expect(config.version).toBe('1.0.0');
      expect(config.template).toBe('Hello {{name}}');
      expect(config.tags).toContain('test');
    });
  });

  describe('ExecutionContext', () => {
    it('should create a valid ExecutionContext', () => {
      const context: ExecutionContext = {
        workflowId: 'test-workflow',
        inputs: { input1: 'value1' },
        outputs: new Map(),
        variables: new Map(),
        checkpoints: [],
        startTime: new Date(),
        status: ExecutionStatus.PENDING,
      };

      expect(context.workflowId).toBe('test-workflow');
      expect(context.status).toBe(ExecutionStatus.PENDING);
      expect(context.inputs.input1).toBe('value1');
    });

    it('should handle execution status enum', () => {
      expect(ExecutionStatus.RUNNING).toBe('running');
      expect(ExecutionStatus.COMPLETED).toBe('completed');
      expect(ExecutionStatus.FAILED).toBe('failed');
    });
  });

  describe('AgentTemplate', () => {
    it('should create a valid AgentTemplate', () => {
      const template: AgentTemplate = {
        role: 'developer',
        goal: 'Write clean code',
        backstory: 'I am a software developer',
        tools: ['code_editor'],
        template_name: 'code-developer-v1',
        tags: ['coding', 'development'],
      };

      expect(template.role).toBe('developer');
      expect(template.goal).toBe('Write clean code');
      expect(template.template_name).toBe('code-developer-v1');
    });
  });

  describe('ValidationRule', () => {
    it('should create a valid ValidationRule', () => {
      const rule: ValidationRule = {
        name: 'required-field-check',
        severity: ValidationSeverity.ERROR,
        condition: 'field !== undefined',
        error_message: 'Field is required',
        suggestion: 'Please provide a value',
        category: 'agent',
        applies_to: ['agent', 'workflow'],
      };

      expect(rule.name).toBe('required-field-check');
      expect(rule.severity).toBe(ValidationSeverity.ERROR);
      expect(rule.error_message).toBe('Field is required');
      expect(rule.applies_to).toContain('agent');
    });
  });

  describe('DeploymentTarget', () => {
    it('should define deployment target constants', () => {
      expect(DeploymentTarget.LOCAL).toBe('local');
      expect(DeploymentTarget.PRODUCTION).toBe('production');
      expect(DeploymentTarget.KUBERNETES).toBe('kubernetes');
      expect(DeploymentTarget.DOCKER).toBe('docker');
    });
  });
});
