import type { Agent, Workflow, Prompt, ValidationResult } from '../types';

describe('Types', () => {
  describe('Agent type', () => {
    it('should define agent structure', () => {
      const agent: Agent = {
        role: 'Test Agent',
        goal: 'Test goal',
        backstory: 'Test backstory',
        tools: ['tool1', 'tool2'],
        llm_config: {
          model: 'claude-3-opus',
          temperature: 0.3
        }
      };

      expect(agent.role).toBe('Test Agent');
      expect(agent.goal).toBe('Test goal');
      expect(agent.tools).toHaveLength(2);
    });

    it('should allow optional fields', () => {
      const minimalAgent: Agent = {
        role: 'Minimal',
        goal: 'Minimal goal',
        backstory: 'Minimal backstory'
      };

      expect(minimalAgent.tools).toBeUndefined();
      expect(minimalAgent.llm_config).toBeUndefined();
    });
  });

  describe('Workflow type', () => {
    it('should define workflow structure', () => {
      const workflow: Workflow = {
        name: 'Test Workflow',
        description: 'Test description',
        pattern: 'prompt_chaining',
        stages: [
          { name: 'stage1', action: 'Do something' },
          { name: 'stage2', action: 'Do something else' }
        ]
      };

      expect(workflow.name).toBe('Test Workflow');
      expect(workflow.pattern).toBe('prompt_chaining');
      expect(workflow.stages).toHaveLength(2);
    });
  });

  describe('Prompt type', () => {
    it('should define prompt structure', () => {
      const prompt: Prompt = {
        path: '/path/to/prompt.md',
        name: 'test-prompt',
        category: 'system',
        size: 1024
      };

      expect(prompt.category).toBe('system');
      expect(prompt.size).toBe(1024);
    });
  });

  describe('ValidationResult type', () => {
    it('should define validation result structure', () => {
      const result: ValidationResult = {
        valid: true,
        errors: [],
        warnings: [{ type: 'warning', message: 'Test warning' }]
      };

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
      expect(result.warnings).toHaveLength(1);
    });
  });
});
