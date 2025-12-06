import { describe, it, expect } from 'vitest';
import { unifiedExecutor } from '../core/AutomationCore';
import { ExecutionStatus } from '../types';

// Integration tests for CLI functionality
describe('CLI Integration Tests', () => {
  describe('Unified Executor', () => {
    it('should handle direct workflow execution', async () => {
      const result = await unifiedExecutor('default', { dryRun: true });

      expect(result.workflowId).toBe('default');
      expect(result.status).toBe(ExecutionStatus.COMPLETED);
      expect(result.startTime).toBeDefined();
      expect(result.endTime).toBeDefined();
    });

    it('should handle natural language task routing', async () => {
      const result = await unifiedExecutor('debug the code', { dryRun: true });

      expect(result.workflowId).toBe('default'); // Routes to default workflow
      expect(result.status).toBe(ExecutionStatus.COMPLETED);
      expect(result.inputs).toHaveProperty('task');
      expect(result.inputs.task).toBe('debug the code');
    });

    it('should pass inputs correctly to execution', async () => {
      const customInputs = { customParam: 'test-value', task: 'run task' };
      const result = await unifiedExecutor('default', { dryRun: true }, customInputs);

      expect(result.inputs.customParam).toBe('test-value');
      expect(result.inputs.task).toBe('run task');
      expect(result.workflowId).toBe('default');
      expect(result.status).toBe(ExecutionStatus.COMPLETED);
    });

    it('should reject tasks with low confidence', async () => {
      // Very generic task should trigger low confidence rejection
      await expect(unifiedExecutor('do something', { dryRun: true })).rejects.toThrow(
        /Low confidence routing/
      );
    });

    it('should support dry run mode', async () => {
      const result = await unifiedExecutor('default', { dryRun: true }, { testMode: true });

      expect(result.status).toBe(ExecutionStatus.COMPLETED);
      expect(result.outputs instanceof Map).toBe(true);
      expect(result.endTime).toBeDefined();
      expect(result.endTime!.getTime()).toBeGreaterThanOrEqual(result.startTime.getTime());
    });
  });

  describe('Execution Context Management', () => {
    it('should create proper execution context structure', async () => {
      const inputs = { task: 'test', priority: 'high' };
      const result = await unifiedExecutor('default', { dryRun: true }, inputs);

      expect(result).toMatchObject({
        workflowId: 'default',
        status: ExecutionStatus.COMPLETED,
        inputs,
      });

      // Verify Map structures exist
      expect(result.outputs instanceof Map).toBe(true);
      expect(result.variables instanceof Map).toBe(true);
      expect(Array.isArray(result.checkpoints)).toBe(true);

      // Verify timestamps
      expect(result.startTime instanceof Date).toBe(true);
      expect(result.endTime instanceof Date).toBe(true);
    });

    it('should maintain execution isolation', async () => {
      const result1 = await unifiedExecutor('default', { dryRun: true }, { task: 'first' });
      const result2 = await unifiedExecutor('default', { dryRun: true }, { task: 'second' });

      expect(result1.inputs.task).toBe('first');
      expect(result2.inputs.task).toBe('second');
      expect(result1.workflowId).toBe(result2.workflowId);

      // Executions should be independent
      expect(result1.startTime.getTime()).not.toBe(result2.startTime.getTime());
    });
  });

  describe('Error Handling', () => {
    it('should handle non-existent workflow gracefully', async () => {
      await expect(unifiedExecutor('non-existent-workflow', { dryRun: true })).rejects.toThrow(
        /not found/
      );
    });

    it('should include error details in failed execution context', async () => {
      try {
        await unifiedExecutor('non-existent-workflow', { dryRun: true });
        expect.fail('Should have thrown');
      } catch (error) {
        expect((error as Error).message).toContain('not found');
      }
    });
  });

  describe('Route Task Integration', () => {
    it('should integrate routing results with execution inputs', async () => {
      const result = await unifiedExecutor('write unit tests for the code', { dryRun: false });

      expect(result.inputs).toHaveProperty('task');
      expect(result.inputs.task).toBe('write unit tests for the code');
      expect(result.inputs).toHaveProperty('tools');
      expect(result.inputs).toHaveProperty('agents');
      expect(Array.isArray(result.inputs.tools)).toBe(true);
      expect(Array.isArray(result.inputs.agents)).toBe(true);
    });

    it('should pass routing-derived tools and agents to workflow', async () => {
      const result = await unifiedExecutor('write documentation', { dryRun: true });

      expect(result.inputs.tools).toContain('claude_code');
      expect(result.inputs.agents).toContain('technical_writer_agent');
    });
  });
});
