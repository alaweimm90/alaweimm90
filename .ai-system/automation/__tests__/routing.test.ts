import { describe, it, expect } from 'vitest';
import { routeTask } from '../core/utils';

describe('Routing System', () => {
  describe('Pattern Matching', () => {
    it('should route debugging tasks correctly', () => {
      const result = routeTask('debug the error in my code');
      expect(result.task_type).toBe('debugging');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.recommended_tools).toContain('cline');
      expect(result.suggested_agents).toContain('debugger_agent');
    });

    it('should route development tasks correctly', () => {
      const result = routeTask('implement new feature');
      expect(result.task_type).toBe('development');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.recommended_tools).toContain('cursor');
      expect(result.suggested_agents).toContain('coder_agent');
    });

    it('should route refactoring tasks correctly', () => {
      const result = routeTask('clean up and optimize the code');
      expect(result.task_type).toBe('refactoring');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.recommended_tools).toContain('cursor');
    });

    it('should route testing tasks correctly', () => {
      const result = routeTask('write unit tests for the component');
      expect(result.task_type).toBe('testing');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.recommended_tools).toContain('cursor');
      expect(result.suggested_agents).toContain('qa_engineer_agent');
    });

    it('should route documentation tasks correctly', () => {
      const result = routeTask('document the code API with examples');
      expect(result.task_type).toBe('documentation');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.recommended_tools).toContain('claude_code');
      expect(result.suggested_agents).toContain('technical_writer_agent');
    });

    it('should route deployment tasks correctly', () => {
      const result = routeTask('deploy the application to production');
      expect(result.task_type).toBe('devops');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.suggested_agents).toContain('devops_agent');
    });

    it('should route research tasks correctly', () => {
      const result = routeTask('investigate the latest research papers on AI');
      expect(result.task_type).toBe('research');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.recommended_tools).toContain('claude_code');
      expect(result.suggested_agents).toContain('scientist_agent');
    });
  });

  describe('Confidence Scoring', () => {
    it('should assign high confidence for clear matches', () => {
      const result = routeTask('implement a new feature with tests');
      expect(result.confidence).toBeGreaterThan(0.5);
    });

    it('should assign reasonable confidence for generic tasks', () => {
      const result = routeTask('work on the project');
      expect(result.confidence).toBeGreaterThan(0.1);
      expect(result.task_type).toBe('general');
    });

    it('should handle ambiguous tasks gracefully', () => {
      const result = routeTask('do everything');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.confidence).toBeLessThanOrEqual(0.5);
      // Should still provide reasonable defaults
      expect(result.recommended_tools.length).toBeGreaterThan(0);
      expect(result.suggested_agents.length).toBeGreaterThan(0);
    });
  });

  describe('Tool and Agent Recommendations', () => {
    it('should recommend appropriate tools for each task type', () => {
      const debugging = routeTask('fix the bug');
      const development = routeTask('build new feature');
      const testing = routeTask('write tests');

      expect(debugging.recommended_tools).toContain('cline');
      expect(development.recommended_tools).toContain('cursor');
      expect(testing.recommended_tools).toContain('cursor');
    });

    it('should recommend appropriate agents for each task type', () => {
      const dev = routeTask('implement code');
      const qa = routeTask('test functionality');
      const docs = routeTask('write docs');

      expect(dev.suggested_agents).toContain('coder_agent');
      expect(qa.suggested_agents).toContain('qa_engineer_agent');
      expect(docs.suggested_agents).toContain('technical_writer_agent');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty input', () => {
      const result = routeTask('');
      expect(result).toBeDefined();
      expect(result.confidence).toBe(0.3); // Default confidence
      expect(result.task_type).toBe('general');
    });

    it('should handle very short input', () => {
      const result = routeTask('hi');
      expect(result.task_type).toBe('general');
      expect(result.confidence).toBe(0.3);
    });

    it('should be case insensitive', () => {
      const lower = routeTask('debug this code');
      const upper = routeTask('DEBUG THIS CODE');
      const mixed = routeTask('Debug This Code');

      expect(lower.task_type).toBe(upper.task_type);
      expect(upper.task_type).toBe(mixed.task_type);
      expect(lower.task_type).toBe('debugging');
    });

    it('should handle punctuation and special characters', () => {
      const result = routeTask("Fix the bug! It's causing issues.");
      expect(result.task_type).toBe('debugging');
      expect(result.confidence).toBeGreaterThan(0);
    });
  });
});
