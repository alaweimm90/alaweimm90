import { routeTask } from '../index';

describe('Task Routing', () => {
  describe('routeTask', () => {
    it('should route debugging tasks', () => {
      const result = routeTask('debug the authentication error');

      expect(result.task_type).toBe('debugging');
      expect(result.confidence).toBeGreaterThan(0.3);
      expect(result.recommended_tools).toContain('cline');
      expect(result.suggested_agents).toContain('debugger_agent');
    });

    it('should route development tasks', () => {
      const result = routeTask('implement a new feature for user management');

      expect(result.task_type).toBe('development');
      expect(result.recommended_tools).toContain('cursor');
      expect(result.suggested_agents).toContain('coder_agent');
    });

    it('should route refactoring tasks', () => {
      const result = routeTask('refactor the database module');

      expect(result.task_type).toBe('refactoring');
      expect(result.recommended_tools).toContain('kilo_code');
    });

    it('should route review tasks', () => {
      const result = routeTask('review the pull request');

      expect(result.task_type).toBe('review');
      expect(result.suggested_agents).toContain('reviewer_agent');
    });

    it('should route testing tasks', () => {
      const result = routeTask('write tests for the API');

      expect(result.task_type).toBe('testing');
      expect(result.suggested_agents).toContain('qa_engineer_agent');
    });

    it('should route documentation tasks', () => {
      const result = routeTask('document the API endpoints');

      expect(result.task_type).toBe('documentation');
      expect(result.suggested_agents).toContain('writer_agent');
    });

    it('should route devops tasks', () => {
      const result = routeTask('deploy to production');

      expect(result.task_type).toBe('devops');
      expect(result.suggested_agents).toContain('devops_agent');
    });

    it('should route research tasks', () => {
      const result = routeTask('research best practices for caching');

      expect(result.task_type).toBe('research');
      expect(result.suggested_agents).toContain('scientist_agent');
    });

    it('should default to general for unknown tasks', () => {
      const result = routeTask('do something random');

      expect(result.task_type).toBe('general');
      expect(result.confidence).toBeLessThanOrEqual(0.5);
    });
  });
});
