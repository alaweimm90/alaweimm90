import type { Crew } from '../types';
import { validateCrew } from '../crews';

describe('Crews', () => {
  describe('validateCrew', () => {
    it('should validate a valid crew', () => {
      const crew: Crew = {
        name: 'test_crew',
        description: 'Test crew',
        version: '1.0',
        agents: [
          {
            name: 'agent1',
            agent_ref: 'coder_agent',
            role_in_crew: 'Developer',
            responsibilities: ['Write code']
          }
        ],
        tasks: [
          {
            name: 'task1',
            description: 'Test task',
            assigned_to: 'agent1',
            expected_output: 'Output',
            priority: 1
          }
        ],
        process: {
          type: 'sequential',
          workflow: [
            { phase: 'main', tasks: ['task1'], parallel: false }
          ]
        }
      };

      const result = validateCrew(crew);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should catch missing name', () => {
      const crew = {
        description: 'Test',
        version: '1.0',
        agents: [],
        tasks: [],
        process: { type: 'sequential', workflow: [] }
      } as unknown as Crew;

      const result = validateCrew(crew);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Crew missing name');
    });

    it('should catch missing agents', () => {
      const crew: Crew = {
        name: 'test',
        description: 'Test',
        version: '1.0',
        agents: [],
        tasks: [{ name: 't1', description: 'd', assigned_to: 'a1', expected_output: 'o', priority: 1 }],
        process: { type: 'sequential', workflow: [] }
      };

      const result = validateCrew(crew);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Crew has no agents');
    });

    it('should catch missing tasks', () => {
      const crew: Crew = {
        name: 'test',
        description: 'Test',
        version: '1.0',
        agents: [{ name: 'a1', agent_ref: 'ref', role_in_crew: 'role', responsibilities: [] }],
        tasks: [],
        process: { type: 'sequential', workflow: [] }
      };

      const result = validateCrew(crew);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Crew has no tasks');
    });

    it('should catch invalid task assignment', () => {
      const crew: Crew = {
        name: 'test',
        description: 'Test',
        version: '1.0',
        agents: [{ name: 'agent1', agent_ref: 'ref', role_in_crew: 'role', responsibilities: [] }],
        tasks: [{ name: 't1', description: 'd', assigned_to: 'unknown_agent', expected_output: 'o', priority: 1 }],
        process: { type: 'sequential', workflow: [] }
      };

      const result = validateCrew(crew);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('unknown agent'))).toBe(true);
    });

    it('should catch invalid task dependencies', () => {
      const crew: Crew = {
        name: 'test',
        description: 'Test',
        version: '1.0',
        agents: [{ name: 'agent1', agent_ref: 'ref', role_in_crew: 'role', responsibilities: [] }],
        tasks: [
          { name: 't1', description: 'd', assigned_to: 'agent1', expected_output: 'o', priority: 1, depends_on: ['unknown_task'] }
        ],
        process: { type: 'sequential', workflow: [] }
      };

      const result = validateCrew(crew);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('unknown task'))).toBe(true);
    });
  });
});
