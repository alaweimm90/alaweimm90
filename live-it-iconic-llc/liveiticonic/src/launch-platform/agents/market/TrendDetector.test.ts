/**
 * TrendDetector Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { TrendDetectorAgent } from './TrendDetector';
import { AgentStatus, Task, TaskStatus, AgentPriority } from '../../types';

describe('TrendDetectorAgent', () => {
  let agent: TrendDetectorAgent;

  beforeEach(() => {
    agent = new TrendDetectorAgent('trend-detector-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Initialization', () => {
    it('should initialize correctly', () => {
      expect(agent.getName()).toBe('Trend Detector');
      expect(agent.getType()).toBe('trend_detector');
      expect(agent.getState().status).toBe(AgentStatus.IDLE);
    });

    it('should have trend detection capabilities', () => {
      const capabilities = agent.getCapabilities();
      const capNames = capabilities.map(c => c.name);

      expect(capNames).toContain('detect_trends');
      expect(capNames).toContain('analyze_velocity');
      expect(capNames).toContain('forecast_trends');
      expect(capNames).toContain('segment_trends');
    });
  });

  describe('Trend Detection', () => {
    it('should detect market trends', async () => {
      const task: Task = {
        id: 'trend-task-001',
        name: 'Detect Trends',
        description: 'Identify emerging trends',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 5000,
        deliverables: ['trend_report'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'detect_trends',
        sources: ['social_media', 'news', 'search'],
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.trends).toBeDefined();
      expect(Array.isArray(result.output.trends)).toBe(true);
      expect(result.output.trends.length).toBeGreaterThan(0);

      // Verify trend structure
      const trend = result.output.trends[0];
      expect(trend).toHaveProperty('name');
      expect(trend).toHaveProperty('category');
      expect(trend).toHaveProperty('velocity');
      expect(trend).toHaveProperty('relevance');
    });

    it('should analyze trend velocity', async () => {
      const task: Task = {
        id: 'velocity-task-001',
        name: 'Analyze Velocity',
        description: 'Measure trend speed',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 4000,
        deliverables: ['velocity_analysis'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'analyze_velocity',
        trend: 'electric_vehicles',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.velocity).toBeDefined();
      expect(result.output.acceleration).toBeDefined();
      expect(result.output.momentum).toBeDefined();
    });

    it('should forecast future trends', async () => {
      const task: Task = {
        id: 'forecast-task-001',
        name: 'Forecast Trends',
        description: 'Predict future trends',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['forecast'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'forecast_trends',
        historical: {},
        timeframe: '6_months',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.predictions).toBeDefined();
      expect(result.output.confidence).toBeGreaterThan(0);
      expect(result.output.timeframe).toBe('6_months');
    });
  });
});
