/**
 * LiveItIconic Launch Platform - State Manager
 *
 * Centralized state management for launch plans, agent status, and execution context
 */

import {
  LaunchPlan,
  AgentStatus,
  AgentType,
  Task,
  LaunchMetrics,
  Experience,
  Pattern,
  LaunchStatus,
} from '../types';
import { eventBus } from './EventBus';

export interface AgentState {
  id: string;
  type: AgentType;
  status: AgentStatus;
  currentTask?: Task;
  metrics: {
    tasksCompleted: number;
    tasksInProgress: number;
    tasksFailed: number;
    averageDuration: number;
    successRate: number;
  };
  lastActivity: Date;
}

export interface LaunchState {
  plan: LaunchPlan;
  agents: Map<string, AgentState>;
  progress: {
    tasksCompleted: number;
    tasksInProgress: number;
    tasksPending: number;
    tasksFailed: number;
    percentComplete: number;
  };
  currentPhase: string;
  budget: {
    allocated: number;
    spent: number;
    remaining: number;
  };
  metrics: LaunchMetrics;
}

export class StateManager {
  private launches: Map<string, LaunchState> = new Map();
  private agents: Map<string, AgentState> = new Map();
  private experiences: Experience[] = [];
  private patterns: Pattern[] = [];
  private listeners: Map<string, Set<(state: Record<string, unknown>) => void>> = new Map();

  /**
   * Create a new launch
   */
  createLaunch(plan: LaunchPlan): void {
    const launchState: LaunchState = {
      plan,
      agents: new Map(),
      progress: {
        tasksCompleted: 0,
        tasksInProgress: 0,
        tasksPending: this.countTotalTasks(plan),
        tasksFailed: 0,
        percentComplete: 0,
      },
      currentPhase: plan.timeline.phases[0]?.id || '',
      budget: {
        allocated: plan.resources.budget.total,
        spent: plan.resources.budget.spent,
        remaining: plan.resources.budget.remaining,
      },
      metrics: plan.metrics,
    };

    this.launches.set(plan.id, launchState);
    this.notifyListeners(`launch:${plan.id}`, launchState);
    this.notifyListeners('launches', this.getAllLaunches());

    console.log(`[StateManager] Created launch: ${plan.id}`);
  }

  /**
   * Get launch state
   */
  getLaunch(launchId: string): LaunchState | undefined {
    return this.launches.get(launchId);
  }

  /**
   * Get all launches
   */
  getAllLaunches(): LaunchState[] {
    return Array.from(this.launches.values());
  }

  /**
   * Update launch state
   */
  updateLaunch(launchId: string, updates: Partial<LaunchState>): void {
    const launchState = this.launches.get(launchId);
    if (!launchState) {
      console.error(`[StateManager] Launch not found: ${launchId}`);
      return;
    }

    Object.assign(launchState, updates);
    this.launches.set(launchId, launchState);
    this.notifyListeners(`launch:${launchId}`, launchState);
    this.notifyListeners('launches', this.getAllLaunches());
  }

  /**
   * Update launch plan
   */
  updateLaunchPlan(launchId: string, plan: LaunchPlan): void {
    const launchState = this.launches.get(launchId);
    if (!launchState) {
      console.error(`[StateManager] Launch not found: ${launchId}`);
      return;
    }

    launchState.plan = plan;
    this.launches.set(launchId, launchState);
    this.notifyListeners(`launch:${launchId}`, launchState);
  }

  /**
   * Delete launch
   */
  deleteLaunch(launchId: string): boolean {
    const deleted = this.launches.delete(launchId);
    if (deleted) {
      this.notifyListeners('launches', this.getAllLaunches());
      console.log(`[StateManager] Deleted launch: ${launchId}`);
    }
    return deleted;
  }

  /**
   * Register an agent
   */
  registerAgent(agent: AgentState): void {
    this.agents.set(agent.id, agent);
    this.notifyListeners('agents', this.getAllAgents());
    this.notifyListeners(`agent:${agent.id}`, agent);

    console.log(`[StateManager] Registered agent: ${agent.id} (${agent.type})`);
  }

  /**
   * Get agent state
   */
  getAgent(agentId: string): AgentState | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Get all agents
   */
  getAllAgents(): AgentState[] {
    return Array.from(this.agents.values());
  }

  /**
   * Get agents by type
   */
  getAgentsByType(type: AgentType): AgentState[] {
    return this.getAllAgents().filter(a => a.type === type);
  }

  /**
   * Get agents by status
   */
  getAgentsByStatus(status: AgentStatus): AgentState[] {
    return this.getAllAgents().filter(a => a.status === status);
  }

  /**
   * Update agent state
   */
  updateAgent(agentId: string, updates: Partial<AgentState>): void {
    const agent = this.agents.get(agentId);
    if (!agent) {
      console.error(`[StateManager] Agent not found: ${agentId}`);
      return;
    }

    Object.assign(agent, updates);
    agent.lastActivity = new Date();
    this.agents.set(agentId, agent);
    this.notifyListeners(`agent:${agentId}`, agent);
    this.notifyListeners('agents', this.getAllAgents());
  }

  /**
   * Update agent status
   */
  updateAgentStatus(agentId: string, status: AgentStatus, task?: Task): void {
    this.updateAgent(agentId, { status, currentTask: task });

    // Broadcast status change
    eventBus.broadcast(
      'agent-status',
      { agentId, status, task },
      'state-manager'
    );
  }

  /**
   * Update task status
   */
  updateTaskStatus(launchId: string, taskId: string, updates: Partial<Task>): void {
    const launchState = this.launches.get(launchId);
    if (!launchState) {
      console.error(`[StateManager] Launch not found: ${launchId}`);
      return;
    }

    // Find and update task in timeline phases
    for (const phase of launchState.plan.timeline.phases) {
      const task = phase.tasks.find(t => t.id === taskId);
      if (task) {
        Object.assign(task, updates);
        break;
      }
    }

    // Recalculate progress
    this.recalculateProgress(launchId);
    this.notifyListeners(`launch:${launchId}`, launchState);
  }

  /**
   * Record experience for learning
   */
  recordExperience(experience: Experience): void {
    this.experiences.push(experience);

    // Limit experience history
    if (this.experiences.length > 10000) {
      this.experiences = this.experiences.slice(-5000);
    }

    this.notifyListeners('experiences', this.experiences);
    console.log(`[StateManager] Recorded experience: ${experience.id}`);
  }

  /**
   * Get experiences
   */
  getExperiences(filter?: {
    launchId?: string;
    productType?: string;
    since?: Date;
  }): Experience[] {
    let filtered = [...this.experiences];

    if (filter) {
      if (filter.launchId) {
        filtered = filtered.filter(e => e.launchId === filter.launchId);
      }
      if (filter.productType) {
        filtered = filtered.filter(e => e.productType === filter.productType);
      }
      if (filter.since) {
        filtered = filtered.filter(e => e.timestamp >= filter.since!);
      }
    }

    return filtered;
  }

  /**
   * Add learned pattern
   */
  addPattern(pattern: Pattern): void {
    this.patterns.push(pattern);
    this.notifyListeners('patterns', this.patterns);
    console.log(`[StateManager] Added pattern: ${pattern.id} (${pattern.type})`);
  }

  /**
   * Get patterns
   */
  getPatterns(filter?: {
    type?: 'success' | 'failure' | 'optimization';
    domain?: string;
    minConfidence?: number;
  }): Pattern[] {
    let filtered = [...this.patterns];

    if (filter) {
      if (filter.type) {
        filtered = filtered.filter(p => p.type === filter.type);
      }
      if (filter.domain) {
        filtered = filtered.filter(p => p.domains.includes(filter.domain!));
      }
      if (filter.minConfidence) {
        filtered = filtered.filter(p => p.confidence >= filter.minConfidence!);
      }
    }

    // Sort by confidence and frequency
    filtered.sort((a, b) => {
      const scoreA = a.confidence * a.frequency;
      const scoreB = b.confidence * b.frequency;
      return scoreB - scoreA;
    });

    return filtered;
  }

  /**
   * Subscribe to state changes
   */
  subscribe(key: string, listener: (state: Record<string, unknown>) => void): () => void {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, new Set());
    }

    this.listeners.get(key)!.add(listener);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(key);
      if (listeners) {
        listeners.delete(listener);
        if (listeners.size === 0) {
          this.listeners.delete(key);
        }
      }
    };
  }

  /**
   * Notify listeners of state changes
   */
  private notifyListeners(key: string, state: Record<string, unknown>): void {
    const listeners = this.listeners.get(key);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(state);
        } catch (error) {
          console.error(`[StateManager] Error in listener for ${key}:`, error);
        }
      });
    }
  }

  /**
   * Recalculate launch progress
   */
  private recalculateProgress(launchId: string): void {
    const launchState = this.launches.get(launchId);
    if (!launchState) return;

    let completed = 0;
    let inProgress = 0;
    let pending = 0;
    let failed = 0;

    for (const phase of launchState.plan.timeline.phases) {
      for (const task of phase.tasks) {
        switch (task.status) {
          case 'completed':
            completed++;
            break;
          case 'in_progress':
            inProgress++;
            break;
          case 'pending':
            pending++;
            break;
          case 'failed':
          case 'cancelled':
            failed++;
            break;
        }
      }
    }

    const total = completed + inProgress + pending + failed;
    const percentComplete = total > 0 ? (completed / total) * 100 : 0;

    launchState.progress = {
      tasksCompleted: completed,
      tasksInProgress: inProgress,
      tasksPending: pending,
      tasksFailed: failed,
      percentComplete,
    };

    // Update launch status based on progress
    if (percentComplete === 100) {
      launchState.plan.status = LaunchStatus.COMPLETED;
    } else if (inProgress > 0) {
      launchState.plan.status = LaunchStatus.IN_PROGRESS;
    }
  }

  /**
   * Count total tasks in a launch plan
   */
  private countTotalTasks(plan: LaunchPlan): number {
    return plan.timeline.phases.reduce((sum, phase) => sum + phase.tasks.length, 0);
  }

  /**
   * Get global statistics
   */
  getStatistics(): {
    totalLaunches: number;
    activeLaunches: number;
    completedLaunches: number;
    totalAgents: number;
    activeAgents: number;
    totalExperiences: number;
    totalPatterns: number;
  } {
    const launches = this.getAllLaunches();
    const agents = this.getAllAgents();

    return {
      totalLaunches: launches.length,
      activeLaunches: launches.filter(
        l => l.plan.status === LaunchStatus.IN_PROGRESS || l.plan.status === LaunchStatus.APPROVED
      ).length,
      completedLaunches: launches.filter(l => l.plan.status === LaunchStatus.COMPLETED).length,
      totalAgents: agents.length,
      activeAgents: agents.filter(
        a => a.status === AgentStatus.EXECUTING || a.status === AgentStatus.THINKING
      ).length,
      totalExperiences: this.experiences.length,
      totalPatterns: this.patterns.length,
    };
  }

  /**
   * Clear all state (useful for testing)
   */
  clearAll(): void {
    this.launches.clear();
    this.agents.clear();
    this.experiences = [];
    this.patterns = [];
    this.listeners.clear();
    console.log('[StateManager] All state cleared');
  }

  /**
   * Export state for persistence
   */
  exportState(): {
    launches: LaunchState[];
    agents: AgentState[];
    experiences: Experience[];
    patterns: Pattern[];
  } {
    return {
      launches: this.getAllLaunches(),
      agents: this.getAllAgents(),
      experiences: this.experiences,
      patterns: this.patterns,
    };
  }

  /**
   * Import state from persistence
   */
  importState(state: {
    launches?: LaunchState[];
    agents?: AgentState[];
    experiences?: Experience[];
    patterns?: Pattern[];
  }): void {
    if (state.launches) {
      state.launches.forEach(launch => {
        this.launches.set(launch.plan.id, launch);
      });
    }

    if (state.agents) {
      state.agents.forEach(agent => {
        this.agents.set(agent.id, agent);
      });
    }

    if (state.experiences) {
      this.experiences = state.experiences;
    }

    if (state.patterns) {
      this.patterns = state.patterns;
    }

    console.log('[StateManager] State imported successfully');
  }
}

// Singleton instance
export const stateManager = new StateManager();
