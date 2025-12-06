/**
 * LiveItIconic Launch Platform - Launch Orchestrator
 *
 * Central coordinator that manages all 26+ agents and orchestrates product launches
 */

import {
  Product,
  LaunchPlan,
  LaunchStrategy,
  LaunchResult,
  ResourceAllocation,
  LaunchTimeline,
  Task,
  AgentType,
  AgentPriority,
  LaunchStatus,
  LaunchMetrics,
  Risk,
  EthicalConstraints,
  HumanOversight,
  ExecutionContext,
} from '../types';
import { BaseAgent } from './BaseAgent';
import { eventBus } from './EventBus';
import { stateManager } from './StateManager';

export class LaunchOrchestrator {
  private agents: Map<string, BaseAgent> = new Map();
  private activeLaunches: Map<string, LaunchPlan> = new Map();
  private constraints: EthicalConstraints;
  private oversight: HumanOversight;

  constructor(constraints?: EthicalConstraints, oversight?: HumanOversight) {
    this.constraints = constraints || this.getDefaultConstraints();
    this.oversight = oversight || this.getDefaultOversight();

    console.log('[LaunchOrchestrator] Initializing...');
  }

  /**
   * Register an agent with the orchestrator
   */
  registerAgent(agent: BaseAgent): void {
    this.agents.set(agent.getId(), agent);
    console.log(`[LaunchOrchestrator] Registered agent: ${agent.getName()}`);
  }

  /**
   * Get all registered agents
   */
  getAgents(): BaseAgent[] {
    return Array.from(this.agents.values());
  }

  /**
   * Get agent by ID
   */
  getAgent(agentId: string): BaseAgent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Get agents by type
   */
  getAgentsByType(type: AgentType): BaseAgent[] {
    return this.getAgents().filter(a => a.getType() === type);
  }

  /**
   * Plan a product launch
   */
  async planLaunch(product: Product): Promise<LaunchPlan> {
    console.log(`[LaunchOrchestrator] Planning launch for: ${product.name}`);

    // Step 1: Market Analysis
    const marketData = await this.analyzeMarket(product);

    // Step 2: Strategy Development
    const strategy = await this.developStrategy(product, marketData);

    // Step 3: Resource Allocation
    const resources = await this.allocateResources(strategy);

    // Step 4: Timeline Creation
    const timeline = await this.createTimeline(strategy, resources);

    // Step 5: Risk Assessment
    const risks = await this.assessRisks(product, strategy, marketData);

    // Step 6: Success Prediction
    const prediction = await this.predictSuccess({
      product,
      strategy,
      resources,
      timeline,
      risks,
    });

    // Create launch plan
    const launchPlan: LaunchPlan = {
      id: `launch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      product,
      strategy,
      resources,
      timeline,
      metrics: {
        targets: {
          revenue: { name: 'Revenue', target: 100000, unit: 'USD' },
          users: { name: 'Users', target: 1000, unit: 'users' },
          conversions: { name: 'Conversions', target: 50, unit: 'conversions' },
        },
        actual: {},
        timestamp: new Date(),
      },
      risks,
      status: LaunchStatus.PLANNING,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    // Check if human approval is required
    if (this.requiresHumanApproval(launchPlan, prediction)) {
      console.log('[LaunchOrchestrator] Launch plan requires human approval');
      launchPlan.status = LaunchStatus.PLANNING;
    } else {
      launchPlan.status = LaunchStatus.APPROVED;
    }

    // Store in state manager
    stateManager.createLaunch(launchPlan);

    console.log(`[LaunchOrchestrator] Launch plan created: ${launchPlan.id}`);
    return launchPlan;
  }

  /**
   * Execute a launch plan
   */
  async executeLaunch(launchPlanId: string): Promise<LaunchResult> {
    console.log(`[LaunchOrchestrator] Executing launch: ${launchPlanId}`);

    const launchState = stateManager.getLaunch(launchPlanId);
    if (!launchState) {
      throw new Error(`Launch plan not found: ${launchPlanId}`);
    }

    const plan = launchState.plan;

    // Check approval status
    if (plan.status !== LaunchStatus.APPROVED) {
      throw new Error(`Launch plan not approved: ${launchPlanId}`);
    }

    // Update status
    plan.status = LaunchStatus.IN_PROGRESS;
    stateManager.updateLaunchPlan(launchPlanId, plan);

    try {
      // Decompose plan into tasks
      const tasks = await this.decomposePlan(plan);

      // Execute tasks in parallel where possible
      const results = await this.executeParallel(tasks);

      // Collect metrics
      const metrics = await this.collectMetrics(launchPlanId);

      // Generate insights
      const insights = await this.generateInsights(launchPlanId, results);

      // Generate next steps
      const nextSteps = await this.generateNextSteps(launchPlanId, metrics, insights);

      // Create result
      const result: LaunchResult = {
        launchId: launchPlanId,
        success: results.every(r => r.success),
        metrics,
        outcomes: results.map(r => ({
          metric: r.output?.metric || 'unknown',
          target: r.output?.target || 0,
          actual: r.output?.actual || 0,
          variance: (r.output?.actual || 0) - (r.output?.target || 0),
          success: r.success,
        })),
        insights,
        nextSteps,
        timestamp: new Date(),
      };

      // Update status
      plan.status = LaunchStatus.COMPLETED;
      stateManager.updateLaunchPlan(launchPlanId, plan);

      console.log(`[LaunchOrchestrator] Launch completed: ${launchPlanId}`);
      return result;
    } catch (error) {
      // Update status to failed
      plan.status = LaunchStatus.FAILED;
      stateManager.updateLaunchPlan(launchPlanId, plan);

      console.error(`[LaunchOrchestrator] Launch failed: ${launchPlanId}`, error);
      throw error;
    }
  }

  /**
   * Analyze market using Market Intelligence agents
   */
  private async analyzeMarket(product: Product): Promise<unknown> {
    console.log('[LaunchOrchestrator] Analyzing market...');

    // Use multiple market intelligence agents in parallel
    const analyses = await Promise.all([
      this.runAgentTask(AgentType.COMPETITOR_ANALYST, {
        product,
        action: 'analyze_competitors',
      }),
      this.runAgentTask(AgentType.TREND_DETECTOR, {
        product,
        action: 'detect_trends',
      }),
      this.runAgentTask(AgentType.CUSTOMER_RESEARCHER, {
        product,
        action: 'research_customers',
      }),
      this.runAgentTask(AgentType.MARKET_SIZER, {
        product,
        action: 'size_market',
      }),
    ]);

    return {
      competitors: analyses[0],
      trends: analyses[1],
      customers: analyses[2],
      marketSize: analyses[3],
    };
  }

  /**
   * Develop launch strategy
   */
  private async developStrategy(product: Product, marketData: Record<string, unknown>): Promise<LaunchStrategy> {
    console.log('[LaunchOrchestrator] Developing launch strategy...');

    // Use strategic agents
    const [messaging, pricing, positioning] = await Promise.all([
      this.runAgentTask(AgentType.STORYTELLER, {
        product,
        marketData,
        action: 'develop_messaging',
      }),
      this.runAgentTask(AgentType.PRICING_STRATEGIST, {
        product,
        marketData,
        action: 'optimize_pricing',
      }),
      this.runAgentTask(AgentType.BRAND_ARCHITECT, {
        product,
        marketData,
        action: 'create_positioning',
      }),
    ]);

    return {
      type: 'phased',
      channels: [],
      messaging,
      positioning,
      budget: {
        total: 50000,
        allocated: {},
        spent: 0,
        remaining: 50000,
        breakdown: [],
      },
    };
  }

  /**
   * Allocate resources
   */
  private async allocateResources(strategy: LaunchStrategy): Promise<ResourceAllocation> {
    console.log('[LaunchOrchestrator] Allocating resources...');

    return {
      agents: [],
      tools: [],
      budget: strategy.budget,
      timeline: 30, // 30 days
    };
  }

  /**
   * Create timeline
   */
  private async createTimeline(
    strategy: LaunchStrategy,
    resources: ResourceAllocation
  ): Promise<LaunchTimeline> {
    console.log('[LaunchOrchestrator] Creating timeline...');

    const now = new Date();
    const launchDate = new Date(now.getTime() + 14 * 24 * 60 * 60 * 1000); // 14 days from now
    const endDate = new Date(launchDate.getTime() + 30 * 24 * 60 * 60 * 1000); // 30 days after launch

    return {
      startDate: now,
      launchDate,
      endDate,
      phases: [],
      milestones: [],
    };
  }

  /**
   * Assess risks
   */
  private async assessRisks(
    product: Product,
    strategy: LaunchStrategy,
    marketData: Record<string, unknown>
  ): Promise<Risk[]> {
    console.log('[LaunchOrchestrator] Assessing risks...');

    const riskAssessment = await this.runAgentTask(AgentType.RISK_ASSESSOR, {
      product,
      strategy,
      marketData,
      action: 'assess_risks',
    });

    return riskAssessment || [];
  }

  /**
   * Predict success probability
   */
  private async predictSuccess(context: Record<string, unknown>): Promise<unknown> {
    console.log('[LaunchOrchestrator] Predicting success...');

    // This would use the LearningEngine in a real implementation
    return {
      probability: 0.75,
      confidence: 0.8,
      factors: [],
      risks: context.risks,
      recommendations: [],
    };
  }

  /**
   * Decompose plan into executable tasks
   */
  private async decomposePlan(plan: LaunchPlan): Promise<Task[]> {
    console.log('[LaunchOrchestrator] Decomposing plan into tasks...');

    const tasks: Task[] = [];

    // Generate tasks from timeline phases
    for (const phase of plan.timeline.phases) {
      tasks.push(...phase.tasks);
    }

    return tasks;
  }

  /**
   * Execute tasks in parallel
   */
  private async executeParallel(tasks: Task[]): Promise<unknown[]> {
    console.log(`[LaunchOrchestrator] Executing ${tasks.length} tasks...`);

    const results = await Promise.allSettled(
      tasks.map(task => this.executeTask(task))
    );

    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        console.error(`Task ${tasks[index].name} failed:`, result.reason);
        return {
          success: false,
          error: result.reason,
          output: null,
          metrics: {},
          duration: 0,
        };
      }
    });
  }

  /**
   * Execute a single task
   */
  private async executeTask(task: Task): Promise<unknown> {
    // Find appropriate agent for the task
    const agent = this.agents.get(task.assignedTo);
    if (!agent) {
      throw new Error(`Agent not found for task: ${task.assignedTo}`);
    }

    return await agent.executeTask(task);
  }

  /**
   * Run a task on a specific agent type
   */
  private async runAgentTask(agentType: AgentType, params: Record<string, unknown>): Promise<unknown> {
    const agents = this.getAgentsByType(agentType);
    if (agents.length === 0) {
      console.warn(`No agents found for type: ${agentType}, returning mock data`);
      return {};
    }

    // Use the first available agent of this type
    const agent = agents[0];

    // Create a simple task
    const task: Task = {
      id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: `${agentType}_task`,
      description: `Execute ${params.action} for ${agentType}`,
      assignedTo: agent.getId(),
      status: 'pending',
      priority: AgentPriority.NORMAL,
      dependencies: [],
      estimatedDuration: 5000,
      deliverables: [],
      createdAt: new Date(),
      dueDate: new Date(Date.now() + 60000),
    };

    const result = await agent.executeTask(task);
    return result.output;
  }

  /**
   * Collect metrics for a launch
   */
  private async collectMetrics(launchId: string): Promise<LaunchMetrics> {
    console.log('[LaunchOrchestrator] Collecting metrics...');

    const analyticsAgent = this.getAgentsByType(AgentType.ANALYTICS_INTERPRETER)[0];
    if (!analyticsAgent) {
      return {
        targets: {},
        actual: {},
        timestamp: new Date(),
      };
    }

    const metrics = await this.runAgentTask(AgentType.ANALYTICS_INTERPRETER, {
      launchId,
      action: 'collect_metrics',
    });

    return metrics || { targets: {}, actual: {}, timestamp: new Date() };
  }

  /**
   * Generate insights
   */
  private async generateInsights(launchId: string, results: unknown[]): Promise<unknown[]> {
    console.log('[LaunchOrchestrator] Generating insights...');

    // Use analytics and feedback agents
    const insights = await this.runAgentTask(AgentType.FEEDBACK_ANALYZER, {
      launchId,
      results,
      action: 'analyze_results',
    });

    return insights || [];
  }

  /**
   * Generate next steps
   */
  private async generateNextSteps(launchId: string, metrics: Record<string, unknown>, insights: unknown[]): Promise<unknown[]> {
    console.log('[LaunchOrchestrator] Generating next steps...');

    // This would use the optimization agents in a real implementation
    return [];
  }

  /**
   * Check if launch requires human approval
   */
  private requiresHumanApproval(plan: LaunchPlan, prediction: Record<string, unknown>): boolean {
    if (!this.oversight.approvalRequired.campaignLaunch) {
      return false;
    }

    // Check thresholds
    if (plan.resources.budget.total >= this.oversight.reviewTriggers.spendThreshold) {
      return true;
    }

    if (prediction.confidence < this.oversight.reviewTriggers.confidenceThreshold) {
      return true;
    }

    const riskScore = plan.risks.reduce((sum, r) => sum + r.probability, 0) / plan.risks.length;
    if (riskScore >= this.oversight.reviewTriggers.riskScore) {
      return true;
    }

    return false;
  }

  /**
   * Pause all agents
   */
  pauseAll(): void {
    console.log('[LaunchOrchestrator] Pausing all agents...');
    this.getAgents().forEach(agent => agent.pause());
  }

  /**
   * Resume all agents
   */
  resumeAll(): void {
    console.log('[LaunchOrchestrator] Resuming all agents...');
    this.getAgents().forEach(agent => agent.resume());
  }

  /**
   * Shutdown orchestrator
   */
  shutdown(): void {
    console.log('[LaunchOrchestrator] Shutting down...');
    this.getAgents().forEach(agent => agent.shutdown());
    this.agents.clear();
    this.activeLaunches.clear();
  }

  /**
   * Get default ethical constraints
   */
  private getDefaultConstraints(): EthicalConstraints {
    return {
      truthfulness: {
        noFalseAdvertising: true,
        accurateClaimsOnly: true,
        transparentPricing: true,
      },
      fairness: {
        noBiasAmplification: true,
        equalOpportunity: true,
        inclusiveMessaging: true,
      },
      privacy: {
        gdprCompliant: true,
        ccpaCompliant: true,
        userConsentRequired: true,
      },
      safety: {
        noBrandDamage: true,
        noReputationalRisk: true,
        noLegalViolations: true,
      },
    };
  }

  /**
   * Get default human oversight configuration
   */
  private getDefaultOversight(): HumanOversight {
    return {
      approvalRequired: {
        campaignLaunch: true,
        majorSpend: true,
        brandChanges: true,
        controversialContent: true,
      },
      reviewTriggers: {
        confidenceThreshold: 0.8,
        spendThreshold: 10000,
        riskScore: 0.7,
      },
      killSwitches: {
        emergencyStop: true,
        rollback: true,
        pauseCampaign: true,
      },
    };
  }

  /**
   * Emergency stop
   */
  emergencyStop(reason: string): void {
    console.error(`[LaunchOrchestrator] EMERGENCY STOP: ${reason}`);
    this.pauseAll();

    // Broadcast emergency stop to all agents
    eventBus.broadcast('emergency-stop', { reason }, 'orchestrator');
  }

  /**
   * Get orchestrator statistics
   */
  getStatistics(): Record<string, unknown> {
    const agentStats = this.getAgents().map(agent => ({
      id: agent.getId(),
      name: agent.getName(),
      type: agent.getType(),
      state: agent.getState(),
    }));

    return {
      totalAgents: this.agents.size,
      agentsByType: agentStats,
      ...stateManager.getStatistics(),
      ...eventBus.getStatistics(),
    };
  }
}
