/**
 * LiveItIconic Launch Platform - Basic Agent Usage Examples
 *
 * Simple examples showing how to use individual agents
 */

import { AgentFactory } from '../agents';
import {
  AgentType,
  Task,
  TaskStatus,
  AgentPriority,
} from '../types';

/**
 * Example 1: Using a Single Agent
 */
async function exampleSingleAgent(): Promise<void> {
  console.log('\n📝 Example 1: Using a Single Agent\n');

  // Create a CompetitorAnalyst agent
  const agent = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST);

  if (!agent) {
    console.error('Failed to create agent');
    return;
  }

  console.log(`Created agent: ${agent.getName()}`);
  console.log(`Agent ID: ${agent.getId()}`);
  console.log(`Agent Type: ${agent.getType()}\n`);

  // View agent capabilities
  const capabilities = agent.getCapabilities();
  console.log(`Agent has ${capabilities.length} capabilities:`);
  capabilities.forEach(cap => {
    console.log(`  • ${cap.name}: ${cap.description}`);
  });
  console.log();

  // Create a task for the agent
  const task: Task = {
    id: 'task-001',
    name: 'Analyze Competitors',
    description: 'Analyze luxury automotive merchandise competitors',
    assignedTo: agent.getId(),
    status: TaskStatus.PENDING,
    priority: AgentPriority.HIGH,
    dependencies: [],
    estimatedDuration: 5000,
    deliverables: ['competitor_analysis'],
    createdAt: new Date(),
    dueDate: new Date(Date.now() + 86400000),
    action: 'analyze_competitors',
    market: 'luxury_automotive',
  };

  console.log('Executing task...');
  try {
    const result = await agent.executeTask(task);
    console.log(`✓ Task completed in ${result.duration}ms`);
    console.log(`  Success: ${result.success}`);
    console.log(`  Output:`, result.output);
  } catch (error) {
    console.error('Task failed:', error);
  }

  // Check agent state
  const state = agent.getState();
  console.log(`\nAgent State:`);
  console.log(`  Status: ${state.status}`);
  console.log(`  Tasks Completed: ${state.metrics.tasksCompleted}`);
  console.log(`  Success Rate: ${(state.metrics.successRate * 100).toFixed(1)}%`);

  // Cleanup
  agent.shutdown();
  console.log('\n✓ Agent shutdown\n');
}

/**
 * Example 2: Using Multiple Agents
 */
async function exampleMultipleAgents(): Promise<void> {
  console.log('\n📝 Example 2: Using Multiple Agents\n');

  // Create all Market Intelligence agents
  const marketAgents = AgentFactory.createAgentsForPhase('analysis');

  console.log(`Created ${marketAgents.length} Market Intelligence agents:\n`);

  marketAgents.forEach(agent => {
    console.log(`  • ${agent.getName()} (${agent.getType()})`);
  });

  console.log('\nEach agent can work independently or collaboratively.\n');

  // Cleanup
  marketAgents.forEach(agent => agent.shutdown());
  console.log('✓ All agents shutdown\n');
}

/**
 * Example 3: Agent Factory Patterns
 */
function exampleAgentFactory(): void {
  console.log('\n📝 Example 3: Agent Factory Patterns\n');

  // Get agent statistics
  const stats = AgentFactory.getImplementedAgentsCount();
  console.log('Agent Statistics:');
  console.log(`  Total Implemented: ${stats.total}`);
  console.log('  By Category:');
  Object.entries(stats.byCategory).forEach(([category, count]) => {
    console.log(`    • ${category}: ${count}`);
  });
  console.log();

  // Check if specific agent types are implemented
  console.log('Checking Agent Implementation:');
  const typesToCheck = [
    AgentType.COMPETITOR_ANALYST,
    AgentType.BRAND_ARCHITECT,
    AgentType.CAMPAIGN_MANAGER,
    AgentType.ANALYTICS_INTERPRETER,
    AgentType.DATA_COLLECTOR,
  ];

  typesToCheck.forEach(type => {
    const implemented = AgentFactory.isImplemented(type);
    console.log(`  • ${type}: ${implemented ? '✓ Implemented' : '✗ Not implemented'}`);
  });
  console.log();

  // Get all implemented types
  const allTypes = AgentFactory.getImplementedTypes();
  console.log(`Total agent types available: ${allTypes.length}\n`);
}

/**
 * Example 4: Agent Lifecycle Management
 */
async function exampleAgentLifecycle(): Promise<void> {
  console.log('\n📝 Example 4: Agent Lifecycle Management\n');

  const agent = AgentFactory.createAgent(AgentType.TREND_DETECTOR);
  if (!agent) return;

  console.log('Agent States:');

  // Initial state
  console.log(`  Initial: ${agent.getState().status}`);

  // Pause agent
  agent.pause();
  console.log(`  After pause: ${agent.getState().status}`);

  // Resume agent
  agent.resume();
  console.log(`  After resume: ${agent.getState().status}`);

  // Shutdown agent
  agent.shutdown();
  console.log(`  After shutdown: ${agent.getState().status}\n`);
}

/**
 * Example 5: Creating Custom Agent Instances
 */
function exampleCustomInstances(): void {
  console.log('\n📝 Example 5: Creating Custom Agent Instances\n');

  // Create agents with custom IDs
  const agent1 = AgentFactory.createAgent(AgentType.COPYWRITER, 'copywriter-web-001');
  const agent2 = AgentFactory.createAgent(AgentType.COPYWRITER, 'copywriter-email-001');
  const agent3 = AgentFactory.createAgent(AgentType.COPYWRITER, 'copywriter-social-001');

  if (agent1 && agent2 && agent3) {
    console.log('Created specialized copywriter instances:');
    console.log(`  • ${agent1.getId()} - For web content`);
    console.log(`  • ${agent2.getId()} - For email campaigns`);
    console.log(`  • ${agent3.getId()} - For social media`);
    console.log('\nEach instance can be configured independently\n');

    // Cleanup
    [agent1, agent2, agent3].forEach(agent => agent.shutdown());
  }
}

/**
 * Run all examples
 */
async function runAllExamples(): Promise<void> {
  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║  LiveItIconic Launch Platform - Basic Usage Examples      ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  try {
    await exampleSingleAgent();
    await exampleMultipleAgents();
    exampleAgentFactory();
    await exampleAgentLifecycle();
    exampleCustomInstances();

    console.log('═══════════════════════════════════════════════════════════\n');
    console.log('✅ All examples completed successfully!\n');
    console.log('Next steps:');
    console.log('  1. Run the full demo: ts-node src/launch-platform/examples/demo-launch.ts');
    console.log('  2. Explore agent capabilities in your code');
    console.log('  3. Build custom launch workflows\n');
    console.log('═══════════════════════════════════════════════════════════\n');
  } catch (error) {
    console.error('Example error:', error);
  }
}

/**
 * Run if called directly
 */
if (require.main === module) {
  runAllExamples()
    .then(() => process.exit(0))
    .catch(error => {
      console.error('Examples failed:', error);
      process.exit(1);
    });
}

export {
  exampleSingleAgent,
  exampleMultipleAgents,
  exampleAgentFactory,
  exampleAgentLifecycle,
  exampleCustomInstances,
  runAllExamples,
};
