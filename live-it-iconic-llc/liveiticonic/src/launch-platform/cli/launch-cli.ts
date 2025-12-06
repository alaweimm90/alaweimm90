#!/usr/bin/env tsx
/**
 * LiveItIconic Launch Platform - CLI Tool
 *
 * Command-line interface for managing product launches
 */

import { AgentFactory } from '../agents';
import { runProductLaunchDemo } from '../examples/demo-launch';
import { runAllExamples } from '../examples/basic-agent-usage';

interface CLICommand {
  name: string;
  description: string;
  action: () => Promise<void> | void;
}

const commands: CLICommand[] = [
  {
    name: 'agents',
    description: 'List all available agents',
    action: listAgents,
  },
  {
    name: 'demo',
    description: 'Run complete product launch demo',
    action: runDemo,
  },
  {
    name: 'examples',
    description: 'Run basic usage examples',
    action: runExamples,
  },
  {
    name: 'stats',
    description: 'Show platform statistics',
    action: showStats,
  },
  {
    name: 'help',
    description: 'Show this help message',
    action: showHelp,
  },
];

/**
 * List all available agents
 */
function listAgents(): void {
  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║  Available Agents in Launch Platform                      ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  const types = AgentFactory.getImplementedTypes();
  const stats = AgentFactory.getImplementedAgentsCount();

  console.log(`Total Agents: ${stats.total}\n`);

  // Group by category
  const categories = {
    'Market Intelligence': types.filter(t => [
      'competitor_analyst',
      'trend_detector',
      'customer_researcher',
      'pricing_strategist',
      'market_sizer',
    ].includes(t)),
    'Creative & Branding': types.filter(t => [
      'brand_architect',
      'copywriter',
      'visual_designer',
      'video_producer',
      'storyteller',
    ].includes(t)),
    'Launch Execution': types.filter(t => [
      'campaign_manager',
      'social_media_strategist',
      'influencer_outreach',
      'pr_manager',
      'email_marketer',
      'content_distributor',
    ].includes(t)),
    'Optimization': types.filter(t => [
      'analytics_interpreter',
      'conversion_optimizer',
      'seo_specialist',
      'paid_ads_manager',
      'feedback_analyzer',
    ].includes(t)),
    'Supporting': types.filter(t => [
      'data_collector',
      'quality_controller',
      'compliance_checker',
      'budget_manager',
      'risk_assessor',
    ].includes(t)),
  };

  Object.entries(categories).forEach(([category, agentTypes]) => {
    console.log(`${category} (${agentTypes.length} agents):`);
    agentTypes.forEach(type => {
      const agent = AgentFactory.createAgent(type);
      if (agent) {
        const caps = agent.getCapabilities();
        console.log(`  • ${agent.getName()}`);
        console.log(`    Type: ${type}`);
        console.log(`    Capabilities: ${caps.length}`);
        agent.shutdown();
      }
    });
    console.log();
  });
}

/**
 * Run complete demo
 */
async function runDemo(): Promise<void> {
  console.log('\n🚀 Launching Complete Product Launch Demo...\n');
  await runProductLaunchDemo();
}

/**
 * Run basic examples
 */
async function runExamples(): Promise<void> {
  console.log('\n📚 Running Basic Usage Examples...\n');
  await runAllExamples();
}

/**
 * Show platform statistics
 */
function showStats(): void {
  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║  Launch Platform Statistics                               ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  const stats = AgentFactory.getImplementedAgentsCount();
  const types = AgentFactory.getImplementedTypes();

  console.log('📊 Agent Statistics:\n');
  console.log(`  Total Agents: ${stats.total}`);
  console.log('  By Category:');
  Object.entries(stats.byCategory).forEach(([category, count]) => {
    console.log(`    • ${category}: ${count} agents`);
  });
  console.log();

  console.log('🎯 Platform Capabilities:\n');
  console.log('  ✓ Multi-agent orchestration');
  console.log('  ✓ Real-time inter-agent communication');
  console.log('  ✓ Event-driven architecture');
  console.log('  ✓ State management and persistence');
  console.log('  ✓ Meta-learning capabilities');
  console.log('  ✓ Ethical constraints and oversight');
  console.log('  ✓ Comprehensive testing suite');
  console.log();

  console.log('📈 Feature Coverage:\n');
  console.log('  ✓ Market Analysis & Intelligence');
  console.log('  ✓ Brand Strategy & Creative Development');
  console.log('  ✓ Campaign Planning & Execution');
  console.log('  ✓ Performance Optimization');
  console.log('  ✓ Data Collection & Analysis');
  console.log('  ✓ Quality Control & Compliance');
  console.log('  ✓ Budget Management & ROI Tracking');
  console.log('  ✓ Risk Assessment & Mitigation');
  console.log();

  console.log('🔧 Technical Stack:\n');
  console.log('  • TypeScript');
  console.log('  • Event-Driven Architecture');
  console.log('  • Pub/Sub Messaging (EventBus)');
  console.log('  • State Management (StateManager)');
  console.log('  • AI Agent Framework (BaseAgent)');
  console.log('  • Testing: Vitest (71 tests)');
  console.log();
}

/**
 * Show help message
 */
function showHelp(): void {
  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║  LiveItIconic Launch Platform CLI                         ║');
  console.log('║  AI-Powered Product Launch Orchestration System            ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  console.log('Usage: npm run launch [command]\n');
  console.log('Available Commands:\n');

  commands.forEach(cmd => {
    console.log(`  ${cmd.name.padEnd(12)} ${cmd.description}`);
  });

  console.log('\nExamples:\n');
  console.log('  npm run launch agents      # List all agents');
  console.log('  npm run launch demo        # Run full demo');
  console.log('  npm run launch examples    # Run usage examples');
  console.log('  npm run launch stats       # Show statistics\n');

  console.log('Documentation:\n');
  console.log('  • Platform: /src/launch-platform/');
  console.log('  • Agents: /src/launch-platform/agents/');
  console.log('  • Examples: /src/launch-platform/examples/');
  console.log('  • Tests: Run `npm test`\n');
}

/**
 * Main CLI entry point
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const commandName = args[0] || 'help';

  const command = commands.find(cmd => cmd.name === commandName);

  if (!command) {
    console.error(`\n❌ Unknown command: ${commandName}\n`);
    showHelp();
    process.exit(1);
  }

  try {
    await command.action();
    process.exit(0);
  } catch (error) {
    console.error('\n❌ Command failed:', error);
    process.exit(1);
  }
}

// Run CLI if called directly
if (require.main === module) {
  main();
}

export { main, commands };
