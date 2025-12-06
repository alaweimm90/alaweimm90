/**
 * LiveItIconic Launch Platform - Basic Launch Example
 *
 * This example demonstrates how to use the Launch Platform to plan and execute a product launch
 */

import {
  LaunchOrchestrator,
  AgentFactory,
  Product,
  ProductStage,
  stateManager,
  eventBus,
} from '../index';

/**
 * Example: Launch a Premium T-Shirt Product
 */
async function launchPremiumTShirt() {
  console.log('='.repeat(80));
  console.log('LiveItIconic Launch Platform - Demo');
  console.log('='.repeat(80));
  console.log('');

  // Step 1: Define the product
  const product: Product = {
    id: 'product_001',
    name: 'Caribbean Legacy Premium T-Shirt',
    description:
      'Premium Pima cotton t-shirt featuring exclusive Caribbean bird embroidery. Crafted with superior materials and sustainable practices.',
    category: 'Premium Apparel',
    targetMarket: {
      segments: [
        {
          name: 'Luxury Automotive Enthusiasts',
          size: 50000,
          growth: 0.12,
          characteristics: [
            'High income ($100k+)',
            'Age 35-65',
            'Premium car owners',
            'Value craftsmanship',
          ],
          painPoints: [
            'Lack of unique automotive lifestyle brands',
            'Generic automotive merchandise',
            'Desire for sustainable luxury',
          ],
        },
      ],
      primaryPersona: {
        name: 'Alex - The Automotive Connoisseur',
        age: [35, 50],
        income: [100000, 250000],
        occupation: ['Business Owner', 'Executive', 'Professional'],
        psychographics: [
          'Appreciates quality over quantity',
          'Values exclusivity and craftsmanship',
          'Environmentally conscious',
          'Passion for automotive culture',
        ],
        behaviors: [
          'Researches products thoroughly',
          'Willing to pay premium for quality',
          'Active on Instagram and car forums',
          'Attends automotive events',
        ],
        painPoints: [
          'Tired of mass-produced merchandise',
          'Wants to express automotive passion subtly',
          'Difficulty finding sustainable luxury',
        ],
        goals: [
          'Own products that reflect personal values',
          'Support brands with authentic stories',
          'Curate a refined lifestyle',
        ],
        channels: ['Instagram', 'Email', 'Automotive forums', 'Events'],
      },
      secondaryPersonas: [],
      geography: ['United States', 'Canada', 'United Kingdom'],
      tam: 1000000,
      sam: 500000,
      som: 50000,
    },
    pricing: {
      model: 'one-time',
      basePrice: 89,
      currency: 'USD',
      discounts: [
        {
          type: 'early-bird',
          value: 15,
          isPercentage: true,
          conditions: ['First 100 customers'],
          validUntil: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000),
        },
      ],
    },
    features: [
      {
        name: '100% Pima Cotton',
        description: 'Luxury long-staple cotton for superior softness',
        benefit: 'Exceptional comfort that lasts',
        priority: 'must-have',
      },
      {
        name: 'Caribbean Bird Embroidery',
        description: 'Exclusive 3D embroidered design',
        benefit: 'Unique artistic expression',
        priority: 'must-have',
      },
      {
        name: 'Sustainable Production',
        description: 'Eco-friendly manufacturing process',
        benefit: 'Feel good about your purchase',
        priority: 'must-have',
      },
      {
        name: 'Reinforced Seams',
        description: 'Premium construction for durability',
        benefit: 'Long-lasting quality',
        priority: 'must-have',
      },
    ],
    differentiators: [
      'Caribbean heritage-inspired designs',
      'Automotive craftsmanship philosophy',
      'Sustainable luxury positioning',
      'Limited edition exclusivity',
    ],
    stage: ProductStage.READY_TO_LAUNCH,
  };

  // Step 2: Initialize the Launch Orchestrator
  console.log('📋 Initializing Launch Orchestrator...');
  const orchestrator = new LaunchOrchestrator();

  // Step 3: Register all available agents
  console.log('🤖 Registering AI agents...');
  const agents = AgentFactory.createAllAgents();
  agents.forEach(agent => {
    orchestrator.registerAgent(agent);
    console.log(`   ✓ Registered: ${agent.getName()}`);
  });

  console.log(`\n✅ ${agents.length} agents registered and ready\n`);

  // Step 4: Create Launch Plan
  console.log('🎯 Creating launch plan...');
  console.log('   This involves:');
  console.log('   - Market analysis');
  console.log('   - Competitor research');
  console.log('   - Trend detection');
  console.log('   - Strategy development');
  console.log('   - Resource allocation');
  console.log('   - Timeline creation');
  console.log('');

  try {
    const launchPlan = await orchestrator.planLaunch(product);

    console.log('✅ Launch plan created successfully!\n');
    console.log('Launch Plan Summary:');
    console.log('─'.repeat(80));
    console.log(`ID: ${launchPlan.id}`);
    console.log(`Product: ${launchPlan.product.name}`);
    console.log(`Status: ${launchPlan.status}`);
    console.log(`Budget: $${launchPlan.resources.budget.total.toLocaleString()}`);
    console.log(`Timeline: ${launchPlan.resources.timeline} days`);
    console.log(`Risks Identified: ${launchPlan.risks.length}`);
    console.log('─'.repeat(80));
    console.log('');

    // Step 5: Display Market Analysis Results
    const launchState = stateManager.getLaunch(launchPlan.id);
    if (launchState) {
      console.log('📊 Market Analysis Results:');
      console.log('─'.repeat(80));

      // Get agent task history
      const competitorAgent = agents.find(
        a => a.getType() === 'competitor_analyst'
      );
      const trendAgent = agents.find(a => a.getType() === 'trend_detector');

      if (competitorAgent) {
        const history = competitorAgent.getTaskHistory();
        if (history.length > 0) {
          console.log(
            `Competitors Analyzed: ${JSON.stringify(history[0].output?.summary?.totalCompetitors || 'N/A')}`
          );
        }
      }

      if (trendAgent) {
        const history = trendAgent.getTaskHistory();
        if (history.length > 0) {
          console.log(
            `Trends Detected: ${JSON.stringify(history[0].output?.summary?.totalTrends || 'N/A')}`
          );
          console.log(
            `Rising Trends: ${JSON.stringify(history[0].output?.summary?.risingTrends || 'N/A')}`
          );
        }
      }

      console.log('─'.repeat(80));
      console.log('');
    }

    // Step 6: Show Statistics
    console.log('📈 Platform Statistics:');
    console.log('─'.repeat(80));
    const stats = orchestrator.getStatistics();
    console.log(`Total Agents: ${stats.totalAgents}`);
    console.log(`Total Launches: ${stats.totalLaunches}`);
    console.log(`Active Launches: ${stats.activeLaunches}`);
    console.log(`Messages Exchanged: ${stats.totalMessages}`);
    console.log('─'.repeat(80));
    console.log('');

    // Step 7: Simulate Launch Execution (if approved)
    if (launchPlan.status === 'approved') {
      console.log('🚀 Executing launch...');
      console.log(
        '   Note: In production, this would coordinate all agents to execute the plan\n'
      );

      // const result = await orchestrator.executeLaunch(launchPlan.id);
      // console.log('Launch Result:', result);
    } else {
      console.log(
        '⏸️  Launch plan requires human approval before execution\n'
      );
      console.log('   Approval required because:');
      console.log(`   - Budget exceeds threshold: $${launchPlan.resources.budget.total} >= $10,000`);
      console.log('   - Risk assessment pending');
      console.log('');
    }

    // Step 8: Show Agent Activity
    console.log('🤖 Agent Activity Summary:');
    console.log('─'.repeat(80));
    agents.forEach(agent => {
      const state = agent.getState();
      console.log(`${agent.getName()}:`);
      console.log(`   Tasks Completed: ${state.metrics.tasksCompleted}`);
      console.log(
        `   Success Rate: ${(state.metrics.successRate * 100).toFixed(1)}%`
      );
      console.log(
        `   Avg Duration: ${state.metrics.averageDuration.toFixed(0)}ms`
      );
    });
    console.log('─'.repeat(80));
    console.log('');

    console.log('✅ Demo completed successfully!');
    console.log('');
    console.log('Next Steps:');
    console.log('1. Review launch plan details');
    console.log('2. Approve launch (if needed)');
    console.log('3. Execute launch with orchestrator.executeLaunch()');
    console.log('4. Monitor real-time metrics');
    console.log('5. Optimize based on performance data');
    console.log('');

    // Cleanup
    orchestrator.shutdown();

    return launchPlan;
  } catch (error) {
    console.error('❌ Error during launch planning:', error);
    throw error;
  }
}

/**
 * Run the demo
 */
if (require.main === module) {
  launchPremiumTShirt()
    .then(() => {
      console.log('Demo finished successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('Demo failed:', error);
      process.exit(1);
    });
}

export { launchPremiumTShirt };
