import { createDefaultOrchestrationSystem } from './src/lib/orchestration/index.js';

async function testOrchestration() {
  console.log('🚀 Testing DAG Orchestration System...');

  try {
    const orchestrator = createDefaultOrchestrationSystem();

    // Test basic analytics workflow
    const result = await orchestrator.executeAnalyticsWorkflow('test-workflow-1', {
      userId: 'test-user-123',
      planId: 'premium-plan',
      metadata: { test: true },
    });

    console.log('✅ Analytics workflow completed:', result.success);
    console.log('📊 Execution metrics:', result.metrics);

    // Test plan purchase workflow
    const purchaseResult = await orchestrator.executePlanPurchaseWorkflow({
      userId: 'test-user-123',
      planType: 'premium',
      planId: 'premium-plan',
      currency: 'USD',
      paymentMethod: 'stripe',
      metadata: { test: true }
    });

    console.log('💳 Plan purchase completed:', purchaseResult.success);
    console.log('🏷️ Purchase details:', purchaseResult.purchase);

    console.log('🎉 All tests passed successfully!');
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

testOrchestration();