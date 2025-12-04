/**
 * Governance Agents Index
 * Multi-agent system for automated code review and governance enforcement
 */

export * from './orchestrator.js';
export * from './skeptic-reviewer.js';
export * from './dashboard.js';

// Re-export types for convenience
export type {
  AgentRole,
  AgentMessage,
  ValidationCheckpoint,
  GovernanceReport,
  GovernanceFinding,
  ProposedChange,
  ChangeReview,
  ValidationResult,
  OrchestratorConfig,
} from './orchestrator.js';

export type {
  SkepticAnalysis,
  Risk,
  Concern,
  RefutationRound,
  SkepticConfig,
} from './skeptic-reviewer.js';

export type { HealthMetrics, CategoryScore, TrendData, DashboardConfig } from './dashboard.js';

/**
 * Quick setup for governance workflow
 */
import { GovernanceOrchestrator, createGovernanceOrchestrator } from './orchestrator.js';
import { SkepticReviewer, createSkepticReviewer } from './skeptic-reviewer.js';

export interface GovernanceSystem {
  orchestrator: GovernanceOrchestrator;
  skeptic: SkepticReviewer;
}

export function createGovernanceSystem(): GovernanceSystem {
  const orchestrator = createGovernanceOrchestrator({
    minReviewRounds: 2,
    requireSkepticApproval: true,
    autoApplyApproved: false,
  });

  const skeptic = createSkepticReviewer({
    minRefutationRounds: 2,
    blockOnCriticalRisks: true,
  });

  // Wire up skeptic to orchestrator events
  orchestrator.on('change-proposed', async (change) => {
    const review = await skeptic.generateReview(change);
    orchestrator.reviewChange(change.id, review);

    // Register findings from skeptic analysis
    const analysis = skeptic.getAnalysis(change.id);
    if (analysis) {
      const findings = skeptic.analysisToFindings(analysis);
      findings.forEach((f) => orchestrator.registerFinding(f));
    }
  });

  return { orchestrator, skeptic };
}
