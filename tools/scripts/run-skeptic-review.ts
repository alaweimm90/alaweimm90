/**
 * Execute SkepticReviewer analysis on the governance system
 * Generates a comprehensive review report with identified risks
 */

import {
  createSkepticReviewer,
  createGovernanceOrchestrator,
} from '../automation/agents/governance/index.js';
import type { ProposedChange } from '../automation/agents/governance/orchestrator.js';
import * as fs from 'fs';
import * as path from 'path';

async function runSkepticReview() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║         SKEPTIC REVIEWER - GOVERNANCE SYSTEM ANALYSIS        ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  // Initialize components
  const skeptic = createSkepticReviewer({
    minRefutationRounds: 3,
    riskThreshold: 0.7,
    requireTestCoverage: true,
    blockOnCriticalRisks: true,
  });

  const orchestrator = createGovernanceOrchestrator({
    workspacePath: process.cwd(),
    minReviewRounds: 2,
    requireSkepticApproval: true,
  });

  // Define governance components to analyze
  const components: Omit<ProposedChange, 'id' | 'status' | 'reviews'>[] = [
    {
      agent: 'analyzer',
      file: 'automation/agents/governance/orchestrator.ts',
      description: 'GovernanceOrchestrator - Multi-agent workflow coordinator',
      diff: fs.readFileSync('automation/agents/governance/orchestrator.ts', 'utf8'),
    },
    {
      agent: 'analyzer',
      file: 'automation/agents/governance/skeptic-reviewer.ts',
      description: 'SkepticReviewer - Adversarial review with self-refutation',
      diff: fs.readFileSync('automation/agents/governance/skeptic-reviewer.ts', 'utf8'),
    },
    {
      agent: 'analyzer',
      file: 'automation/agents/governance/dashboard.ts',
      description: 'GovernanceDashboard - Health metrics and monitoring',
      diff: fs.readFileSync('automation/agents/governance/dashboard.ts', 'utf8'),
    },
    {
      agent: 'analyzer',
      file: 'automation/workflows/auto-refactor.ts',
      description: 'AutoRefactorPipeline - Automated refactoring with validation',
      diff: fs.readFileSync('automation/workflows/auto-refactor.ts', 'utf8'),
    },
  ];

  const allAnalyses = [];
  const allFindings = [];

  // Analyze each component
  for (const component of components) {
    console.log(`\n🔍 Analyzing: ${component.file}`);
    console.log(`   ${component.description}`);
    console.log('─'.repeat(60));

    // Propose the change
    const changeId = orchestrator.proposeChange(component);
    const change = { ...component, id: changeId, status: 'pending' as const, reviews: [] };

    // Run skeptic analysis
    const analysis = await skeptic.analyzeChange(change);
    allAnalyses.push({ file: component.file, analysis });

    // Convert to findings
    const findings = skeptic.analysisToFindings(analysis);
    allFindings.push(...findings);

    // Display results
    console.log(`   Recommendation: ${analysis.recommendation.toUpperCase()}`);
    console.log(`   Confidence: ${(analysis.confidence * 100).toFixed(0)}%`);
    console.log(`   Risks: ${analysis.risks.length}`);
    console.log(`   Concerns: ${analysis.concerns.length}`);
    console.log(`   Refutation Rounds: ${analysis.refutationRounds.length}`);

    if (analysis.risks.length > 0) {
      console.log('\n   Risks:');
      for (const risk of analysis.risks) {
        console.log(`   ├─ [${risk.severity.toUpperCase()}] ${risk.category}`);
        console.log(`   │  ${risk.description}`);
        if (risk.mitigation) {
          console.log(`   │  💡 ${risk.mitigation}`);
        }
      }
    }

    if (analysis.concerns.length > 0) {
      console.log('\n   Concerns:');
      for (const concern of analysis.concerns) {
        console.log(`   ├─ [${concern.type}] ${concern.description}`);
      }
    }

    console.log('\n   Refutation Rounds:');
    for (const round of analysis.refutationRounds) {
      const icon = round.resolved ? '✓' : '✗';
      console.log(`   ├─ Round ${round.round}: ${icon} ${round.challenge}`);
    }

    // Generate review for orchestrator
    const review = await skeptic.generateReview(change);
    orchestrator.reviewChange(changeId, review);
  }

  // Summary
  console.log('\n\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║                      ANALYSIS SUMMARY                        ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  const status = orchestrator.getStatus();
  console.log(`📊 Orchestrator Status:`);
  console.log(`   Pending Changes: ${status.pendingChanges}`);
  console.log(`   Approved Changes: ${status.approvedChanges}`);
  console.log(`   Rejected Changes: ${status.rejectedChanges}`);
  console.log(`\n📋 Findings by Severity:`);
  console.log(`   Critical: ${status.findings.critical}`);
  console.log(`   High: ${status.findings.high}`);
  console.log(`   Medium: ${status.findings.medium}`);
  console.log(`   Low: ${status.findings.low}`);

  // Save report
  const reportDir = path.join(process.cwd(), '.archive/reports/governance');
  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
  }

  const reportPath = path.join(
    reportDir,
    `skeptic-review-${new Date().toISOString().split('T')[0]}.json`
  );
  const report = {
    timestamp: new Date().toISOString(),
    analyses: allAnalyses,
    findings: allFindings,
    summary: status,
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n💾 Report saved to: ${reportPath}`);

  // Generate governance report
  const govReport = orchestrator.generateReport();
  console.log(`\n📄 Governance Report: ${govReport.overallStatus.toUpperCase()}`);

  return { analyses: allAnalyses, findings: allFindings, report: govReport };
}

runSkepticReview().catch(console.error);
