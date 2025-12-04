/**
 * Skeptic Reviewer Agent
 * Challenges every proposed change with "what could go wrong?" analysis
 * Implements self-refutation pattern for robust code review
 */

import type { ProposedChange, GovernanceFinding, ChangeReview, AgentRole } from './orchestrator.js';

export interface SkepticAnalysis {
  changeId: string;
  risks: Risk[];
  questions: string[];
  concerns: Concern[];
  recommendation: 'approve' | 'reject' | 'request-changes';
  confidence: number;
  refutationRounds: RefutationRound[];
}

export interface Risk {
  category: 'breaking-change' | 'security' | 'performance' | 'regression' | 'compatibility';
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  mitigation?: string;
}

export interface Concern {
  type: 'missing-tests' | 'incomplete-change' | 'side-effects' | 'unclear-intent' | 'scope-creep';
  description: string;
  evidence?: string;
}

export interface RefutationRound {
  round: number;
  challenge: string;
  response: string;
  resolved: boolean;
}

export interface SkepticConfig {
  minRefutationRounds: number;
  riskThreshold: number;
  requireTestCoverage: boolean;
  blockOnCriticalRisks: boolean;
}

const DEFAULT_CONFIG: SkepticConfig = {
  minRefutationRounds: 2,
  riskThreshold: 0.7,
  requireTestCoverage: true,
  blockOnCriticalRisks: true,
};

/**
 * Skeptic Reviewer Agent
 * Provides adversarial review of all proposed changes
 */
export class SkepticReviewer {
  private config: SkepticConfig;
  private analyses: Map<string, SkepticAnalysis> = new Map();
  public readonly role: AgentRole = 'skeptic';

  constructor(config: Partial<SkepticConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Analyze a proposed change for potential issues
   */
  public async analyzeChange(change: ProposedChange): Promise<SkepticAnalysis> {
    const analysis: SkepticAnalysis = {
      changeId: change.id,
      risks: [],
      questions: [],
      concerns: [],
      recommendation: 'approve',
      confidence: 1.0,
      refutationRounds: [],
    };

    // Phase 1: Risk Assessment
    analysis.risks = await this.assessRisks(change);

    // Phase 2: Generate Skeptical Questions
    analysis.questions = this.generateQuestions(change);

    // Phase 3: Identify Concerns
    analysis.concerns = await this.identifyConcerns(change);

    // Phase 4: Self-Refutation Loop
    for (let round = 1; round <= this.config.minRefutationRounds; round++) {
      const refutation = await this.performRefutation(change, analysis, round);
      analysis.refutationRounds.push(refutation);
    }

    // Phase 5: Final Recommendation
    analysis.recommendation = this.determineRecommendation(analysis);
    analysis.confidence = this.calculateConfidence(analysis);

    this.analyses.set(change.id, analysis);
    return analysis;
  }

  private async assessRisks(change: ProposedChange): Promise<Risk[]> {
    const risks: Risk[] = [];

    // Check for breaking changes
    if (this.detectBreakingChange(change)) {
      risks.push({
        category: 'breaking-change',
        severity: 'high',
        description: 'This change modifies public interfaces or exports',
        mitigation: 'Ensure backward compatibility or update all consumers',
      });
    }

    // Check for security implications
    if (this.detectSecurityRisk(change)) {
      risks.push({
        category: 'security',
        severity: 'critical',
        description: 'Change touches security-sensitive code paths',
        mitigation: 'Conduct security review before merging',
      });
    }

    // Check for performance implications
    if (this.detectPerformanceRisk(change)) {
      risks.push({
        category: 'performance',
        severity: 'medium',
        description: 'Change may impact performance',
        mitigation: 'Run performance benchmarks',
      });
    }

    return risks;
  }

  private detectBreakingChange(change: ProposedChange): boolean {
    const breakingPatterns = [
      /export\s+(function|class|const|interface|type)\s+\w+/,
      /public\s+(async\s+)?(\w+)\s*\(/,
      /^-.*export/m,
    ];
    return change.diff ? breakingPatterns.some((p) => p.test(change.diff!)) : false;
  }

  private detectSecurityRisk(change: ProposedChange): boolean {
    const securityPatterns = [
      /password|secret|token|api[_-]?key|auth/i,
      /eval\(|exec\(|child_process|spawn/,
      /fs\.writeFile|fs\.unlink|fs\.rmdir/,
    ];
    return change.diff ? securityPatterns.some((p) => p.test(change.diff!)) : false;
  }

  private detectPerformanceRisk(change: ProposedChange): boolean {
    const perfPatterns = [
      /\.forEach\(|\.map\(.*\.map\(/,
      /while\s*\(true\)/,
      /setTimeout|setInterval/,
    ];
    return change.diff ? perfPatterns.some((p) => p.test(change.diff!)) : false;
  }

  private generateQuestions(change: ProposedChange): string[] {
    const questions: string[] = [
      'What happens if this change fails silently?',
      'Are there any edge cases not covered by tests?',
      'Could this change break existing functionality?',
      'Is the error handling sufficient?',
      'Are there any race conditions possible?',
    ];

    // Add context-specific questions based on the change
    if (change.file.includes('test')) {
      questions.push('Do these tests actually validate the intended behavior?');
    }
    if (change.file.includes('config')) {
      questions.push('Are all configuration defaults sensible?');
    }
    return questions;
  }

  private async identifyConcerns(change: ProposedChange): Promise<Concern[]> {
    const concerns: Concern[] = [];

    // Check for missing tests
    if (!change.file.includes('test') && this.config.requireTestCoverage) {
      concerns.push({
        type: 'missing-tests',
        description: 'No corresponding test file changes detected',
        evidence: `Change to ${change.file} without test updates`,
      });
    }

    // Check for incomplete changes
    if (change.description.includes('TODO') || change.description.includes('FIXME')) {
      concerns.push({
        type: 'incomplete-change',
        description: 'Change contains TODO/FIXME markers',
      });
    }

    return concerns;
  }

  private async performRefutation(
    change: ProposedChange,
    analysis: SkepticAnalysis,
    round: number
  ): Promise<RefutationRound> {
    // Generate a challenge based on current analysis
    const challenge = this.generateChallenge(analysis, round);

    // Attempt to address the challenge (simulated response)
    const response = this.attemptToAddress(challenge, change);

    // Determine if the challenge was resolved
    const resolved = this.evaluateResponse(challenge, response);

    return { round, challenge, response, resolved };
  }

  private generateChallenge(_analysis: SkepticAnalysis, round: number): string {
    const challenges = [
      'What is the worst-case scenario if this change is deployed?',
      'How do we know this change is actually correct?',
      'What assumptions are we making that might be wrong?',
      'Is there a simpler solution we are missing?',
      'How will this change interact with concurrent operations?',
    ];
    return challenges[round % challenges.length] || challenges[0];
  }

  private attemptToAddress(challenge: string, change: ProposedChange): string {
    // This would be enhanced with actual analysis or LLM calls
    return `Analysis of "${change.description}" against challenge: ${challenge}`;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- Parameters reserved for future semantic analysis
  private evaluateResponse(_challenge: string, _response: string): boolean {
    // Simplified evaluation - would be enhanced with semantic analysis
    return true;
  }

  private determineRecommendation(
    analysis: SkepticAnalysis
  ): 'approve' | 'reject' | 'request-changes' {
    const criticalRisks = analysis.risks.filter((r) => r.severity === 'critical');
    const highRisks = analysis.risks.filter((r) => r.severity === 'high');
    const unresolvedRefutations = analysis.refutationRounds.filter((r) => !r.resolved);

    if (this.config.blockOnCriticalRisks && criticalRisks.length > 0) {
      return 'reject';
    }

    if (highRisks.length > 2 || unresolvedRefutations.length > 1) {
      return 'request-changes';
    }

    if (analysis.concerns.length > 3) {
      return 'request-changes';
    }

    return 'approve';
  }

  private calculateConfidence(analysis: SkepticAnalysis): number {
    let confidence = 1.0;

    // Reduce confidence based on risks
    analysis.risks.forEach((risk) => {
      switch (risk.severity) {
        case 'critical':
          confidence -= 0.3;
          break;
        case 'high':
          confidence -= 0.2;
          break;
        case 'medium':
          confidence -= 0.1;
          break;
        case 'low':
          confidence -= 0.05;
          break;
      }
    });

    // Reduce confidence for unresolved refutations
    const unresolvedCount = analysis.refutationRounds.filter((r) => !r.resolved).length;
    confidence -= unresolvedCount * 0.15;

    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Generate a review for integration with orchestrator
   */
  public async generateReview(change: ProposedChange): Promise<ChangeReview> {
    const analysis = await this.analyzeChange(change);

    const comments = [
      `Skeptic Analysis (Confidence: ${(analysis.confidence * 100).toFixed(0)}%)`,
      '',
      `Risks identified: ${analysis.risks.length}`,
      ...analysis.risks.map((r) => `  - [${r.severity.toUpperCase()}] ${r.description}`),
      '',
      `Concerns: ${analysis.concerns.length}`,
      ...analysis.concerns.map((c) => `  - [${c.type}] ${c.description}`),
      '',
      `Refutation rounds: ${analysis.refutationRounds.length}`,
      ...analysis.refutationRounds.map(
        (r) => `  Round ${r.round}: ${r.resolved ? '✓' : '✗'} ${r.challenge}`
      ),
    ].join('\n');

    return {
      reviewer: this.role,
      decision: analysis.recommendation,
      comments,
      timestamp: new Date(),
    };
  }

  /**
   * Get analysis for a specific change
   */
  public getAnalysis(changeId: string): SkepticAnalysis | undefined {
    return this.analyses.get(changeId);
  }

  /**
   * Generate finding from analysis
   */
  public analysisToFindings(analysis: SkepticAnalysis): GovernanceFinding[] {
    return analysis.risks.map((risk) => ({
      agent: this.role,
      severity: risk.severity,
      category: risk.category,
      message: risk.description,
      suggestion: risk.mitigation,
    }));
  }
}

export function createSkepticReviewer(config?: Partial<SkepticConfig>): SkepticReviewer {
  return new SkepticReviewer(config);
}
