/**
 * Governance Agent Orchestrator
 * Coordinates multi-agent workflows for automated code review and governance enforcement
 */

import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';

// Agent role definitions
export type AgentRole = 'analyzer' | 'refactor' | 'skeptic' | 'organizer' | 'security';

export interface AgentMessage {
  from: AgentRole;
  to: AgentRole | 'orchestrator';
  type: 'proposal' | 'review' | 'approval' | 'rejection' | 'question' | 'report';
  content: unknown;
  timestamp: Date;
  correlationId: string;
}

export interface ValidationCheckpoint {
  name: string;
  check: () => Promise<boolean>;
  required: boolean;
}

export interface GovernanceReport {
  timestamp: Date;
  agents: AgentRole[];
  findings: GovernanceFinding[];
  changes: ProposedChange[];
  validationResults: ValidationResult[];
  overallStatus: 'pass' | 'fail' | 'partial';
}

export interface GovernanceFinding {
  agent: AgentRole;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  category: string;
  message: string;
  file?: string;
  line?: number;
  suggestion?: string;
}

export interface ProposedChange {
  id: string;
  agent: AgentRole;
  file: string;
  description: string;
  diff?: string;
  status: 'pending' | 'approved' | 'rejected' | 'applied';
  reviews: ChangeReview[];
}

export interface ChangeReview {
  reviewer: AgentRole;
  decision: 'approve' | 'reject' | 'request-changes';
  comments: string;
  timestamp: Date;
}

export interface ValidationResult {
  checkpoint: string;
  passed: boolean;
  message: string;
  timestamp: Date;
}

export interface OrchestratorConfig {
  workspacePath: string;
  minReviewRounds: number;
  requireSkepticApproval: boolean;
  autoApplyApproved: boolean;
  reportPath: string;
}

const DEFAULT_CONFIG: OrchestratorConfig = {
  workspacePath: process.cwd(),
  minReviewRounds: 2,
  requireSkepticApproval: true,
  autoApplyApproved: false,
  reportPath: '.archive/reports/governance',
};

/**
 * Multi-Agent Governance Orchestrator
 * Coordinates analyzer, refactor, skeptic, organizer, and security agents
 */
export class GovernanceOrchestrator extends EventEmitter {
  private config: OrchestratorConfig;
  private _messages: AgentMessage[] = [];
  private pendingChanges: Map<string, ProposedChange> = new Map();
  private findings: GovernanceFinding[] = [];
  private validationResults: ValidationResult[] = [];
  private checkpoints: ValidationCheckpoint[] = [];

  /** Get all messages in the orchestrator */
  public get messages(): AgentMessage[] {
    return this._messages;
  }

  constructor(config: Partial<OrchestratorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initializeCheckpoints();
  }

  private initializeCheckpoints(): void {
    this.checkpoints = [
      {
        name: 'TypeScript Compilation',
        check: () => this.runCommand('npx tsc --noEmit'),
        required: true,
      },
      {
        name: 'ESLint',
        check: () => this.runCommand('npm run lint'),
        required: true,
      },
      {
        name: 'Tests',
        check: () => this.runCommand('npm run test:run'),
        required: true,
      },
      {
        name: 'Security Scan',
        check: () => this.runCommand('npm audit --audit-level=high'),
        required: false,
      },
    ];
  }

  private async runCommand(cmd: string): Promise<boolean> {
    const { exec } = await import('child_process');
    return new Promise((resolve) => {
      exec(cmd, { cwd: this.config.workspacePath }, (error) => {
        resolve(!error);
      });
    });
  }

  /**
   * Register a finding from an agent
   */
  public registerFinding(finding: GovernanceFinding): void {
    this.findings.push(finding);
    this.emit('finding', finding);
  }

  /**
   * Propose a change for review
   */
  public proposeChange(change: Omit<ProposedChange, 'id' | 'status' | 'reviews'>): string {
    const id = `change-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const proposedChange: ProposedChange = {
      ...change,
      id,
      status: 'pending',
      reviews: [],
    };
    this.pendingChanges.set(id, proposedChange);
    this.emit('change-proposed', proposedChange);
    return id;
  }

  /**
   * Submit a review for a proposed change
   */
  public reviewChange(changeId: string, review: Omit<ChangeReview, 'timestamp'>): boolean {
    const change = this.pendingChanges.get(changeId);
    if (!change) return false;

    change.reviews.push({ ...review, timestamp: new Date() });
    this.emit('change-reviewed', { change, review });

    // Check if we have enough reviews
    if (change.reviews.length >= this.config.minReviewRounds) {
      this.evaluateChangeStatus(change);
    }
    return true;
  }

  private evaluateChangeStatus(change: ProposedChange): void {
    const approvals = change.reviews.filter((r) => r.decision === 'approve').length;
    const rejections = change.reviews.filter((r) => r.decision === 'reject').length;
    const skepticReview = change.reviews.find((r) => r.reviewer === 'skeptic');

    if (
      this.config.requireSkepticApproval &&
      (!skepticReview || skepticReview.decision !== 'approve')
    ) {
      change.status = 'pending';
      return;
    }

    if (rejections > 0) {
      change.status = 'rejected';
    } else if (approvals >= this.config.minReviewRounds) {
      change.status = 'approved';
      if (this.config.autoApplyApproved) {
        this.applyChange(change.id);
      }
    }
    this.emit('change-status-updated', change);
  }

  /**
   * Apply an approved change
   */
  public async applyChange(changeId: string): Promise<boolean> {
    const change = this.pendingChanges.get(changeId);
    if (!change || change.status !== 'approved') return false;

    try {
      // Apply change logic would go here
      change.status = 'applied';
      this.emit('change-applied', change);
      return true;
    } catch (error) {
      this.emit('change-error', { change, error });
      return false;
    }
  }

  /**
   * Run all validation checkpoints
   */
  public async runValidation(): Promise<ValidationResult[]> {
    this.validationResults = [];

    for (const checkpoint of this.checkpoints) {
      const passed = await checkpoint.check();
      const result: ValidationResult = {
        checkpoint: checkpoint.name,
        passed,
        message: passed ? `${checkpoint.name} passed` : `${checkpoint.name} failed`,
        timestamp: new Date(),
      };
      this.validationResults.push(result);
      this.emit('validation-result', result);

      if (checkpoint.required && !passed) {
        this.emit('validation-failed', result);
      }
    }
    return this.validationResults;
  }

  /**
   * Generate comprehensive governance report
   */
  public generateReport(): GovernanceReport {
    const allPassed = this.validationResults.every((r) => r.passed);
    const requiredPassed = this.validationResults
      .filter((_, i) => this.checkpoints[i]?.required)
      .every((r) => r.passed);

    const report: GovernanceReport = {
      timestamp: new Date(),
      agents: ['analyzer', 'refactor', 'skeptic', 'organizer', 'security'],
      findings: this.findings,
      changes: Array.from(this.pendingChanges.values()),
      validationResults: this.validationResults,
      overallStatus: allPassed ? 'pass' : requiredPassed ? 'partial' : 'fail',
    };

    // Save report to file
    this.saveReport(report);
    return report;
  }

  private saveReport(report: GovernanceReport): void {
    const reportDir = path.join(this.config.workspacePath, this.config.reportPath);
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    const filename = `governance-${new Date().toISOString().split('T')[0]}.json`;
    fs.writeFileSync(path.join(reportDir, filename), JSON.stringify(report, null, 2));
  }

  /**
   * Get current orchestrator status
   */
  public getStatus(): {
    pendingChanges: number;
    approvedChanges: number;
    rejectedChanges: number;
    findings: { critical: number; high: number; medium: number; low: number };
  } {
    const changes = Array.from(this.pendingChanges.values());
    return {
      pendingChanges: changes.filter((c) => c.status === 'pending').length,
      approvedChanges: changes.filter((c) => c.status === 'approved').length,
      rejectedChanges: changes.filter((c) => c.status === 'rejected').length,
      findings: {
        critical: this.findings.filter((f) => f.severity === 'critical').length,
        high: this.findings.filter((f) => f.severity === 'high').length,
        medium: this.findings.filter((f) => f.severity === 'medium').length,
        low: this.findings.filter((f) => f.severity === 'low').length,
      },
    };
  }
}

export function createGovernanceOrchestrator(
  config?: Partial<OrchestratorConfig>
): GovernanceOrchestrator {
  return new GovernanceOrchestrator(config);
}
