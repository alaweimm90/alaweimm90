const { join } = require('path');
const { writeFile, mkdir } = require('fs').promises;

class ApprovalPolicy {
  constructor({ logger, securityModule, rbac, logDir }) {
    this.logger = logger;
    this.securityModule = securityModule;
    this.rbac = rbac;
    this.logDir = logDir;
  }

  async autoApprove(task, context) {
    const decision = { approved: true, reasons: [], task: task.name };

    const actor = context?.actor || { role: 'developer' };
    const resource = context?.resource || 'repo';
    const action = context?.action || task.name;

    if (!this.rbac.hasPermission(actor.role, action, resource)) {
      decision.approved = false;
      decision.reasons.push('rbac_denied');
    }

    const scan = await this.securityModule.scanSecrets({ targetDir: context?.cwd || '.' });
    if (scan.status === 'critical' || scan.secretsFound.length > 0) {
      decision.approved = false;
      decision.reasons.push('security_findings');
    }

    await this.persistDecision(decision);
    this.logger.info('Approval decision', decision);
    return decision;
  }

  async persistDecision(decision) {
    const dir = join(this.logDir, 'approvals');
    await mkdir(dir, { recursive: true });
    const file = join(dir, `approval-${Date.now()}.json`);
    await writeFile(file, JSON.stringify(decision));
  }
}

module.exports = { ApprovalPolicy };
