# Operations Documentation

Operational guides, runbooks, and procedures for maintaining the repository.

## Contents

- [Runbooks](./runbooks/) - Step-by-step operational procedures
- [Deployment](./deployment.md) - Deployment procedures and checklist
- [Incident Response](./incident-response.md) - Handling production incidents
- [Monitoring](./monitoring.md) - Observability and alerting

## Quick Links

### Common Operations

- [Deploy to Production](./runbooks/deploy-production.md)
- [Rollback Deployment](./runbooks/rollback.md)
- [Handle Security Incident](./incident-response.md#security-incidents)
- [Update Dependencies](./runbooks/dependency-updates.md)

### Troubleshooting

- [Build Failures](./runbooks/troubleshoot-builds.md)
- [Test Failures](./runbooks/troubleshoot-tests.md)
- [CI/CD Issues](./runbooks/troubleshoot-ci.md)

## On-Call Procedures

When issues arise:

1. **Assess Severity** - Use incident severity matrix
2. **Follow Runbook** - Use appropriate operational runbook
3. **Communicate** - Update stakeholders
4. **Document** - Record resolution in postmortem

## Emergency Contacts

| Role             | Contact  | Scope              |
| ---------------- | -------- | ------------------ |
| Repository Owner | @alawein | All issues         |
| CI/CD            | @alawein | Pipeline issues    |
| Security         | @alawein | Security incidents |

## Escalation Path

```
L1: Self-service (runbooks) → L2: Repository maintainer → L3: External support
```
