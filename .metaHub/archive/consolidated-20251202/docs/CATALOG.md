# Tool Catalog

Searchable index of all 57+ tools in the unified toolkit.

---

## AI Orchestration (14 tools)

| Tool | File | Description |
|------|------|-------------|
| Task Router | `tools/ai-orchestration/task-router.sh` | Bayesian ML-based task routing to best AI tool |
| Parallel Executor | `tools/ai-orchestration/parallel-executor.sh` | Git worktree-based concurrent execution |
| Dashboard | `tools/ai-orchestration/dashboard.sh` | Live metrics dashboard with auto-refresh |
| Self-Improving | `tools/ai-orchestration/self-improving.sh` | ML model training from outcomes |
| Context Compressor | `tools/ai-orchestration/context-compressor.sh` | Semantic TF-IDF context optimization |
| Cost Tracker | `tools/ai-orchestration/cost-tracker.sh` | Budget monitoring with alerts |
| Test Runner | `tools/ai-orchestration/test-runner.sh` | Multi-framework test execution |
| Checkpoint | `tools/ai-orchestration/checkpoint.sh` | Git-based undo/redo system |
| Validate | `tools/ai-orchestration/validate.sh` | Script and config validation |
| Secrets Manager | `tools/ai-orchestration/secrets-manager.sh` | Secure API key storage |
| Template Manager | `tools/ai-orchestration/template-manager.sh` | Prompt template management |
| Tool Chainer | `tools/ai-orchestration/tool-chainer.sh` | Workflow pipeline execution |
| Start MCP | `tools/ai-orchestration/mcp/start-ecosystem.sh` | Start MCP server ecosystem |
| Stop MCP | `tools/ai-orchestration/mcp/stop-ecosystem.sh` | Stop MCP servers |

## Governance (8 tools)

| Tool | File | Description |
|------|------|-------------|
| Compliance Validator | `tools/governance/compliance_validator.py` | Unified compliance checking |
| Structure Validator | `tools/governance/structure_validator.py` | Root structure validation |
| Enforce | `tools/governance/enforce.py` | Policy enforcement |
| Catalog | `tools/governance/catalog.py` | Asset cataloging |
| Checkpoint | `tools/governance/checkpoint.py` | State checkpointing |
| Sync Governance | `tools/governance/sync_governance.py` | Cross-project sync |
| AI Audit | `tools/governance/ai_audit.py` | AI configuration audit |
| Meta | `tools/governance/meta.py` | Meta-repository management |

## Orchestration (5 tools)

| Tool | File | Description |
|------|------|-------------|
| Orchestration Checkpoint | `tools/orchestration/orchestration_checkpoint.py` | Workflow state preservation |
| Orchestration Validator | `tools/orchestration/orchestration_validator.py` | Handoff envelope validation |
| Orchestration Telemetry | `tools/orchestration/orchestration_telemetry.py` | Metrics collection and reporting |
| Hallucination Verifier | `tools/orchestration/hallucination_verifier.py` | Three-layer verification cascade |
| Self-Healing Workflow | `tools/orchestration/self_healing_workflow.py` | Error recovery and degradation |

## DevOps CLI (6 modules)

| Tool | File | Description |
|------|------|-------------|
| Builder | `tools/devops-cli/builder.ts` | Template selection and copying |
| Coder | `tools/devops-cli/coder.ts` | Code generation with preview |
| Config | `tools/devops-cli/config.ts` | Target directory resolution |
| FS | `tools/devops-cli/fs.ts` | File system utilities |
| Install | `tools/devops-cli/install.ts` | Dependency installation |
| Bootstrap | `tools/devops-cli/bootstrap.ts` | Workspace initialization |

## Security (5 tools)

| Tool | File | Description |
|------|------|-------------|
| Dependency Scan | `tools/security/dependency-scan.sh` | Dependency vulnerability scanning |
| SAST Scan | `tools/security/sast-scan.sh` | Static application security testing |
| Secret Scan | `tools/security/secret-scan.sh` | Secret and credential detection |
| Trivy Scan | `tools/security/trivy-scan.sh` | Container image scanning |
| Security Scan All | `tools/security/security-scan-all.sh` | Combined security scanner |

## MCP Servers (3 tools)

| Tool | File | Description |
|------|------|-------------|
| MCP CLI Wrapper | `tools/mcp-servers/mcp_cli_wrapper.py` | MCP CLI wrapper with commands |
| MCP Server Tester | `tools/mcp-servers/mcp_server_tester.py` | Automated server testing |
| Agent MCP Integrator | `tools/mcp-servers/agent_mcp_integrator.py` | Agent integration |

## Automation (5 tools)

| Tool | File | Description |
|------|------|-------------|
| DevOps Workflow Runner | `tools/automation/devops_workflow_runner.py` | Workflow automation |
| Quick Start | `tools/automation/quick_start.py` | Quick start wizard |
| Setup Org | `tools/automation/setup_org.py` | Organization setup |
| Setup Repo CI | `tools/automation/setup_repo_ci.py` | CI/CD configuration |
| Push Monorepos | `tools/automation/push_monorepos.py` | Monorepo management |

## Meta (3 tools)

| Tool | File | Description |
|------|------|-------------|
| Telemetry Dashboard | `tools/meta/telemetry_dashboard.py` | Telemetry visualization |
| Create GitHub Repos | `tools/meta/create_github_repos.py` | Repository creation |

## Infrastructure (5 directories)

| Directory | Contents |
|-----------|----------|
| `tools/infrastructure/docker/` | Docker configurations, Compose files |
| `tools/infrastructure/kubernetes/` | K8s manifests, deployments |
| `tools/infrastructure/terraform/` | Terraform modules, providers |
| `tools/infrastructure/ansible/` | Ansible playbooks, roles |
| `tools/infrastructure/gitops/` | GitOps configurations |

---

## Templates (17 across 8 categories)

| Category | Path | Contents |
|----------|------|----------|
| CI/CD | `templates/cicd/` | GitHub Actions, Jenkins, CircleCI |
| Kubernetes | `templates/k8s/` | Deployment, Service, Helm |
| Monitoring | `templates/monitoring/` | Prometheus, Grafana |
| Logging | `templates/logging/` | ELK stack pipeline |
| IaC | `templates/iac/` | Terraform, Pulumi |
| Database | `templates/db/` | Migration templates |
| UI | `templates/ui/` | Frontend templates |
| Demos | `templates/demos/` | Complete demo apps |

---

## Quick Reference

### By Use Case

| I want to... | Use this tool |
|--------------|---------------|
| Route a task to the best AI | `task-router.sh` |
| Run tasks in parallel | `parallel-executor.sh` |
| View metrics | `dashboard.sh` |
| Check compliance | `compliance_validator.py` |
| Enforce policies | `enforce.py` |
| Generate from template | `builder.ts` |
| Scan for vulnerabilities | `security-scan-all.sh` |
| Start MCP servers | `mcp/start-ecosystem.sh` |
| Automate workflow | `devops_workflow_runner.py` |

### By Language

| Language | Tools |
|----------|-------|
| Bash | 19 scripts in ai-orchestration/, security/ |
| Python | 21 scripts in governance/, orchestration/, mcp-servers/, automation/, meta/ |
| TypeScript | 6 modules in devops-cli/ |
