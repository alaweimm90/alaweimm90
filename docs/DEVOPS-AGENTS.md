# DevOps Agent Reference (Top 20 Essentials)

This document catalogs 20 essential DevOps agents with specifications, invocation syntax, activation scenarios, version compatibility, security considerations, and practical examples. Agents are expressed generically; adapt CLI/API names to your platform (`agentctl`, `agentd`, or your orchestrator).

---

## 1) Pipeline Orchestrator Agent

### Smart Generate Agent [DESCRIPTION]

- Orchestrates multi-stage CI/CD workflows: fan-in/out, gates, matrices, approvals, retries.
- Core: DAG scheduling, artifact passing, conditional steps, concurrency and queueing.
- Inputs: `pipelineFile`, `env`, `secretsRef`, `parameters` map; Outputs: status, stage logs, artifacts index.
- Perf: Parallel stages up to host core count; queue fairness; limitation: long-running stages can block shared runners.

### What to Call [pipeline-orchestrator]

- Identifier: `pipeline-orchestrator` • Aliases: `workflow`, `ci-pipeline`.
- CLI: `agentctl run pipeline-orchestrator --pipeline .\ci\main.yml --env prod --params .\params.json`
- API: `POST /agents/v1/pipeline-orchestrator/run`

### When to Call

- Trigger on PRs, merges to `main`, nightly builds; gated releases.
- Preconditions: reachable SCM, valid pipeline YAML, runner capacity.
- Auto-invoke on stage dependency resolution or re-run on transient failures.
- Integrates with Build, Test, Deploy, and Notification agents.

### Version Compatibility

- Works with GitHub Actions YAML 1.0+, Azure DevOps pipelines, GitLab CI 16+, Jenkinsfiles.
- Windows Server 2019+, PowerShell 7+, .NET 6+ runner or container runtime.

### Security Considerations

- Least-privilege tokens for SCM and artifact stores; sanitize env and masked logs.
- Disallow untrusted script injection; approve external reusable workflows.

### Examples

```powershell
agentctl run pipeline-orchestrator --pipeline .\ci\main.yml --env prod --params .\ci\params.prod.json
```

```http
POST /agents/v1/pipeline-orchestrator/run
Content-Type: application/json

{
  "pipelineFile": "ci/main.yml",
  "env": "prod",
  "parameters": {"release": "2025.11.30"},
  "secretsRef": "kv://ci/prod"
}
```

---

## 2) Build Agent

### Smart Generate Agent [DESCRIPTION]

- Compiles source, resolves deps, produces versioned artifacts (binaries, images).
- Inputs: `sourcePath`, `buildSpec`, `cacheKey`; Outputs: build logs, artifact IDs.
- Perf: Incremental caching, parallel module builds; limits: cache invalidation on lockfile changes.

### What to Call [build]

- Identifier: `build` • Aliases: `compiler`, `builder`.
- CLI: `agentctl run build --spec .\buildspec.yml --src . --out .\dist`
- API: `POST /agents/v1/build/run`

### When to Call

- After SCM checkout; before tests; on tag creation.
- Preconditions: toolchain installed; network for dependencies.

### Version Compatibility

- Node 20+/npm 10+, Java 17+, .NET 8, Go 1.22+. Windows x64 build tools.

### Security Considerations

- Pin dependencies; verify checksums; sandbox builds.

### Examples

```powershell
agentctl run build --spec .\buildspec.yml --src . --out .\dist --cacheKey app-v1
```

```json
POST /agents/v1/build/run
{
  "sourcePath": ".",
  "buildSpec": "buildspec.yml",
  "cacheKey": "app-v1"
}
```

---

## 3) Test Runner Agent

### Smart Generate Agent [DESCRIPTION]

- Executes unit/integration/e2e suites; collects coverage, flaky test detection.
- Inputs: `testSpec`, `shard`, `env`; Outputs: JUnit XML, coverage reports, trend metrics.
- Perf: Sharding by historical runtime; retries for flaky tests.

### What to Call [test-runner]

- Identifier: `test-runner` • Aliases: `qa`, `ci-tests`.
- CLI: `agentctl run test-runner --spec .\tests.yml --report .\reports`
- API: `POST /agents/v1/test-runner/run`

### When to Call

- After build; before packaging/deploy; on PR updates.
- Preconditions: Test data/fixtures; services mocked or available.

### Version Compatibility

- Jest 29+, NUnit 3+, JUnit 5, Cypress 13+. Windows runners supported.

### Security Considerations

- No real secrets in test env; mask logs; isolate e2e against prod data.

### Examples

```powershell
agentctl run test-runner --spec .\tests.yml --env staging --coverage
```

```json
POST /agents/v1/test-runner/run
{"testSpec":"tests.yml","env":"staging","shard":"auto"}
```

---

## 4) Artifact Repository Agent

### Smart Generate Agent [DESCRIPTION]

- Publishes and retrieves artifacts from registries (Nexus, Artifactory, S3).
- Inputs: `artifactPath`, `repoRef`, `metadata`; Outputs: immutable artifact URL, checksum.

### What to Call [artifact-repo]

- Identifier: `artifact-repo` • Aliases: `publisher`, `registry-client`.
- CLI: `agentctl run artifact-repo --push .\dist\app.zip --repo nuget://internal`
- API: `POST /agents/v1/artifact-repo/push`

### When to Call

- After successful build/test; before deploy.
- Preconditions: repo credentials; size quotas.

### Version Compatibility

- Supports Maven/NPM/NuGet/Docker registries versions current.

### Security Considerations

- Signed artifacts; enforce immutability; use scoped tokens.

### Examples

```powershell
agentctl run artifact-repo --push .\dist\service.exe --repo s3://artifacts/prod
```

```json
POST /agents/v1/artifact-repo/push
{"artifactPath":"dist/service.exe","repoRef":"s3://artifacts/prod","metadata":{"version":"1.2.3"}}
```

---

## 5) Container Builder Agent

### Smart Generate Agent [DESCRIPTION]

- Builds OCI images via Docker/BuildKit, multi-stage, SBOM generation.
- Inputs: `dockerfile`, `context`, `tags`; Outputs: image digest, SBOM.

### What to Call [container-build]

- Identifier: `container-build` • Aliases: `image-builder`.
- CLI: `agentctl run container-build --file .\Dockerfile --context . --tag mysvc:1.2.3`
- API: `POST /agents/v1/container-build/run`

### When to Call

- After build artifacts available; before scanner/deploy.
- Preconditions: Docker 24+/Buildx; registry access.

### Version Compatibility

- OCI v1; Docker 24+, containerd 1.7+.

### Security Considerations

- Minimal base images; non-root user; SBOM export; secrets not baked in.

### Examples

```powershell
agentctl run container-build --file .\Dockerfile --context . --tag registry.local/mysvc:1.2.3
```

```json
POST /agents/v1/container-build/run
{"dockerfile":"Dockerfile","context":".","tags":["registry.local/mysvc:1.2.3"]}
```

---

## 6) Image Vulnerability Scanner Agent

### Smart Generate Agent [DESCRIPTION]

- Scans images and dependencies (Trivy/Grype) for CVEs; policy gates.
- Inputs: `imageRef`, `policy`; Outputs: report, pass/fail gate, remediation hints.

### What to Call [image-scan]

- Identifier: `image-scan` • Aliases: `vuln-scan`.
- CLI: `agentctl run image-scan --image registry.local/mysvc:1.2.3 --policy high-only`
- API: `POST /agents/v1/image-scan/run`

### When to Call

- Pre-deploy gate; nightly scans; on base-image updates.

### Version Compatibility

- Trivy 0.50+, Grype 0.76+; supports OCI.

### Security Considerations

- CVE database freshness; allowlists reviewed; produce SBOM-linked findings.

### Examples

```powershell
agentctl run image-scan --image registry.local/mysvc:1.2.3 --policy "maxSeverity=High"
```

```json
POST /agents/v1/image-scan/run
{"imageRef":"registry.local/mysvc:1.2.3","policy":{"maxSeverity":"High"}}
```

---

## 7) Secret Management Agent

### Smart Generate Agent [DESCRIPTION]

- Fetches, rotates, and injects secrets from KMS/Vault/Azure Key Vault.
- Inputs: `secretsRef`, `bindings`; Outputs: ephemeral env files, rotation status.

### What to Call [secrets]

- Identifier: `secrets` • Aliases: `vault-client`, `kms`.
- CLI: `agentctl run secrets --ref kv://ci/prod --bind .\bindings.yml`
- API: `POST /agents/v1/secrets/fetch`

### When to Call

- Before build/test/deploy steps requiring credentials; on rotation windows.

### Version Compatibility

- HashiCorp Vault 1.15+, AWS KMS, Azure Key Vault v7+.

### Security Considerations

- Short TTL tokens; just-in-time injection; redact logs; audit access.

### Examples

```powershell
agentctl run secrets --ref kv://apps/prod --bind .\bindings.yml --ttl 900
```

```json
POST /agents/v1/secrets/fetch
{"secretsRef":"kv://apps/prod","bindings":{"DB_PASSWORD":"env:DB_PASSWORD"}}
```

---

## 8) Infrastructure Provisioner Agent

### Smart Generate Agent [DESCRIPTION]

- Plans/applies IaC (Terraform/Pulumi) with state management and drift detection.
- Inputs: `iacPath`, `workspace`, `vars`; Outputs: plan diff, apply status.

### What to Call [infra-provisioner]

- Identifier: `infra-provisioner` • Aliases: `terraform-agent`, `pulumi-agent`.
- CLI: `agentctl run infra-provisioner --iac .\infra --workspace prod --varFile .\prod.tfvars`
- API: `POST /agents/v1/infra-provisioner/plan|apply`

### When to Call

- On environment creation/updates; drift remediation; disaster recovery.

### Version Compatibility

- Terraform 1.6+, Pulumi 3.0+; Azure/AWS/GCP providers current.

### Security Considerations

- Scoped cloud roles; remote state encryption; approval for applies.

### Examples

```powershell
agentctl run infra-provisioner --iac .\infra --workspace prod --action plan --varFile .\prod.tfvars
```

```json
POST /agents/v1/infra-provisioner/apply
{"iacPath":"infra","workspace":"prod","vars":{"instance_size":"B2ms"}}
```

---

## 9) Configuration Management Agent

### Smart Generate Agent [DESCRIPTION]

- Applies declarative config (Ansible/Chef) to VMs/services; idempotent runs.
- Inputs: `playbook`, `inventory`, `vars`; Outputs: change set, compliance report.

### What to Call [config-manager]

- Identifier: `config-manager` • Aliases: `ansible-agent`.
- CLI: `agentctl run config-manager --playbook .\site.yml --inventory .\hosts.ini`
- API: `POST /agents/v1/config-manager/run`

### When to Call

- Baseline setup; patching; post-provision configuration.

### Version Compatibility

- Ansible 2.16+, WinRM/SSH on Windows/Linux targets.

### Security Considerations

- Vault for vars; strict SSH keys; minimize privilege escalation.

### Examples

```powershell
agentctl run config-manager --playbook .\site.yml --inventory .\hosts.ini --tags baseline
```

```json
POST /agents/v1/config-manager/run
{"playbook":"site.yml","inventory":"hosts.ini","vars":{"role":"web"}}
```

---

## 10) Kubernetes Deploy Agent

### Smart Generate Agent [DESCRIPTION]

- Applies manifests/Helm charts to clusters, validates health probes/rollouts.
- Inputs: `clusterRef`, `manifests|chart`, `values`; Outputs: rollout status, kubectl logs.

### What to Call [k8s-deploy]

- Identifier: `k8s-deploy` • Aliases: `helm-deploy`.
- CLI: `agentctl run k8s-deploy --cluster prod --chart .\charts\app --values .\values.prod.yaml`
- API: `POST /agents/v1/k8s-deploy/run`

### When to Call

- After container build/scan; release to staging/prod.

### Version Compatibility

- Kubernetes 1.27+; Helm 3.14+; Windows client; Linux nodes.

### Security Considerations

- RBAC-scoped service accounts; admission policies; signed charts.

### Examples

```powershell
agentctl run k8s-deploy --cluster prod --chart .\charts\app --values .\values.prod.yaml --namespace app
```

```json
POST /agents/v1/k8s-deploy/run
{"clusterRef":"prod","chartPath":"charts/app","valuesFile":"values.prod.yaml","namespace":"app"}
```

---

## 11) Blue-Green/Canary Rollout Agent

### Smart Generate Agent [DESCRIPTION]

- Performs safe deployment strategies (blue-green, canary with weighted traffic).
- Inputs: `strategy`, `serviceRef`, `weights`, `checks`; Outputs: promotion or rollback status.

### What to Call [progressive-delivery]

- Identifier: `progressive-delivery` • Aliases: `canary`, `blue-green`.
- CLI: `agentctl run progressive-delivery --service app --strategy canary --weights 10,30,60`
- API: `POST /agents/v1/progressive-delivery/run`

### When to Call

- After deploy manifests applied; pre-production traffic ramp.

### Version Compatibility

- Works with Kubernetes/Service Mesh (Istio/Linkerd), NGINX/ALB routing.

### Security Considerations

- Guardrails: SLO thresholds, error budgets; approve promotions.

### Examples

```powershell
agentctl run progressive-delivery --service app --strategy blue-green --checks .\checks.yml
```

```json
POST /agents/v1/progressive-delivery/run
{"serviceRef":"app","strategy":"canary","weights":[10,30,60],"checks":{"maxErrorRate":0.01}}
```

---

## 12) Auto Rollback Agent

### Smart Generate Agent [DESCRIPTION]

- Monitors post-deploy health; triggers rollback on SLO violation or errors.
- Inputs: `deploymentId`, `rollbackPlan`, `monitors`; Outputs: rollback status, incident link.

### What to Call [rollback]

- Identifier: `rollback` • Aliases: `revert`, `undo-deploy`.
- CLI: `agentctl run rollback --deploymentId 20251130-1234 --plan .\rollback.yml`
- API: `POST /agents/v1/rollback/run`

### When to Call

- On alert spikes, failed readiness, canary failure; automated or manual.

### Version Compatibility

- Compatible with k8s, VM rollouts, feature flags.

### Security Considerations

- Access limited to deploy scopes; audit all actions.

### Examples

```powershell
agentctl run rollback --deploymentId 20251130-1234 --plan .\rollback.yml --confirm
```

```json
POST /agents/v1/rollback/run
{"deploymentId":"20251130-1234","rollbackPlan":"rollback.yml"}
```

---

## 13) Observability Metrics Agent

### Smart Generate Agent [DESCRIPTION]

- Scrapes/pushes metrics (Prometheus/OpenTelemetry), sets alerts; dashboards.
- Inputs: `targets`, `otelConfig`; Outputs: time-series, alert states.

### What to Call [metrics]

- Identifier: `metrics` • Aliases: `telemetry`.
- CLI: `agentctl run metrics --targets .\scrape.yml --otel .\otel.yaml`
- API: `POST /agents/v1/metrics/run`

### When to Call

- Platform bootstrap; post-deploy verification; capacity planning.

### Version Compatibility

- Prometheus 2.50+, OTel Collector 0.103+.

### Security Considerations

- TLS/mTLS scraping; rate limits; PII-free metrics.

### Examples

```powershell
agentctl run metrics --targets .\scrape.yml --otel .\otel.yaml --env prod
```

```json
POST /agents/v1/metrics/run
{"targetsFile":"scrape.yml","otelConfig":"otel.yaml","env":"prod"}
```

---

## 14) Log Shipping Agent

### Smart Generate Agent [DESCRIPTION]

- Collects, parses, and ships logs (Fluent Bit/Vector) with schema enforcement.
- Inputs: `pipeline`, `sinks`, `parsers`; Outputs: structured logs, ingestion metrics.

### What to Call [log-shipper]

- Identifier: `log-shipper` • Aliases: `fluent`, `vector`.
- CLI: `agentctl run log-shipper --pipeline .\logs\pipeline.toml`
- API: `POST /agents/v1/log-shipper/run`

### When to Call

- At service boot; during migrations; audit trails.

### Version Compatibility

- Fluent Bit 3+, Vector 0.38+.

### Security Considerations

- Redact secrets; TLS to sinks; storage retention policies.

### Examples

```powershell
agentctl run log-shipper --pipeline .\logs\pipeline.toml --sink elastic://logs/prod
```

```json
POST /agents/v1/log-shipper/run
{"pipeline":"logs/pipeline.toml","sinks":["elastic://logs/prod"]}
```

---

## 15) Alert Routing Agent

### Smart Generate Agent [DESCRIPTION]

- Routes alerts to on-call (PagerDuty/Opsgenie/MS Teams), dedup and escalation.
- Inputs: `rules`, `receivers`, `schedules`; Outputs: incidents, acknowledgements.

### What to Call [alert-router]

- Identifier: `alert-router` • Aliases: `pager`, `notify`.
- CLI: `agentctl run alert-router --rules .\alerts\routes.yml`
- API: `POST /agents/v1/alert-router/run`

### When to Call

- After metrics/logs setup; in all environments.

### Version Compatibility

- Alertmanager 0.27+, PagerDuty v3 API, Opsgenie v2.

### Security Considerations

- Secrets for webhooks; suppression windows; avoid alert storms.

### Examples

```powershell
agentctl run alert-router --rules .\alerts\routes.yml --receivers .\alerts\receivers.yml
```

```json
POST /agents/v1/alert-router/run
{"rulesFile":"alerts/routes.yml","receiversFile":"alerts/receivers.yml"}
```

---

## 16) Incident Triage Agent

### Smart Generate Agent [DESCRIPTION]

- Classifies incidents, suggests runbooks, correlates signals, opens tickets.
- Inputs: `eventStream`, `runbooks`, `priorityRules`; Outputs: ticket IDs, triage notes.

### What to Call [triage]

- Identifier: `triage` • Aliases: `incident-bot`.
- CLI: `agentctl run triage --events .\events\current.json --runbooks .\runbooks`
- API: `POST /agents/v1/triage/run`

### When to Call

- On alert creation; during major incidents; postmortems.

### Version Compatibility

- Works with Jira/ServiceNow APIs; webhook integrations.

### Security Considerations

- Restrict PII; audit ticket changes; minimal scopes.

### Examples

```powershell
agentctl run triage --events .\events\current.json --runbooks .\runbooks --priorityRules .\priority.yml
```

```json
POST /agents/v1/triage/run
{"eventStream":"events/current.json","runbooksDir":"runbooks","priorityRules":"priority.yml"}
```

---

## 17) Release Manager Agent

### Smart Generate Agent [DESCRIPTION]

- Assembles release notes, tags, change logs; coordinates approvals.
- Inputs: `commitsRange`, `artifacts`, `templates`; Outputs: release record, tag.

### What to Call [release-manager]

- Identifier: `release-manager` • Aliases: `releaser`.
- CLI: `agentctl run release-manager --from v1.2.2 --to HEAD --notes .\templates\release.md`
- API: `POST /agents/v1/release-manager/run`

### When to Call

- After successful staging/prod deploy; at sprint end.

### Version Compatibility

- Git 2.44+; GitHub/GitLab release APIs.

### Security Considerations

- Signed tags; reviewer approvals; immutable records.

### Examples

```powershell
agentctl run release-manager --from v1.2.2 --to HEAD --publish
```

```json
POST /agents/v1/release-manager/run
{"from":"v1.2.2","to":"HEAD","publish":true}
```

---

## 18) Feature Flag Agent

### Smart Generate Agent [DESCRIPTION]

- Manages runtime flags, user cohorts, gradual rollouts; kill-switch support.
- Inputs: `flagConfig`, `targets`, `schedule`; Outputs: flag states, audit log.

### What to Call [feature-flags]

- Identifier: `feature-flags` • Aliases: `toggle`, `ff`.
- CLI: `agentctl run feature-flags --config .\flags\prod.yaml --enable checkout_v2`
- API: `POST /agents/v1/feature-flags/run`

### When to Call

- During progressive delivery; rollback via disabling flags.

### Version Compatibility

- LaunchDarkly v2 API, OpenFeature SDKs.

### Security Considerations

- Scoped SDK keys; no PII in targeting rules.

### Examples

```powershell
agentctl run feature-flags --config .\flags\prod.yaml --enable checkout_v2 --percentage 25
```

```json
POST /agents/v1/feature-flags/run
{"config":"flags/prod.yaml","changes":[{"flag":"checkout_v2","percentage":25}]}
```

---

## 19) Cost Monitoring Agent

### Smart Generate Agent [DESCRIPTION]

- Ingests cloud billing, maps to workloads, anomaly detection, budgets.
- Inputs: `cloudAccounts`, `budgets`, `tags`; Outputs: reports, alerts, recommendations.

### What to Call [cost-monitor]

- Identifier: `cost-monitor` • Aliases: `finops`.
- CLI: `agentctl run cost-monitor --accounts .\cloud.json --budgets .\budgets.yml`
- API: `POST /agents/v1/cost-monitor/run`

### When to Call

- Nightly; after scaling events; pre-approval for heavy jobs.

### Version Compatibility

- AWS CE API, Azure Cost Management, GCP Billing.

### Security Considerations

- Read-only billing roles; anonymize usage; limit data retention.

### Examples

```powershell
agentctl run cost-monitor --accounts .\cloud.json --budgets .\budgets.yml --notify
```

```json
POST /agents/v1/cost-monitor/run
{"cloudAccounts":"cloud.json","budgetsFile":"budgets.yml","notify":true}
```

---

## 20) Compliance Audit Agent

### Smart Generate Agent [DESCRIPTION]

- Checks infra/app against policies (CIS, SOC2), produces evidence bundles.
- Inputs: `policies`, `scope`, `exceptions`; Outputs: pass/fail, evidence artifacts.

### What to Call [compliance-audit]

- Identifier: `compliance-audit` • Aliases: `policy-check`.
- CLI: `agentctl run compliance-audit --policies .\policies --scope prod`
- API: `POST /agents/v1/compliance-audit/run`

### When to Call

- Pre-release audits; quarterly reviews; on major infra changes.

### Version Compatibility

- Open Policy Agent (OPA) 0.64+, Conftest 0.53+.

### Security Considerations

- Immutable evidence storage; redact sensitive configs; reviewer sign-off.

### Examples

```powershell
agentctl run compliance-audit --policies .\policies --scope prod --exceptions .\exceptions.yml
```

```json
POST /agents/v1/compliance-audit/run
{"policiesDir":"policies","scope":"prod","exceptionsFile":"exceptions.yml"}
```

---

## Cross-Agent Integration Patterns

- Chaining: Build → Test → Container Build → Image Scan → K8s Deploy → Progressive Delivery → Metrics/Logs → Alert Router → Release Manager.
- Error gates: Image Scan or Compliance Audit failure triggers Auto Rollback and Incident Triage.
- Secrets: All agents consuming credentials call `secrets` with short TTL.
- Evidence: Artifact Repo stores SBOMs, test reports, compliance evidence.

## Global Version Matrix

| Component       | Version                                 |
| --------------- | --------------------------------------- |
| Windows runners | Windows Server 2019/2022, PowerShell 7+ |
| Container stack | Docker 24+, containerd 1.7+, OCI v1     |
| Kubernetes      | 1.27–1.30; Helm 3.14+                   |
| IaC             | Terraform 1.6+; Pulumi 3+               |
| Observability   | Prometheus 2.50+, OTel 0.103+           |

## Security Baselines

- Least-privilege access per agent, scoped tokens, mandatory audit logging.
- No secrets in source or images; strict redaction and masked logs.
- Signed artifacts/images/manifests; SBOM generation and storage.
- Policy-as-code enforced via Compliance Audit and admission controls.

## Agent Index

| #   | Agent                       | Identifier              | Aliases                           |
| --- | --------------------------- | ----------------------- | --------------------------------- |
| 1   | Pipeline Orchestrator       | `pipeline-orchestrator` | `workflow`, `ci-pipeline`         |
| 2   | Build                       | `build`                 | `compiler`, `builder`             |
| 3   | Test Runner                 | `test-runner`           | `qa`, `ci-tests`                  |
| 4   | Artifact Repository         | `artifact-repo`         | `publisher`, `registry-client`    |
| 5   | Container Builder           | `container-build`       | `image-builder`                   |
| 6   | Image Vulnerability Scanner | `image-scan`            | `vuln-scan`                       |
| 7   | Secret Management           | `secrets`               | `vault-client`, `kms`             |
| 8   | Infrastructure Provisioner  | `infra-provisioner`     | `terraform-agent`, `pulumi-agent` |
| 9   | Configuration Management    | `config-manager`        | `ansible-agent`                   |
| 10  | Kubernetes Deploy           | `k8s-deploy`            | `helm-deploy`                     |
| 11  | Blue-Green/Canary Rollout   | `progressive-delivery`  | `canary`, `blue-green`            |
| 12  | Auto Rollback               | `rollback`              | `revert`, `undo-deploy`           |
| 13  | Observability Metrics       | `metrics`               | `telemetry`                       |
| 14  | Log Shipping                | `log-shipper`           | `fluent`, `vector`                |
| 15  | Alert Routing               | `alert-router`          | `pager`, `notify`                 |
| 16  | Incident Triage             | `triage`                | `incident-bot`                    |
| 17  | Release Manager             | `release-manager`       | `releaser`                        |
| 18  | Feature Flag                | `feature-flags`         | `toggle`, `ff`                    |
| 19  | Cost Monitoring             | `cost-monitor`          | `finops`                          |
| 20  | Compliance Audit            | `compliance-audit`      | `policy-check`                    |
