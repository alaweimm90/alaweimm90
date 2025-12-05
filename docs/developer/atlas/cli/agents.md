# DevOps Agents Catalog

This page catalogs 20 essential DevOps agents used across CI/CD, infrastructure, and operations.

Conventions:

- CLI: `agentctl run <agent> [options]`
- API (generic): `POST /agents/v1/<agent>/run` with JSON body per agent

---

## 1) pipeline-orchestrator

- Orchestrates multi-stage CI/CD DAGs, matrices, approvals; artifacts and logs.
- Perf/limits: parallel to runner cores; long stages block queues.
- When: on PRs/merges/tags; requires valid pipeline YAML and runner capacity.
- Compat/security: GitHub/GitLab/Azure/Jenkins; tokens least-privilege.

```powershell
agentctl run pipeline-orchestrator --pipeline .\ci\main.yml --env prod
```

## 2) build

- Compiles and packages artifacts with caching.
- Perf/limits: incremental cache; cache bust on lockfile change.
- When: after checkout; toolchains installed.
- Compat/security: Node/Java/.NET/Go; pin deps, sandbox.

```powershell
agentctl run build --spec .\buildspec.yml --src . --out .\dist
```

## 3) test-runner

- Runs unit/integration/e2e; coverage; flaky detection.
- Perf/limits: sharding by runtime; retry flakies.
- When: after build and before deploy; fixtures ready.
- Compat/security: Jest/JUnit/NUnit/Cypress; no real secrets.

```powershell
agentctl run test-runner --spec .\tests.yml --coverage
```

## 4) artifact-repo

- Publishes/retrieves artifacts and checksums.
- Perf/limits: throughput per repo quotas.
- When: post-build/test; credentials valid.
- Compat/security: Maven/NPM/NuGet/Docker; signed, immutable.

```powershell
agentctl run artifact-repo --push .\dist\service.exe --repo s3://artifacts/prod
```

## 5) container-build

- Builds OCI images; multi-stage; SBOM.
- Perf/limits: BuildKit cache; network-bound pulls.
- When: after build; registry access.
- Compat/security: Docker 24+/containerd; non-root, minimal base.

```powershell
agentctl run container-build --file .\Dockerfile --tag registry/app:1.2.3
```

## 6) image-scan

- Scans images/deps for CVEs; policy gate.
- Perf/limits: DB freshness; scan time by image size.
- When: pre-deploy gate; nightly.
- Compat/security: Trivy/Grype; SBOM-linked findings.

```powershell
agentctl run image-scan --image registry/app:1.2.3 --policy "maxSeverity=High"
```

## 7) secrets

- Fetches/rotates/injects secrets with TTL.
- Perf/limits: rate limits; short-lived tokens.
- When: before steps needing creds.
- Compat/security: Vault/KeyVault/KMS; redact logs.

```powershell
agentctl run secrets --ref kv://apps/prod --ttl 900
```

## 8) infra-provisioner

- Plans/applies IaC; drift detection.
- Perf/limits: provider API rate limits; state locks.
- When: env create/update; DR.
- Compat/security: Terraform/Pulumi; remote encrypted state.

```powershell
agentctl run infra-provisioner --iac .\infra --workspace prod --action plan
```

## 9) config-manager

- Applies declarative configs to VMs/services; idempotent.
- Perf/limits: host concurrency; WinRM/SSH latency.
- When: baseline setup; patching.
- Compat/security: Ansible; vault vars; minimal sudo.

```powershell
agentctl run config-manager --playbook .\site.yml --inventory .\hosts.ini
```

## 10) k8s-deploy

- Applies manifests/Helm; validates rollouts.
- Perf/limits: API server QPS; rollout timeout.
- When: release to clusters.
- Compat/security: K8s 1.27+; RBAC-scoped SA; signed charts.

```powershell
agentctl run k8s-deploy --cluster prod --chart .\charts\app --values .\values.prod.yaml
```

## 11) progressive-delivery

- Blue-green/canary with traffic weights and health checks.
- Perf/limits: mesh/ingress propagation delays.
- When: post-deploy traffic ramp.
- Compat/security: Istio/Linkerd/ALB; SLO guardrails.

```powershell
agentctl run progressive-delivery --service app --strategy canary --weights 10,30,60
```

## 12) rollback

- Auto/manual rollback on SLO violations/errors.
- Perf/limits: stateful migrations may need custom plan.
- When: health fails/canary fails.
- Compat/security: k8s/VM/flags; audited actions.

```powershell
agentctl run rollback --deploymentId 20251130-1234 --plan .\rollback.yml
```

## 13) metrics

- Scrapes/pushes metrics (Prom/OTel); alerts.
- Perf/limits: scrape intervals vs load.
- When: bootstrap; post-deploy; capacity.
- Compat/security: Prometheus/OTel; TLS/mTLS; PII-free.

```powershell
agentctl run metrics --targets .\scrape.yml --otel .\otel.yaml
```

## 14) log-shipper

- Parses and ships logs; schema enforcement.
- Perf/limits: backpressure handling; buffer sizes.
- When: service boot; audits.
- Compat/security: Fluent Bit/Vector; redaction; TLS.

```powershell
agentctl run log-shipper --pipeline .\logs\pipeline.toml
```

## 15) alert-router

- Dedups/escalates alerts to on-call.
- Perf/limits: anti-storm rules; rate limits.
- When: after obs setup.
- Compat/security: Alertmanager/PagerDuty/Opsgenie; secure webhooks.

```powershell
agentctl run alert-router --rules .\alerts\routes.yml
```

## 16) triage

- Classifies incidents; runbooks; tickets.
- Perf/limits: correlation accuracy vs noise.
- When: on alert creation/major incident.
- Compat/security: Jira/ServiceNow; PII minimization.

```powershell
agentctl run triage --events .\events\current.json --runbooks .\runbooks
```

## 17) release-manager

- Tags/releases; notes; approvals.
- Perf/limits: API quotas.
- When: after staging/prod deploy; sprint end.
- Compat/security: Git/GitHub/GitLab; signed tags.

```powershell
agentctl run release-manager --from v1.2.2 --to HEAD --publish
```

## 18) feature-flags

- Manages runtime flags/cohorts; kill-switch.
- Perf/limits: SDK cache warmup; targeting size.
- When: progressive delivery/quick rollback.
- Compat/security: OpenFeature/LaunchDarkly; scoped keys.

```powershell
agentctl run feature-flags --config .\flags\prod.yaml --enable checkout_v2 --percentage 25
```

## 19) cost-monitor

- Ingests billing; anomalies; budgets.
- Perf/limits: provider data latency.
- When: nightly; pre heavy jobs; post scaling.
- Compat/security: AWS/Azure/GCP billing; read-only roles.

```powershell
agentctl run cost-monitor --accounts .\cloud.json --budgets .\budgets.yml
```

## 20) compliance-audit

- Checks policies (CIS/SOC2); evidence bundles.
- Perf/limits: scan scope vs runtime.
- When: pre-release; quarterly; infra changes.
- Compat/security: OPA/Conftest; immutable evidence.

```powershell
agentctl run compliance-audit --policies .\policies --scope prod
```

---

See `docs/ai-tools/*.md` for deep dives on orchestrator, security, compliance, telemetry, etc.

---

## Detailed Reference

### 1) Pipeline Orchestrator Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: DAG scheduler with stage dependencies, matrices, approvals, retries, artifact handoff; supports conditional execution and concurrency controls.
- Core functionality: CI/CD workflow orchestration, gated releases, fan-out/in testing, environment promotion.
- Inputs: `pipelineFile`, `env`, `parameters`, `secretsRef`.
- Outputs: pipeline status, stage logs, artifact index.
- Performance/limits: parallelism bounded by runner cores; long-running stages can block queues; caching advised.

#### What to Call [pipeline-orchestrator]

- Identifier: `pipeline-orchestrator`; aliases: `workflow`, `ci-pipeline`.
- CLI:

```powershell
agentctl run pipeline-orchestrator --pipeline .\ci\main.yml --env prod --params .\ci\params.prod.json
```

- API:

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

#### When to Call [short description]

- Scenarios: PRs, merges to `main`, nightly builds, gated production releases.
- Preconditions: SCM reachable, valid YAML, runner capacity.
- Error auto-invoke: retries on transient failures, stage-level re-runs.
- Integrations: Build, Test Runner, Container Builder, Image Scanner, Notifications.

#### Version Compatibility

- GitHub Actions/GitLab/Azure DevOps/Jenkins; Windows Server 2019+, PowerShell 7+.

#### Security Considerations

- Least-privilege tokens; redact env; approve external workflow usage.

#### Practical Example

```powershell
agentctl run pipeline-orchestrator --pipeline .\ci\release.yml --env prod --params .\ci\release.params.json
```

---

### 2) Build Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: compiles source, resolves dependencies, performs caching, produces artifacts with checksums.
- Core functionality: language-specific builds (Node/Java/.NET/Go), incremental builds.
- Inputs: `sourcePath`, `buildSpec`, `cacheKey`.
- Outputs: artifact IDs, build logs.
- Performance/limits: incremental caching accelerates; cache invalidates on lockfile changes.

#### What to Call [build]

- Identifier: `build`; aliases: `compiler`, `builder`.
- CLI:

```powershell
agentctl run build --spec .\buildspec.yml --src . --out .\dist --cacheKey app-v1
```

- API:

```http
POST /agents/v1/build/run
{
  "sourcePath": ".",
  "buildSpec": "buildspec.yml",
  "cacheKey": "app-v1"
}
```

#### When to Call [short description]

- Scenarios: post-checkout, pre-test, on tag creation.
- Preconditions: toolchain installed, network for dependencies.
- Error auto-invoke: dependency resolution failures trigger help output; can re-run with `--no-cache`.
- Integrations: Artifact Repo, Test Runner.

#### Version Compatibility

- Node 20+/npm 10+, Java 17+, .NET 8, Go 1.22+.

#### Security Considerations

- Pin dependencies, verify checksums; sandbox builds.

#### Practical Example

```powershell
agentctl run build --spec .\buildspec.yml --src . --out .\dist
```

---

### 3) Test Runner Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: executes unit/integration/e2e suites, collects coverage, detects flaky tests via retries and historical timing.
- Core functionality: sharding, parallelization, report aggregation.
- Inputs: `testSpec`, `env`, `shard`.
- Outputs: JUnit XML, coverage reports, pass/fail matrix.
- Performance/limits: optimal with balanced shards; heavy e2e limited by environment readiness.

#### What to Call [test-runner]

- Identifier: `test-runner`; aliases: `qa`, `ci-tests`.
- CLI:

```powershell
agentctl run test-runner --spec .\tests.yml --env staging --coverage
```

- API:

```http
POST /agents/v1/test-runner/run
{
  "testSpec": "tests.yml",
  "env": "staging",
  "shard": "auto"
}
```

#### When to Call [short description]

- Scenarios: after build; before deploy; PR updates.
- Preconditions: fixtures/data ready; mocks configured.
- Error auto-invoke: flaky detection triggers targeted retries.
- Integrations: Pipeline Orchestrator, Metrics.

#### Version Compatibility

- Jest 29+, JUnit 5, NUnit 3, Cypress 13+.

#### Security Considerations

- Avoid real secrets; isolate test data; mask logs.

#### Practical Example

```powershell
agentctl run test-runner --spec .\tests.yml --report .\reports
```

---

### 4) Artifact Repository Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: publishes/retrieves artifacts to registries (Nexus/Artifactory/S3); handles metadata and checksums.
- Core functionality: versioning, immutability, provenance.
- Inputs: `artifactPath`, `repoRef`, `metadata`.
- Outputs: immutable URL, checksum.
- Performance/limits: throughput bound by repo quotas and network.

#### What to Call [artifact-repo]

- Identifier: `artifact-repo`; aliases: `publisher`, `registry-client`.
- CLI:

```powershell
agentctl run artifact-repo --push .\dist\service.exe --repo s3://artifacts/prod
```

- API:

```http
POST /agents/v1/artifact-repo/push
{
  "artifactPath": "dist/service.exe",
  "repoRef": "s3://artifacts/prod",
  "metadata": {"version": "1.2.3"}
}
```

#### When to Call [short description]

- Scenarios: after successful build/test; before deploy.
- Preconditions: credentials valid; sufficient quota.
- Error auto-invoke: checksum mismatch triggers re-upload.
- Integrations: Container Builder, Release Manager.

#### Version Compatibility

- Maven/NPM/NuGet/Docker registries.

#### Security Considerations

- Signed artifacts; scoped tokens; immutability enforced.

#### Practical Example

```powershell
agentctl run artifact-repo --push .\dist\app.zip --repo nuget://internal
```

---

### 5) Container Builder Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: builds OCI images using BuildKit; multi-stage; SBOM emission.
- Core functionality: image tagging, cache use, platform targets.
- Inputs: `dockerfile`, `context`, `tags`.
- Outputs: image digest, SBOM.
- Performance/limits: cache speeds up; large contexts slow I/O.

#### What to Call [container-build]

- Identifier: `container-build`; alias: `image-builder`.
- CLI:

```powershell
agentctl run container-build --file .\Dockerfile --context . --tag registry.local/mysvc:1.2.3
```

- API:

```http
POST /agents/v1/container-build/run
{
  "dockerfile": "Dockerfile",
  "context": ".",
  "tags": ["registry.local/mysvc:1.2.3"]
}
```

#### When to Call [short description]

- Scenarios: post-build; pre-scan; pre-deploy.
- Preconditions: Docker/Buildx available; registry access.
- Error auto-invoke: cache miss triggers full rebuild.
- Integrations: Image Scanner, K8s Deploy.

#### Version Compatibility

- OCI v1; Docker 24+, containerd 1.7+.

#### Security Considerations

- Minimal base images; non-root user; secrets not baked; SBOM stored.

#### Practical Example

```powershell
agentctl run container-build --file .\Dockerfile --tag registry/app:1.2.3
```

---

### 6) Image Vulnerability Scanner Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: scans images and dependencies for CVEs with policy gates; uses CVE DBs and SBOM correlation.
- Core functionality: report generation, severity gating, remediation hints.
- Inputs: `imageRef`, `policy`.
- Outputs: scan report, pass/fail status.
- Performance/limits: scan time scales with image size; DB freshness required.

#### What to Call [image-scan]

- Identifier: `image-scan`; alias: `vuln-scan`.
- CLI:

```powershell
agentctl run image-scan --image registry.local/mysvc:1.2.3 --policy "maxSeverity=High"
```

- API:

```http
POST /agents/v1/image-scan/run
{
  "imageRef": "registry.local/mysvc:1.2.3",
  "policy": {"maxSeverity": "High"}
}
```

#### When to Call [short description]

- Scenarios: pre-deploy gate; nightly base image scan.
- Preconditions: registry reachable; SBOM available.
- Error auto-invoke: failed gate triggers rollback agent.
- Integrations: Container Builder, Compliance Audit.

#### Version Compatibility

- Trivy 0.50+, Grype 0.76+.

#### Security Considerations

- Maintain CVE DB freshness; managed allowlists; store evidence.

#### Practical Example

```powershell
agentctl run image-scan --image registry/app:1.2.3 --policy high-only
```

---

### 7) Secret Management Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: fetches, rotates, and injects secrets from Vault/KeyVault/KMS with TTL.
- Core functionality: ephemeral env files, template bindings, rotation orchestration.
- Inputs: `secretsRef`, `bindings`, `ttl`.
- Outputs: injection status, rotation status.
- Performance/limits: rate limits and token TTLs; caching discouraged.

#### What to Call [secrets]

- Identifier: `secrets`; aliases: `vault-client`, `kms`.
- CLI:

```powershell
agentctl run secrets --ref kv://apps/prod --bind .\bindings.yml --ttl 900
```

- API:

```http
POST /agents/v1/secrets/fetch
{
  "secretsRef": "kv://apps/prod",
  "bindings": {"DB_PASSWORD": "env:DB_PASSWORD"},
  "ttl": 900
}
```

#### When to Call [short description]

- Scenarios: before any step requiring credentials; rotation windows.
- Preconditions: authenticated client; binding templates valid.
- Error auto-invoke: expired tokens trigger re-fetch.
- Integrations: Build, Test, Deploy agents.

#### Version Compatibility

- Vault 1.15+, Azure Key Vault v7+, AWS KMS.

#### Security Considerations

- Least privileges; audit; redact logs; short TTLs.

#### Practical Example

```powershell
agentctl run secrets --ref kv://ci/prod --ttl 600
```

---

### 8) Infrastructure Provisioner Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: plans/applies IaC (Terraform/Pulumi); drift detection; state locks.
- Core functionality: workspace management, variable injection, plan diffs.
- Inputs: `iacPath`, `workspace`, `vars`, `action`.
- Outputs: plan/apply status, diff report.
- Performance/limits: provider API rate limits; lock contention.

#### What to Call [infra-provisioner]

- Identifier: `infra-provisioner`; aliases: `terraform-agent`, `pulumi-agent`.
- CLI:

```powershell
agentctl run infra-provisioner --iac .\infra --workspace prod --action plan --varFile .\prod.tfvars
```

- API:

```http
POST /agents/v1/infra-provisioner/apply
{
  "iacPath": "infra",
  "workspace": "prod",
  "vars": {"instance_size": "B2ms"}
}
```

#### When to Call [short description]

- Scenarios: environment creation/update; drift remediation; DR restore.
- Preconditions: cloud credentials; remote state configured.
- Error auto-invoke: drift detection can trigger plan.
- Integrations: Config Manager, Compliance Audit.

#### Version Compatibility

- Terraform 1.6+, Pulumi 3+; Azure/AWS/GCP providers.

#### Security Considerations

- Encrypted remote state; scoped roles; approval for applies.

#### Practical Example

```powershell
agentctl run infra-provisioner --iac .\infra --workspace prod --action apply --varFile .\prod.tfvars
```

---

### 9) Configuration Management Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: idempotent configuration application (Ansible) across hosts/services.
- Core functionality: inventory targeting, role/tag selection, change reports.
- Inputs: `playbook`, `inventory`, `vars`, `tags`.
- Outputs: change set, compliance report.
- Performance/limits: concurrency limited by hosts/transport; WinRM latency on Windows.

#### What to Call [config-manager]

- Identifier: `config-manager`; alias: `ansible-agent`.
- CLI:

```powershell
agentctl run config-manager --playbook .\site.yml --inventory .\hosts.ini --tags baseline
```

- API:

```http
POST /agents/v1/config-manager/run
{
  "playbook": "site.yml",
  "inventory": "hosts.ini",
  "vars": {"role": "web"}
}
```

#### When to Call [short description]

- Scenarios: baseline setup; patching; post-provision.
- Preconditions: connectivity; credentials; idempotent tasks.
- Error auto-invoke: failed host tasks can re-run with `--limit`.
- Integrations: Infra Provisioner, Compliance Audit.

#### Version Compatibility

- Ansible 2.16+; SSH/WinRM targets.

#### Security Considerations

- Secrets via vault; minimize privilege escalation; audit changes.

#### Practical Example

```powershell
agentctl run config-manager --playbook .\site.yml --inventory .\hosts.ini --tags hardening
```

---

### 10) Kubernetes Deploy Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: applies manifests or Helm charts; monitors rollout health; supports strategic/rolling updates.
- Core functionality: namespace targeting, values overrides, health probes validation.
- Inputs: `clusterRef`, `manifests|chart`, `values`, `namespace`.
- Outputs: rollout status, resource diffs, logs.
- Performance/limits: API server QPS; rollout timeouts.

#### What to Call [k8s-deploy]

- Identifier: `k8s-deploy`; alias: `helm-deploy`.
- CLI:

```powershell
agentctl run k8s-deploy --cluster prod --chart .\charts\app --values .\values.prod.yaml --namespace app
```

- API:

```http
POST /agents/v1/k8s-deploy/run
{
  "clusterRef": "prod",
  "chartPath": "charts/app",
  "valuesFile": "values.prod.yaml",
  "namespace": "app"
}
```

#### When to Call [short description]

- Scenarios: deploy to staging/prod; upgrades; hotfixes.
- Preconditions: cluster access; image available; configs valid.
- Error auto-invoke: failed health triggers rollback.
- Integrations: Progressive Delivery, Metrics/Logs.

#### Version Compatibility

- Kubernetes 1.27+; Helm 3.14+.

#### Security Considerations

- RBAC-scoped service accounts; admission policies; signed charts.

#### Practical Example

```powershell
agentctl run k8s-deploy --cluster prod --manifests .\k8s\app --namespace app
```

---

### 11) Blue-Green/Canary Rollout Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: controls traffic shifting (weights/headers) during phased rollout; evaluates health checks.
- Core functionality: canary steps, promotion/abort, blue-green switching.
- Inputs: `strategy`, `serviceRef`, `weights`, `checks`.
- Outputs: promotion or rollback status, metrics snapshot.
- Performance/limits: mesh/ingress propagation delays.

#### What to Call [progressive-delivery]

- Identifier: `progressive-delivery`; aliases: `canary`, `blue-green`.
- CLI:

```powershell
agentctl run progressive-delivery --service app --strategy canary --weights 10,30,60 --checks .\checks.yml
```

- API:

```http
POST /agents/v1/progressive-delivery/run
{
  "serviceRef": "app",
  "strategy": "canary",
  "weights": [10,30,60],
  "checks": {"maxErrorRate": 0.01}
}
```

#### When to Call [short description]

- Scenarios: post-deploy traffic ramp; risky changes.
- Preconditions: metrics/logs configured; baselines known.
- Error auto-invoke: health threshold breach triggers rollback.
- Integrations: K8s Deploy, Metrics, Alert Router.

#### Version Compatibility

- Istio/Linkerd; NGINX/ALB routing.

#### Security Considerations

- Enforce SLO guardrails; approvals for promotions.

#### Practical Example

```powershell
agentctl run progressive-delivery --service payments --strategy blue-green
```

---

### 12) Auto Rollback Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: monitors post-deploy health and triggers rollback according to SLOs or error signals.
- Core functionality: rollback execution, incident linking, evidence capture.
- Inputs: `deploymentId`, `rollbackPlan`, `monitors`.
- Outputs: rollback status, incident link.
- Performance/limits: stateful migrations may require custom rollback.

#### What to Call [rollback]

- Identifier: `rollback`; aliases: `revert`, `undo-deploy`.
- CLI:

```powershell
agentctl run rollback --deploymentId 20251130-1234 --plan .\rollback.yml --confirm
```

- API:

```http
POST /agents/v1/rollback/run
{
  "deploymentId": "20251130-1234",
  "rollbackPlan": "rollback.yml"
}
```

#### When to Call [short description]

- Scenarios: alert spikes, failed readiness, canary failures.
- Preconditions: rollback plan available.
- Error auto-invoke: exceeded thresholds automatically invoke rollback.
- Integrations: Progressive Delivery, Triage.

#### Version Compatibility

- k8s/VM rollouts; feature flags.

#### Security Considerations

- Access scoped to deploy objects; audited actions.

#### Practical Example

```powershell
agentctl run rollback --deploymentId $(Get-Date -Format yyyyMMdd) --plan .\rollback.yml
```

---

### 13) Observability Metrics Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: scrapes or receives OpenTelemetry metrics; sets alerts; exports dashboards.
- Core functionality: target discovery, exporters, alert rules.
- Inputs: `targets`, `otelConfig`.
- Outputs: time-series data, alert states.
- Performance/limits: scrape interval vs load; exporter backpressure.

#### What to Call [metrics]

- Identifier: `metrics`; alias: `telemetry`.
- CLI:

```powershell
agentctl run metrics --targets .\scrape.yml --otel .\otel.yaml --env prod
```

- API:

```http
POST /agents/v1/metrics/run
{
  "targetsFile": "scrape.yml",
  "otelConfig": "otel.yaml",
  "env": "prod"
}
```

#### When to Call [short description]

- Scenarios: platform bootstrap; post-deploy verification; capacity planning.
- Preconditions: scrape configs valid; endpoints reachable.
- Error auto-invoke: target failures trigger alerts.
- Integrations: Alert Router, Dashboard.

#### Version Compatibility

- Prometheus 2.50+; OTel Collector 0.103+.

#### Security Considerations

- TLS/mTLS; rate limits; PII-free metrics.

#### Practical Example

```powershell
agentctl run metrics --targets .\scrape.yml --otel .\otel.yaml --env dev
```

---

### 14) Log Shipping Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: collects, parses, and ships logs; enforces schemas; backpressure handling.
- Core functionality: pipelines, sinks, parsers, redaction.
- Inputs: `pipeline`, `sinks`, `parsers`.
- Outputs: structured logs, ingestion metrics.
- Performance/limits: buffer sizes and throughput affect latency.

#### What to Call [log-shipper]

- Identifier: `log-shipper`; aliases: `fluent`, `vector`.
- CLI:

```powershell
agentctl run log-shipper --pipeline .\logs\pipeline.toml --sink elastic://logs/prod
```

- API:

```http
POST /agents/v1/log-shipper/run
{
  "pipeline": "logs/pipeline.toml",
  "sinks": ["elastic://logs/prod"]
}
```

#### When to Call [short description]

- Scenarios: service boot; audit trails; migration windows.
- Preconditions: sink availability; schema defined.
- Error auto-invoke: ingestion failures route to dead-letter.
- Integrations: Alert Router, Triage.

#### Version Compatibility

- Fluent Bit 3+; Vector 0.38+.

#### Security Considerations

- Redaction; TLS to sinks; retention controls.

#### Practical Example

```powershell
agentctl run log-shipper --pipeline .\logs\pipeline.toml
```

---

### 15) Alert Routing Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: routes alerts to on-call systems, deduplicates, escalates; integrates with schedules.
- Core functionality: rules, receivers, silences, escalation policies.
- Inputs: `rules`, `receivers`, `schedules`.
- Outputs: incidents, acknowledgements.
- Performance/limits: anti-storm and rate limits protect receivers.

#### What to Call [alert-router]

- Identifier: `alert-router`; aliases: `pager`, `notify`.
- CLI:

```powershell
agentctl run alert-router --rules .\alerts\routes.yml --receivers .\alerts\receivers.yml
```

- API:

```http
POST /agents/v1/alert-router/run
{
  "rulesFile": "alerts/routes.yml",
  "receiversFile": "alerts/receivers.yml"
}
```

#### When to Call [short description]

- Scenarios: after observability setup; continuous operation.
- Preconditions: receiver credentials; schedules configured.
- Error auto-invoke: dead-letter routing for failed notifications.
- Integrations: Metrics, Logs, Triage.

#### Version Compatibility

- Alertmanager 0.27+; PagerDuty v3; Opsgenie v2.

#### Security Considerations

- Secure webhooks; secret storage; suppression windows.

#### Practical Example

```powershell
agentctl run alert-router --rules .\alerts\routes.yml
```

---

### 16) Incident Triage Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: classifies incidents, correlates signals, suggests runbooks, opens tickets.
- Core functionality: priority rules, enrichment, knowledge base linking.
- Inputs: `eventStream`, `runbooks`, `priorityRules`.
- Outputs: ticket IDs, triage notes, suggested actions.
- Performance/limits: correlation accuracy vs noise; requires clean signals.

#### What to Call [triage]

- Identifier: `triage`; alias: `incident-bot`.
- CLI:

```powershell
agentctl run triage --events .\events\current.json --runbooks .\runbooks --priorityRules .\priority.yml
```

- API:

```http
POST /agents/v1/triage/run
{
  "eventStream": "events/current.json",
  "runbooksDir": "runbooks",
  "priorityRules": "priority.yml"
}
```

#### When to Call [short description]

- Scenarios: on alert creation; during major incidents.
- Preconditions: integration to ticketing; runbooks available.
- Error auto-invoke: failed escalation triggers fallback receiver.
- Integrations: Alert Router, Release Manager.

#### Version Compatibility

- Jira/ServiceNow APIs.

#### Security Considerations

- Restrict PII; audit ticket changes; minimal scopes.

#### Practical Example

```powershell
agentctl run triage --events .\events\current.json --runbooks .\runbooks
```

---

### 17) Release Manager Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: assembles release notes, tags, changelogs; coordinates approvals and publishing.
- Core functionality: commit range analysis, artifact linkage, provenance.
- Inputs: `commitsRange`, `artifacts`, `templates`, `publish`.
- Outputs: release record, tag, notes.
- Performance/limits: SCM API quotas apply.

#### What to Call [release-manager]

- Identifier: `release-manager`; alias: `releaser`.
- CLI:

```powershell
agentctl run release-manager --from v1.2.2 --to HEAD --notes .\templates\release.md --publish
```

- API:

```http
POST /agents/v1/release-manager/run
{
  "from": "v1.2.2",
  "to": "HEAD",
  "publish": true
}
```

#### When to Call [short description]

- Scenarios: after successful staging/prod deploy; sprint end.
- Preconditions: artifacts present; approvals ready.
- Error auto-invoke: failed publish retries with backoff.
- Integrations: Artifact Repo, Compliance Audit.

#### Version Compatibility

- Git 2.44+; GitHub/GitLab release APIs.

#### Security Considerations

- Signed tags; reviewer approvals; immutable records.

#### Practical Example

```powershell
agentctl run release-manager --from v1.2.2 --to HEAD --publish
```

---

### 18) Feature Flag Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: manages runtime flags, cohorts, gradual rollouts; kill-switch.
- Core functionality: percentage rollout, targeting rules, audit.
- Inputs: `flagConfig`, `targets`, `schedule`.
- Outputs: flag states, audit log.
- Performance/limits: SDK cache warmup; large targeting lists.

#### What to Call [feature-flags]

- Identifier: `feature-flags`; aliases: `toggle`, `ff`.
- CLI:

```powershell
agentctl run feature-flags --config .\flags\prod.yaml --enable checkout_v2 --percentage 25
```

- API:

```http
POST /agents/v1/feature-flags/run
{
  "config": "flags/prod.yaml",
  "changes": [{"flag": "checkout_v2", "percentage": 25}]
}
```

#### When to Call [short description]

- Scenarios: progressive delivery; fast rollback via kill-switch.
- Preconditions: SDK keys; targeting rules validated.
- Error auto-invoke: mis-targeting reverts to default.
- Integrations: Progressive Delivery, Metrics.

#### Version Compatibility

- LaunchDarkly v2 API; OpenFeature SDKs.

#### Security Considerations

- Scoped SDK keys; avoid PII in targeting.

#### Practical Example

```powershell
agentctl run feature-flags --config .\flags\prod.yaml --disable legacy_checkout
```

---

### 19) Cost Monitoring Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: ingests cloud billing, maps to workloads via tags, detects anomalies, enforces budgets.
- Core functionality: reports, alerts, recommendations.
- Inputs: `cloudAccounts`, `budgets`, `tags`.
- Outputs: cost reports, anomaly alerts.
- Performance/limits: provider data latency and granularity constraints.

#### What to Call [cost-monitor]

- Identifier: `cost-monitor`; alias: `finops`.
- CLI:

```powershell
agentctl run cost-monitor --accounts .\cloud.json --budgets .\budgets.yml --notify
```

- API:

```http
POST /agents/v1/cost-monitor/run
{
  "cloudAccounts": "cloud.json",
  "budgetsFile": "budgets.yml",
  "notify": true
}
```

#### When to Call [short description]

- Scenarios: nightly; pre heavy jobs; post scaling.
- Preconditions: read-only billing roles; tags applied.
- Error auto-invoke: budget breach triggers alert routing.
- Integrations: Alert Router, Release Manager.

#### Version Compatibility

- AWS CE API; Azure Cost Management; GCP Billing.

#### Security Considerations

- Read-only roles; anonymize usage; retention limits.

#### Practical Example

```powershell
agentctl run cost-monitor --accounts .\cloud.json --budgets .\budgets.yml
```

---

### 20) Compliance Audit Agent

#### Smart Generate Agent [DESCRIPTION]

- Technical: checks infra/app against policies (CIS/SOC2) using OPA/Conftest; produces evidence bundles.
- Core functionality: policy evaluation, exceptions management, evidence storage.
- Inputs: `policies`, `scope`, `exceptions`.
- Outputs: pass/fail, detailed report, evidence artifacts.
- Performance/limits: runtime scales with scope; policy complexity impacts speed.

#### What to Call [compliance-audit]

- Identifier: `compliance-audit`; alias: `policy-check`.
- CLI:

```powershell
agentctl run compliance-audit --policies .\policies --scope prod --exceptions .\exceptions.yml
```

- API:

```http
POST /agents/v1/compliance-audit/run
{
  "policiesDir": "policies",
  "scope": "prod",
  "exceptionsFile": "exceptions.yml"
}
```

#### When to Call [short description]

- Scenarios: pre-release audits; quarterly reviews; major infra changes.
- Preconditions: policies loaded; exception lists reviewed.
- Error auto-invoke: policy failure gates deployments.
- Integrations: Infra Provisioner, K8s Deploy, Release Manager.

#### Version Compatibility

- OPA 0.64+; Conftest 0.53+.

#### Security Considerations

- Immutable evidence storage; redact sensitive configs; reviewer sign-off.

#### Practical Example

```powershell
agentctl run compliance-audit --policies .\policies --scope prod
```

---

## Cross-Agent Patterns

- Build → Test → Container Build → Image Scan → K8s Deploy → Progressive Delivery → Metrics/Logs → Alert Router → Release Manager.
- Fail gates (Image Scan/Compliance) auto-invoke Rollback and Triage.
- Secrets injected just-in-time across agents; evidence stored in Artifact Repo.
