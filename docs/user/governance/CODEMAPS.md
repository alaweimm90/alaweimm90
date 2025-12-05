# Architectural Codemaps

> Visual trace diagrams for the governance, orchestration, and automation systems.

**Generated:** 2025-12-04  
**Systems Covered:** 6  
**Total Traces:** 24

---

## Table of Contents

1. [Unified Automation System](#1-unified-automation-system)
2. [MCP Integration & CLI Wrapper](#2-mcp-integration--cli-wrapper)
3. [Orchestration Governance](#3-orchestration-governance)
4. [MetaHub Governance System](#4-metahub-governance-system)
5. [DevOps Template Library](#5-devops-template-library)
6. [ORCHEX Workflow Orchestration](#6-ORCHEX-workflow-orchestration)

---

## 1. Unified Automation System

**Files:** `automation/core/AutomationCore.ts`, `automation/cli/index.ts`, `automation/core/utils.ts`

### Trace 1.1: AutomationCore Initialization

```text
AutomationCore Initialization Flow
├── constructor() <-- AutomationCore.ts:20
│   └── initialize() <-- AutomationCore.ts:27
│       ├── loadAgents() <-- AutomationCore.ts:37
│       │   └── readYamlFile(agents.yaml) <-- AutomationCore.ts:42
│       │       └── Map<string, Agent> populated
│       ├── loadWorkflows() <-- AutomationCore.ts:54
│       │   └── readYamlFile(workflows.yaml) <-- AutomationCore.ts:59
│       │       └── Map<string, Workflow> populated
│       ├── loadPrompts() <-- AutomationCore.ts:71
│       │   └── listFilesRecursive(prompts/) <-- AutomationCore.ts:83
│       │       └── Prompt[] populated (system/project/tasks)
│       └── loadPatterns() <-- AutomationCore.ts:31
│           └── Anthropic patterns loaded
```

| Location               | Line          | Description                          |
| ---------------------- | ------------- | ------------------------------------ |
| `AutomationCore.ts:20` | constructor   | Entry point for automation system    |
| `AutomationCore.ts:37` | loadAgents    | Loads 24 agents from YAML config     |
| `AutomationCore.ts:54` | loadWorkflows | Loads 11 workflows from YAML         |
| `AutomationCore.ts:71` | loadPrompts   | Scans 49 prompts across 3 categories |

### Trace 1.2: Task Routing Flow

```
Task Routing Flow (CLI → Core → Result)
├── CLI Entry Point <-- cli/index.ts:19
│   └── program.command('route <task>') <-- cli/index.ts:15
│       └── action handler invoked
│           ├── new AutomationCore() <-- cli/index.ts:21
│           └── core.routeTask(task) <-- cli/index.ts:22
│               └── routeTask() <-- core/utils.ts
│                   ├── Extract keywords from task
│                   ├── Score against task_types
│                   │   ├── architecture keywords
│                   │   ├── implementation keywords
│                   │   ├── debugging keywords
│                   │   └── testing keywords
│                   ├── Select highest scoring type
│                   └── Return RouteResult
│                       ├── task_type: string
│                       ├── confidence: number
│                       ├── recommended_tools: string[]
│                       └── suggested_agents: string[]
```

| Location          | Line          | Description                       |
| ----------------- | ------------- | --------------------------------- |
| `cli/index.ts:19` | route command | CLI entry for task routing        |
| `core/utils.ts`   | routeTask     | Keyword-based task classification |

### Trace 1.3: Workflow Execution

```
Workflow Execution Flow
├── CLI Entry Point <-- cli/index.ts:44
│   └── program.command('execute <workflow>')
│       └── action handler
│           ├── Parse JSON input options <-- cli/index.ts:53
│           └── unifiedExecutor(workflow, opts, inputs) <-- cli/index.ts:66
│               └── ExecutionContext created
│                   ├── startTime: Date
│                   ├── status: ExecutionStatus
│                   ├── dryRun: boolean
│                   └── outputs: Map
│                       └── Execute workflow steps
│                           └── Return completed context
```

---

## 2. MCP Integration & CLI Wrapper

**Files:** `tools/cli/mcp.py`, `.metaHub/scripts/integration/mcp_cli_wrapper.py`

### Trace 2.1: MCP Server Discovery

```
MCP Server Discovery Flow
├── CLI Entry Point <-- mcp.py:main()
│   └── argparse routes to 'list' command
│       └── cmd_list(args, manager) <-- mcp.py
│           └── MCPManager.list_servers() <-- mcp.py
│               ├── Load server-registry.yaml
│               │   └── .ai/mcp/server-registry.yaml
│               ├── Parse server definitions
│               │   └── for name, config in servers:
│               │       └── MCPServerInfo(
│               │           name, command, args,
│               │           description, capabilities, tags
│               │       )
│               └── Return List[MCPServerInfo]
```

| Location            | Line          | Description                            |
| ------------------- | ------------- | -------------------------------------- |
| `mcp.py:main`       | CLI entry     | Parses commands and routes to handlers |
| `mcp.py:MCPManager` | Manager class | Handles server operations              |

### Trace 2.2: Server Health Check (Ping)

```
Server Ping Flow
├── cmd_ping(args, manager) <-- mcp.py
│   └── manager.ping_server(server_name)
│       ├── Get server config from registry
│       ├── Build command array
│       │   └── [command] + args
│       ├── Execute subprocess with timeout
│       │   └── subprocess.run(cmd, timeout=10)
│       ├── Calculate latency
│       │   └── end_time - start_time
│       ├── Determine status
│       │   ├── SUCCESS → ServerStatus.CONNECTED
│       │   ├── TIMEOUT → ServerStatus.ERROR
│       │   └── NOT_FOUND → ServerStatus.NOT_INSTALLED
│       └── Record telemetry event
│           └── Telemetry.record_event(
│               EventType.MCP_PING,
│               status, latency
│           )
```

### Trace 2.3: Tool Execution

```
MCP Tool Execution Flow
├── cmd_execute(args, manager) <-- mcp.py
│   └── manager.execute_tool(server, tool, arguments)
│       ├── Validate server exists
│       ├── Build mcp-cli command
│       │   └── ["mcp-cli", "cmd", "--server", server,
│       │        "--tool", tool, "--args", json.dumps(args)]
│       ├── Execute via subprocess
│       │   └── _run_mcp_command(cmd, timeout)
│       │       ├── subprocess.run(cmd, capture_output=True)
│       │       └── Return (success, stdout, stderr)
│       ├── Parse JSON output
│       └── Create ToolExecutionResult
│           ├── success: bool
│           ├── output: Any
│           ├── duration_ms: int
│           └── error: Optional[str]
```

### Trace 2.4: Agent-MCP Integration Report

```
Agent-MCP Integration Report Generation
├── cmd_integrate(args, manager) <-- mcp.py
│   └── AgentIntegrator.generate_report()
│       ├── Load agent frameworks
│       │   ├── get_meathead_physicist_mappings()
│       │   │   └── Scientist, Literature, Scout, Theory, Critic
│       │   ├── get_turingo_mappings()
│       │   │   └── QuantumQuokka, MLMagician, PuzzleProdigy
│       │   └── get_atlas_mappings()
│       │       └── Coordinator, Analyst, Synthesizer
│       ├── Map agents to MCP servers
│       │   └── for agent in agents:
│       │       └── AgentMCPMapping(
│       │           agent_name, role,
│       │           mcp_servers, capabilities, use_cases
│       │       )
│       ├── Calculate summary statistics
│       │   └── unique_mcps, total_mappings
│       └── Output report (JSON/Markdown)
```

---

## 3. Orchestration Governance

**Files:** `.metaHub/scripts/orchestration/orchestration_validator.py`, `orchestration_telemetry.py`, `orchestration_checkpoint.py`

### Trace 3.1: Handoff Envelope Validation

```
Handoff Envelope Validation Pipeline
├── CLI Entry Point
│   └── @click.command('validate')
│       └── validator.validate_envelope(envelope)
│           ├── Schema Validation Layer
│           │   └── _validate_schema(envelope)
│           │       └── jsonschema.validate(
│           │           envelope,
│           │           handoff-envelope-schema.json
│           │       )
│           ├── Policy Validation Layer
│           │   └── _validate_policy(envelope)
│           │       ├── Check mandatory fields
│           │       │   └── metadata, context, instructions
│           │       ├── Validate context size (max 500KB)
│           │       └── Verify timestamp format (ISO 8601)
│           └── Routing Validation Layer
│               └── _validate_routing(envelope)
│                   ├── Validate source_tool in VALID_TOOLS
│                   ├── Validate target_tool in VALID_TOOLS
│                   └── check_routing(source, target)
│                       └── Verify transition allowed
```

| Location                        | Line                   | Description              |
| ------------------------------- | ---------------------- | ------------------------ |
| `orchestration_validator.py:69` | OrchestrationValidator | Main validator class     |
| `orchestration_validator.py:76` | VALID_TOOLS            | Set of 14 valid AI tools |

### Trace 3.2: Task-to-Tool Routing

```
Task Routing with Intent Extraction
├── CLI Entry Point
│   └── @click.command('route')
│       └── validator.route_task(task_description)
│           ├── Normalize task
│           │   └── task_lower = task.lower()
│           ├── Load intent keywords from policy
│           │   └── orchestration-governance.yaml
│           │       ├── architecture: [design, architect, plan...]
│           │       ├── implementation: [build, implement, create...]
│           │       ├── debugging: [fix, debug, error...]
│           │       └── testing: [test, coverage, spec...]
│           ├── Score each task type
│           │   └── for task_type, keywords in intent_config:
│           │       └── score = sum(kw in task_lower for kw in keywords)
│           ├── Select best matching type
│           │   └── best_type = max(scores, key=scores.get)
│           ├── Lookup recommended tools
│           │   └── routing_rules[best_type]["tools"]
│           └── Match agent frameworks
│               └── MeatheadPhysicist, Turingo, ORCHEX
```

### Trace 3.3: Telemetry Event Recording

```
Telemetry Event Recording and Aggregation
├── record_event(event_type, status, tool, duration)
│   └── TelemetryEvent created
│       ├── event_id: uuid
│       ├── timestamp: ISO 8601
│       ├── event_type: handoff|tool_invocation|hallucination
│       ├── status: success|failure|warning
│       ├── tool: string
│       └── duration_ms: int
│           └── Append to events.jsonl
│
├── generate_summary(period="24h")
│   ├── load_events() from JSONL
│   │   └── Filter by timestamp
│   ├── Aggregate by type/status/tool
│   ├── Calculate percentiles
│   │   └── P50, P95, P99 latencies
│   └── Compute success rates
│       ├── handoff_success_rate
│       ├── workflow_completion_rate
│       └── hallucination_rate
│
└── check_targets()
    ├── Load policy thresholds
    │   ├── handoff_success: ≥95%
    │   ├── hallucination_rate: ≤2%
    │   └── workflow_completion: ≥90%
    └── Compare actual vs target
```

### Trace 3.4: Checkpoint Creation

```
Checkpoint Creation Flow
├── create_checkpoint(workflow, tool, task, files)
│   ├── Generate checkpoint metadata
│   │   ├── checkpoint_id: uuid
│   │   ├── timestamp: ISO 8601
│   │   ├── workflow_name: string
│   │   └── tool: string
│   ├── Snapshot files
│   │   └── _snapshot_files(files)
│   │       └── for file in files:
│   │           ├── Read content
│   │           ├── Generate hash
│   │           └── Write to snapshot dir
│   ├── Build checkpoint object
│   │   └── OrchestrationCheckpoint(
│   │       id, timestamp, workflow,
│   │       tool, task, snapshots, checksum
│   │   )
│   ├── Generate integrity checksum
│   │   └── SHA256(checkpoint_data)
│   └── Save checkpoint
│       ├── Write JSON to checkpoints/
│       └── Update latest pointer
```

---

## 4. MetaHub Governance System

**Files:** `.metaHub/scripts/compliance/enforce.py`, `.metaHub/scripts/checkpoint.py`, `.metaHub/scripts/catalog.py`, `.metaHub/scripts/meta.py`

### Trace 4.1: Policy Enforcement

```
Policy Enforcement Flow
├── CLI Entry Point
│   └── @click.command()
│       └── enforce_organization(org_path)
│           └── for repo_dir in org_path:
│               └── PolicyEnforcer(repo_dir)
│                   ├── _load_schema()
│                   │   └── repo-schema.json
│                   ├── check_all()
│                   │   ├── check_metadata()
│                   │   │   ├── Validate .meta/repo.yaml exists
│                   │   │   └── jsonschema.validate(metadata, schema)
│                   │   ├── check_repo_structure()
│                   │   │   └── Verify tier-based required files
│                   │   │       ├── Tier 1: metadata, readme, license,
│                   │   │       │          ci, codeowners, tests
│                   │   │       ├── Tier 2: metadata, readme, license,
│                   │   │       │          ci, codeowners
│                   │   │       └── Tier 3-4: metadata, readme
│                   │   └── check_docker()
│                   │       ├── Non-root USER check
│                   │       ├── HEALTHCHECK present
│                   │       ├── No :latest tags
│                   │       └── No secrets in ENV
│                   └── Return (violations, warnings)
```

| Location        | Line                     | Description                        |
| --------------- | ------------------------ | ---------------------------------- |
| `enforce.py:30` | PolicyEnforcer           | Main enforcement class             |
| `enforce.py:34` | TIER_REQUIREMENTS        | Tier-based file requirements       |
| `enforce.py:59` | DOCKER_SECURITY_PATTERNS | Regex patterns for Docker security |

### Trace 4.2: Drift Detection (Checkpoint)

```
Drift Detection Flow
├── CheckpointManager(base_path)
│   └── generate_current_state()
│       └── for org_dir in organizations/:
│           └── for repo_dir in org_dir:
│               └── _analyze_repo(repo_dir)
│                   ├── Load .meta/repo.yaml
│                   ├── Determine tier
│                   ├── Check compliance items
│                   │   ├── metadata exists
│                   │   ├── readme exists
│                   │   ├── license exists
│                   │   ├── ci workflows exist
│                   │   ├── codeowners exists
│                   │   └── tests directory exists
│                   └── Generate file hash
│
├── load_previous_checkpoint()
│   └── Read checkpoint-latest.json
│
└── detect_drift()
    ├── Compare repo sets
    │   ├── new_repos = current - previous
    │   └── deleted_repos = previous - current
    ├── For common repos:
    │   ├── Compare file hashes
    │   │   └── changed if hash differs
    │   └── Compare compliance status
    │       ├── improved: was non-compliant, now compliant
    │       └── degraded: was compliant, now non-compliant
    └── Generate drift report
```

### Trace 4.3: Service Catalog Generation

```
Service Catalog Generation Flow
├── CatalogBuilder(base_path)
│   └── scan_organizations()
│       └── for org_dir in organizations/:
│           └── for repo_dir in org_dir:
│               └── _scan_repository(repo_dir)
│                   ├── Load .meta/repo.yaml
│                   ├── Infer type from name prefix
│                   │   ├── lib- → library
│                   │   ├── svc- → service
│                   │   ├── app- → application
│                   │   └── tool- → tool
│                   ├── Detect language
│                   │   ├── pyproject.toml → python
│                   │   ├── tsconfig.json → typescript
│                   │   ├── go.mod → go
│                   │   └── Cargo.toml → rust
│                   ├── Check compliance
│                   │   ├── has_readme
│                   │   ├── has_ci
│                   │   └── has_tests
│                   └── Return repo_data
│
├── Aggregate statistics
│   ├── by_language: Dict[str, int]
│   ├── by_type: Dict[str, int]
│   ├── by_tier: Dict[int, int]
│   └── compliance_rate: float
│
└── Generate output
    ├── generate_json(output_path)
    ├── generate_markdown(output_path)
    └── generate_html(output_path)
```

### Trace 4.4: Project Audit & Promotion

```
Meta Auditor: Project Audit & Promotion
├── MetaAuditor.scan_all_projects(org_filter)
│   └── for org_dir in organizations/:
│       └── for project_dir in org_dir:
│           └── _audit_project(project_dir, org)
│               ├── Check metadata gaps
│               │   └── Missing .meta/repo.yaml → P0
│               ├── Check documentation gaps
│               │   ├── Missing README.md → P0
│               │   └── Short README → P2
│               ├── Check legal gaps
│               │   └── Missing LICENSE → P1/P3 (by tier)
│               ├── Check ownership gaps
│               │   └── Missing CODEOWNERS → P1
│               ├── Check CI gaps
│               │   └── Missing workflows → P1
│               └── Calculate compliance score
│                   └── 100 - (P0*30 + P1*15 + P2*5 + P3*2)
│
└── ProjectPromoter.promote_project(path, org)
    ├── Create .meta/repo.yaml
    │   └── type, language, tier, status
    ├── Create .github/CODEOWNERS
    │   └── * @{organization}
    ├── Create .github/workflows/ci.yml
    │   └── Language-specific workflow
    └── Create .pre-commit-config.yaml
```

---

## 5. DevOps Template Library

**Files:** `tools/cli/devops.ts`, `tools/cli/devops-generators.ts`, `tools/lib/fs.ts`

### Trace 5.1: Template Discovery

```
Template Discovery Flow
├── CLI Entry Point <-- devops.ts
│   └── program.command('template list')
│       └── cmdTemplateList(options)
│           └── discoverTemplates()
│               ├── findManifests(TEMPLATES_DIR)
│               │   └── fs.readdirSync(templates/devops/)
│               │       └── Filter for template.json, manifest.json
│               └── for manifestPath in manifests:
│                   └── readJson<TemplateManifest>(path)
│                       └── DiscoveredTemplate {
│                           manifest, manifestPath, templateDir
│                       }
│
└── filterTemplates(templates, search)
    └── Filter by name, description, tags
```

| Location       | Line              | Description                  |
| -------------- | ----------------- | ---------------------------- |
| `devops.ts:43` | discoverTemplates | Scans for template manifests |
| `devops.ts:64` | filterTemplates   | Filters by search pattern    |

### Trace 5.2: Template Application

```
Template Application Flow
├── cmdTemplateApply(name, options)
│   ├── discoverTemplates()
│   │   └── Find template by name
│   ├── parsePlaceholders(argv)
│   │   └── Extract KEY=VALUE pairs
│   ├── validateTemplate(manifest, dir)
│   │   └── Check required files exist
│   ├── copyTemplateFiles(manifest, src, dest, vars)
│   │   └── for file in manifest.files:
│   │       ├── Read source file
│   │       ├── replaceVars(content, vars)
│   │       │   └── Replace {{PLACEHOLDER}} patterns
│   │       └── Write to destination
│   └── writeTemplateMeta(dest, manifest)
│       └── Write .template-meta.json
```

### Trace 5.3: Node.js Service Generation

```
Node.js Service Code Generation
├── cmdGenerate(type, options)
│   └── switch(type)
│       └── case 'node-service':
│           └── planNodeService(vars)
│               ├── generateServiceSource(name)
│               │   └── HTTP server with /health endpoint
│               ├── generatePackageJson(name)
│               │   └── Dependencies, scripts
│               ├── generateDockerfile()
│               │   └── Multi-stage build
│               ├── generateCiWorkflow()
│               │   └── GitHub Actions for build/test
│               ├── generateK8sDeployment(name, registry, tag)
│               │   └── Deployment manifest with probes
│               ├── generateK8sService()
│               │   └── Service manifest
│               ├── generateHelmChart(name)
│               │   └── Chart.yaml
│               ├── generateHelmValues()
│               │   └── values.yaml
│               └── generatePrometheusConfig()
│                   └── Scrape config
│
└── applyChanges(files, targetDir)
    └── for file in files:
        ├── ensureDir(dirname)
        └── fs.writeFileSync(path, content)
```

---

## 6. ORCHEX Workflow Orchestration

**Files:** `tools/orchex/orchestration/devops-agents.ts`, `tools/orchex/cli/commands/devops.ts`

### Trace 6.1: DevOps Agent Registry

```
DevOps Agent Registry Loading
├── loadDevOpsAgents(configPath?)
│   ├── Read devops-agents.yaml
│   │   └── automation/agents/config/devops-agents.yaml
│   ├── Parse YAML content
│   │   └── yaml.load(content)
│   ├── Build category lookup
│   │   └── for [category, data] in categories:
│   │       └── categoryLookup.set(agentName, category)
│   └── Register agents
│       └── for [name, agentData] in agents:
│           └── DevOpsAgent {
│               id, role, goal, aliases,
│               tools, inputs, outputs, category
│           }
│               ├── devOpsAgentRegistry.set(id, agent)
│               └── for alias in aliases:
│                   └── aliasMap.set(alias, id)
```

| Location              | Line                | Description                |
| --------------------- | ------------------- | -------------------------- |
| `devops-agents.ts:16` | DevOpsAgentId       | Union type of 20 agent IDs |
| `devops-agents.ts:78` | devOpsAgentRegistry | Map of agent definitions   |
| `devops-agents.ts:84` | loadDevOpsAgents    | Loads agents from YAML     |

### Trace 6.2: Workflow Execution with DAG

```
DevOps Workflow Execution
├── executeDevOpsWorkflow(workflow, cbConfig)
│   ├── Initialize tracking
│   │   ├── pending: Set<step>
│   │   ├── completed: Set<stepId>
│   │   ├── failed: string[]
│   │   └── stepResults: Map<stepId, result>
│   │
│   ├── DAG Execution Loop
│   │   └── while pending.size > 0:
│   │       ├── Find ready steps
│   │       │   └── filter(step =>
│   │       │       step.dependsOn.every(d => completed.has(d))
│   │       │   )
│   │       ├── Execute ready steps
│   │       │   └── for step in ready:
│   │       │       └── executeDevOpsStep(step, cbConfig)
│   │       │           ├── Check circuit breaker
│   │       │           │   └── allowRequest(step.agentId)
│   │       │           ├── Execute with retry
│   │       │           │   └── retry up to maxRetries
│   │       │           ├── Record result
│   │       │           │   ├── recordSuccess() or
│   │       │           │   └── recordFailure()
│   │       │           └── Handle error strategy
│   │       │               ├── 'continue' → mark failed, continue
│   │       │               ├── 'abort' → throw
│   │       │               └── 'retry' → retry with backoff
│   │       └── Update tracking
│   │           ├── completed.add(step.id) if success
│   │           └── failed.push(step.id) if failure
│   │
│   └── Return DevOpsExecutionResult
│       ├── workflowName
│       ├── success: failed.length === 0
│       ├── stepResults
│       ├── totalDuration
│       └── failedSteps
```

### Trace 6.3: CLI Command Registration

```
ORCHEX DevOps CLI Registration
├── registerDevOpsCommands(program)
│   ├── loadDevOpsAgents()
│   │   └── Initialize agent registry
│   │
│   └── program.command('devops')
│       ├── .command('list')
│       │   └── List all DevOps agents
│       │
│       ├── .command('run <workflow>')
│       │   └── Execute workflow
│       │       ├── Parse workflow YAML
│       │       ├── executeDevOpsWorkflow(wf, config)
│       │       └── Display results
│       │
│       └── .command('agent <id>')
│           └── Show agent details
│               └── getDevOpsAgent(id)
```

---

## Cross-System Integration Points

### Integration Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED AUTOMATION SYSTEM                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ AutomationCore  │  │   CLI (TS)      │  │  Task Router    │  │
│  │ - 24 agents     │──│ - route         │──│ - keywords      │  │
│  │ - 11 workflows  │  │ - execute       │  │ - scoring       │  │
│  │ - 49 prompts    │  │ - list          │  │ - confidence    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌───────────────────────────────────────────────────────────────────┐
│                    MCP INTEGRATION LAYER                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ MCP CLI (Python)│  │ Server Registry │  │ Agent-MCP Map   │   │
│  │ - list          │──│ - 12 servers    │──│ - MeatheadPhys  │   │
│  │ - ping          │  │ - capabilities  │  │ - Turingo       │   │
│  │ - execute       │  │ - status        │  │ - ORCHEX         │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
└───────────┼─────────────────────┼─────────────────────┼───────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌───────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATION GOVERNANCE                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Validator       │  │ Telemetry       │  │ Checkpoint      │   │
│  │ - schema        │──│ - events.jsonl  │──│ - snapshots     │   │
│  │ - policy        │  │ - metrics       │  │ - drift detect  │   │
│  │ - routing       │  │ - targets       │  │ - recovery      │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
└───────────┼─────────────────────┼─────────────────────┼───────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌───────────────────────────────────────────────────────────────────┐
│                   METAHUB GOVERNANCE                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ PolicyEnforcer  │  │ CatalogBuilder  │  │ MetaAuditor     │   │
│  │ - tier rules    │──│ - scan orgs     │──│ - gap detection │   │
│  │ - docker sec    │  │ - statistics    │  │ - promotion     │   │
│  │ - schema valid  │  │ - multi-format  │  │ - scoring       │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
└───────────┼─────────────────────┼─────────────────────┼───────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌───────────────────────────────────────────────────────────────────┐
│                    DEVOPS & ORCHEX                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Template CLI    │  │ ORCHEX Agents    │  │ Workflow Engine │   │
│  │ - discover      │──│ - 20 DevOps     │──│ - DAG execution │   │
│  │ - apply         │  │ - registry      │  │ - circuit break │   │
│  │ - generate      │  │ - aliases       │  │ - retry logic   │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

---

## File Reference Index

| System                   | Key Files                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Unified Automation**   | `automation/core/AutomationCore.ts`, `automation/cli/index.ts`, `automation/core/utils.ts`                                            |
| **MCP Integration**      | `tools/cli/mcp.py`, `.metaHub/scripts/integration/mcp_cli_wrapper.py`                                                                 |
| **Orchestration**        | `.metaHub/scripts/orchestration/orchestration_validator.py`, `orchestration_telemetry.py`, `orchestration_checkpoint.py`              |
| **MetaHub Governance**   | `.metaHub/scripts/compliance/enforce.py`, `.metaHub/scripts/checkpoint.py`, `.metaHub/scripts/catalog.py`, `.metaHub/scripts/meta.py` |
| **DevOps Templates**     | `tools/cli/devops.ts`, `tools/cli/devops-generators.ts`, `tools/lib/fs.ts`                                                            |
| **ORCHEX Orchestration** | `tools/orchex/orchestration/devops-agents.ts`, `tools/orchex/cli/commands/devops.ts`                                                  |

---

## Related Documentation

- [AI-TOOLS-ORCHESTRATION.md](../AI-TOOLS-ORCHESTRATION.md) - Multi-agent orchestration guide
- [LLM-MODEL-CATALOG.md](LLM-MODEL-CATALOG.md) - 500+ LLM models across 8 IDEs
- [GOVERNANCE_SYSTEM.md](GOVERNANCE_SYSTEM.md) - Full governance system documentation

---

_Generated from codebase analysis. Update when significant architectural changes occur._
