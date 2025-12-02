# MetaHub Python Scripts

Organized Python automation modules for governance, orchestration, and DevOps.

## Module Structure

```
scripts/
├── ai/                 # AI auditing and verification
│   ├── ai_audit.py
│   ├── agent_mcp_integrator.py
│   └── hallucination_verifier.py
│
├── compliance/         # Compliance and validation
│   ├── compliance_validator.py
│   ├── enforce.py
│   └── structure_validator.py
│
├── integration/        # External integrations
│   ├── mcp_cli_wrapper.py
│   └── mcp_server_tester.py
│
├── monitoring/         # Telemetry and dashboards
│   └── telemetry_dashboard.py
│
├── orchestration/      # Workflow orchestration
│   ├── checkpoint.py
│   ├── orchestration_checkpoint.py
│   ├── orchestration_telemetry.py
│   ├── orchestration_validator.py
│   └── self_healing_workflow.py
│
├── setup/              # Repository and org setup
│   ├── create_github_repos.py
│   ├── push_monorepos.py
│   ├── setup_org.py
│   └── setup_repo_ci.py
│
├── utils/              # Shared utilities
│   ├── catalog.py
│   ├── meta.py
│   └── sync_governance.py
│
├── workflows/          # Workflow runners
│   ├── devops_workflow_runner.py
│   └── quick_start.py
│
└── requirements.txt    # Python dependencies
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run compliance validation
python -m scripts.compliance.compliance_validator

# Run AI audit
python -m scripts.ai.ai_audit

# Test MCP servers
python -m scripts.integration.mcp_server_tester
```

## Module Categories

| Category | Purpose |
|----------|---------|
| `ai/` | AI tool auditing, hallucination detection, MCP integration |
| `compliance/` | Policy enforcement, structure validation |
| `integration/` | External tool integration (MCP, APIs) |
| `monitoring/` | Telemetry collection and visualization |
| `orchestration/` | Workflow checkpoints, self-healing |
| `setup/` | Repository and organization provisioning |
| `utils/` | Shared utilities and catalog management |
| `workflows/` | Workflow execution and quick-start |
