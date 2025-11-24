# Agent Orchestration Overview

## Agent Registration & Configuration

- Agents are initialized by `agent-orchestrator` on framework start.
- Inventory: run `automation list agents` or task `agent-orchestrator:agent-inventory`.
- Configure capabilities/limitations via module code or capability matrix parameter on task assignment.

## Suggestion Approval & Overrides

- Automatic approval uses RBAC and security scans.
- Blocked approvals produce `awaiting-override` executions.
- Override with task `agent-orchestrator:override-approval` by an `admin` role.

## Execution Visibility & Audit

- Real-time status via WebSocket on `ws://localhost:7070`.
- All events are recorded under `.automation/logs/audit/`.

## Monitoring & Alerting

- Unified monitoring emits periodic `agents_monitoring` events.
- Integrate logs into existing Grafana/Loki stack.

## Performance Metrics

- Execution times captured per task; agent metrics include load and success rates.
- Use CI to aggregate trends and alert on thresholds.
## Agentic Orchestration Usage

- Task routing:
  - `automation execute "agent-orchestrator:assign-task" "{ \"task\": { \"name\": \"api:sync\" }, \"requirements\": [\"api-integration\"], \"priority\": 2 }"`
- Workflow:
  - `automation execute "agent-orchestrator:orchestrate-workflow" "{ \"workflow\": { \"name\": \"data-sync\", \"steps\": [ { \"name\": \"process\", \"requirements\": [\"data-processing\"], \"task\": { \"name\": \"process\" } }, { \"name\": \"push\", \"requirements\": [\"api-integration\"], \"task\": { \"name\": \"push\" }, \"dependsOn\": [\"process\"] } ] } }"`
- Scheduled tasks:
  - `automation execute "scheduler:schedule" "{ \"name\": \"nightly\", \"intervalMs\": 3600000, \"task\": { \"name\": \"process\", \"requirements\": [\"data-processing\"] } }"`
- Human-in-the-loop:
  - Approvals blocked are returned with `status: awaiting-override`; override via `agent-orchestrator:override-approval` using an `admin` role.
- Persistent state:
  - Workflow/execution snapshots under `.automation/logs/state/store.jsonl`; query via `fw.stateStore.listWorkflows()` programmatically.
