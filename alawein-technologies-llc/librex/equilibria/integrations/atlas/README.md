# ORCHEX-Librex Integration

This integration layer enables MEZAN/ORCHEX to leverage Librex's optimization capabilities through a Redis-based asynchronous task management system that complies with the ORCHEX agent protocol.

## Overview

The ORCHEX-Librex integration provides:

- **Asynchronous Optimization**: Submit optimization tasks and retrieve results asynchronously
- **ORCHEX Agent Protocol**: Fully compliant with ORCHEX multi-agent orchestration
- **Redis Integration**: Share state via ORCHEX blackboard architecture
- **Method Selection**: AI-powered automatic optimization method selection
- **Task Management**: Priority queues, retries, and error handling
- **Batch Operations**: Submit multiple optimization tasks efficiently

## Architecture

```
ORCHEX Orchestration Engine
         │
         ├─── ORCHEX Agents (Research, Synthesis, etc.)
         │
         └─── LibrexAgent
                 │
                 ├─── ATLASOptimizationAdapter
                 │         │
                 │         ├─── Method Selector (AI)
                 │         └─── Domain Adapters (QAP, TSP, etc.)
                 │
                 └─── OptimizationTaskQueue (Redis)
                           │
                           └─── Librex Core
```

## Components

### 1. LibrexAgent

ORCHEX-compliant agent that exposes Librex as a research agent.

**Features:**
- Implements ORCHEX ResearchAgent interface
- Manages workload and task execution
- Reports to ORCHEX blackboard
- Provides feature vectors for Libria solvers

### 2. ATLASOptimizationAdapter

Main adapter for optimization services.

**Features:**
- Task submission and management
- Method recommendation via AI selector
- Result retrieval and callbacks
- Batch operations support

### 3. OptimizationTaskQueue

Redis-based asynchronous task queue.

**Features:**
- Priority-based task scheduling
- Task status tracking
- Retry logic with exponential backoff
- Result caching

### 4. ATLASConfig

Configuration management for the integration.

**Features:**
- Environment variable support
- Method-specific defaults
- Performance tuning options

## Installation

1. Install Librex with ORCHEX integration:

```bash
cd /home/user/AlaweinOS/Librex
pip install -e ".[ORCHEX]"
```

2. Install Redis (if not already installed):

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

3. Configure environment variables:

```bash
export ATLAS_REDIS_URL="redis://localhost:6379/0"
export Librex_AGENT_ID="Librex-agent-1"
export Librex_DEFAULT_METHOD="auto"  # Use AI selector
export Librex_MAX_TASKS="10"
export Librex_ENABLE_GPU="true"
```

## Usage

### From ORCHEX Agent

When implementing an ORCHEX agent that needs optimization:

```python
from atlas_core.agent import ResearchAgent
from Librex.integrations.orchex import ATLASOptimizationAdapter

class MyResearchAgent(ResearchAgent):
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = ATLASOptimizationAdapter()

    def execute(self, task):
        # Submit optimization task
        result = self.optimizer.submit_optimization_request(
            agent_id=self.config.agent_id,
            problem_type="qap",
            problem_data={
                "flow_matrix": flow_matrix,
                "distance_matrix": distance_matrix
            },
            method="auto",  # AI will select best method
            priority=1
        )

        task_id = result["task_id"]

        # Check status
        status = self.optimizer.get_task_status(task_id)

        # Result will be in Redis blackboard
        blackboard_key = result["redis_blackboard_key"]

        return {
            "optimization_task_id": task_id,
            "blackboard_key": blackboard_key,
            "status": status
        }
```

### Standalone Librex Agent

Deploy Librex as a standalone ORCHEX agent:

```python
from Librex.integrations.orchex import LibrexAgent, ATLASConfig

# Initialize agent
config = ATLASConfig.from_env()
agent = LibrexAgent(config)

# Register with ORCHEX
agent.register_with_atlas()

# Start background processor
agent.start_background_processor()

# Agent is now available for ORCHEX orchestration
# It will automatically process tasks from the queue

# When ready to shutdown
agent.stop_background_processor()
agent.deregister_from_atlas()
```

### Direct Task Submission

Submit tasks directly to the queue:

```python
from Librex.integrations.orchex import OptimizationTaskQueue, ATLASConfig

# Initialize queue
config = ATLASConfig.from_env()
queue = OptimizationTaskQueue(config)

# Submit task
task_id = queue.submit_task(
    problem_type="tsp",
    problem_data={
        "coordinates": [[0, 0], [1, 1], [2, 0], [1, -1]]
    },
    method="simulated_annealing",
    agent_id="my-agent",
    priority=2
)

# Check status
task = queue.get_task(task_id)
print(f"Task status: {task.status}")

# Get results when ready
if task.status == TaskStatus.COMPLETED:
    print(f"Solution: {task.result}")
```

### Method Recommendation

Get AI-powered method recommendations:

```python
from Librex.integrations.orchex import ATLASOptimizationAdapter

adapter = ATLASOptimizationAdapter()

recommendation = adapter.get_method_recommendation(
    agent_id="my-agent",
    problem_type="qap",
    problem_data=problem_data
)

print(f"Recommended method: {recommendation['recommendation']['method']}")
print(f"Confidence: {recommendation['recommendation']['confidence']:.2%}")
print(f"Explanation: {recommendation['recommendation']['explanation']}")
```

### Batch Operations

Submit multiple tasks at once:

```python
tasks = [
    {
        "problem_type": "qap",
        "problem_data": qap_data_1,
        "method": "genetic_algorithm",
        "priority": 1
    },
    {
        "problem_type": "tsp",
        "problem_data": tsp_data_2,
        "method": "auto",
        "priority": 2
    }
]

results = adapter.batch_submit("my-agent", tasks)

for result in results:
    print(f"Task {result['task_id']}: {result['status']}")
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ATLAS_REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `ATLAS_REDIS_DB` | Redis database number | `0` |
| `Librex_AGENT_ID` | Agent identifier | `Librex-agent` |
| `Librex_DEFAULT_METHOD` | Default optimization method | `auto` |
| `Librex_ENABLE_GPU` | Enable GPU acceleration | `true` |
| `Librex_MAX_TASKS` | Max concurrent tasks | `10` |
| `Librex_TASK_QUEUE` | Task queue name | `Librex:tasks` |

### Method-Specific Configurations

Configure optimization methods via `ATLASConfig`:

```python
config = ATLASConfig(
    method_configs={
        "simulated_annealing": {
            "initial_temperature": 1000,
            "cooling_rate": 0.95,
            "max_iterations": 10000
        },
        "genetic_algorithm": {
            "population_size": 200,
            "generations": 1000,
            "mutation_rate": 0.02
        }
    }
)
```

## Redis Schema

### Task Keys

- `Librex:task:{task_id}` - Task metadata and status
- `Librex:tasks` - Priority queue of pending tasks (sorted set)
- `Librex:results` - Queue of completed task IDs (list)
- `Librex:agent_tasks:{agent_id}` - Set of task IDs for an agent

### ORCHEX Blackboard Keys

- `ORCHEX:agent:{agent_id}` - Agent metadata and status
- `ORCHEX:blackboard:{agent_id}:Librex:{task_id}` - Task results
- `ORCHEX:blackboard:task:{task_id}:result` - ORCHEX task results
- `ORCHEX:available_agents` - Set of available agent IDs

### Events (Pub/Sub)

- `Librex:events:task_submitted` - Task submission events
- `Librex:events:task_status_updated` - Status change events
- `ORCHEX:events` - ORCHEX system events

## Monitoring

### Queue Statistics

```python
stats = queue.get_queue_stats()
print(f"Pending tasks: {stats['pending_tasks']}")
print(f"Results available: {stats['results_available']}")
print(f"Status breakdown: {stats['status_breakdown']}")
```

### Agent Status

```python
status = agent.get_agent_status()
print(f"Workload: {status['current_workload']}/{status['max_tasks']}")
print(f"Success rate: {status['recent_success_rate']:.2%}")
print(f"Average quality: {status['average_quality']:.2f}")
```

## Error Handling

The integration includes comprehensive error handling:

1. **Automatic Retries**: Failed tasks are automatically retried up to `max_retries` times
2. **Timeout Protection**: Tasks that exceed timeout are marked as failed
3. **Graceful Degradation**: Falls back to simulated annealing if AI selector fails
4. **Error Logging**: All errors are logged with context for debugging

## Performance Tuning

### Concurrency

Adjust `max_concurrent_tasks` based on available resources:

```python
config = ATLASConfig(
    max_concurrent_tasks=20  # For high-performance systems
)
```

### Caching

Enable result caching to avoid redundant computations:

```python
config = ATLASConfig(
    cache_results=True,
    cache_ttl=7200  # Cache for 2 hours
)
```

### GPU Acceleration

Enable GPU for supported methods:

```python
config = ATLASConfig(
    enable_gpu=True  # Accelerates genetic algorithms, PSO, etc.
)
```

## Troubleshooting

### Redis Connection Issues

```python
# Test Redis connection
import redis
r = redis.Redis.from_url("redis://localhost:6379/0")
r.ping()  # Should return True
```

### Task Not Processing

1. Check agent is registered:
```python
r.sismember("ORCHEX:available_agents", "Librex-agent")
```

2. Check task queue:
```python
r.zcard("Librex:tasks")  # Number of pending tasks
```

3. Check agent workload:
```python
agent.get_agent_status()["current_workload"]
```

### Method Selection Failures

If AI selector fails, check:
- Problem data is properly formatted
- Domain adapter exists for problem type
- Fallback to manual method selection

## Example: Complete Integration

```python
#!/usr/bin/env python3
"""
Example: Complete ORCHEX-Librex integration
"""

import numpy as np
from Librex.integrations.orchex import (
    LibrexAgent,
    ATLASConfig,
    ATLASOptimizationAdapter
)

def main():
    # Configure
    config = ATLASConfig(
        agent_id="Librex-main",
        max_concurrent_tasks=5,
        default_method="auto",
        enable_gpu=True
    )

    # Initialize agent
    agent = LibrexAgent(config)
    agent.register_with_atlas()
    agent.start_background_processor()

    # Initialize adapter for direct submission
    adapter = ATLASOptimizationAdapter(config)

    # Submit QAP task
    qap_result = adapter.submit_optimization_request(
        agent_id="test-agent",
        problem_type="qap",
        problem_data={
            "flow_matrix": np.random.rand(10, 10),
            "distance_matrix": np.random.rand(10, 10)
        },
        method="auto"
    )

    print(f"Submitted QAP task: {qap_result['task_id']}")

    # Submit TSP task
    tsp_result = adapter.submit_optimization_request(
        agent_id="test-agent",
        problem_type="tsp",
        problem_data={
            "coordinates": np.random.rand(20, 2).tolist()
        },
        method="simulated_annealing"
    )

    print(f"Submitted TSP task: {tsp_result['task_id']}")

    # Check status
    import time
    time.sleep(5)

    qap_status = adapter.get_task_status(qap_result['task_id'])
    tsp_status = adapter.get_task_status(tsp_result['task_id'])

    print(f"QAP status: {qap_status['status']}")
    print(f"TSP status: {tsp_status['status']}")

    # Get agent status
    agent_status = agent.get_agent_status()
    print(f"Agent status: {agent_status}")

    # Cleanup
    agent.stop_background_processor()
    agent.deregister_from_atlas()

if __name__ == "__main__":
    main()
```

## Future Enhancements

1. **WebSocket Support**: Real-time task status updates
2. **Distributed Processing**: Multi-node task distribution
3. **Advanced Scheduling**: Machine learning-based task scheduling
4. **Result Visualization**: Web dashboard for monitoring
5. **Quantum Integration**: Support for quantum optimization methods
6. **AutoML Integration**: Automatic hyperparameter tuning

## Support

For issues or questions about the ORCHEX-Librex integration:

1. Check the [Librex Documentation](../../../README.md)
2. Review [ORCHEX Documentation](/home/user/AlaweinOS/MEZAN/ORCHEX/START_HERE.md)
3. Contact: meshal@berkeley.edu

## License

Apache 2.0 - See LICENSE file for details.