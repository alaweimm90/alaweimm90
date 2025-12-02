# ATLAS Python SDK

[![PyPI version](https://badge.fury.io/py/atlas-sdk.svg)](https://pypi.org/project/atlas-sdk/)
[![Python Versions](https://img.shields.io/pypi/pyversions/atlas-sdk)](https://pypi.org/project/atlas-sdk/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python SDK for the ATLAS (Autonomous Technical Leadership & Adaptive System) multiagent LLM orchestration platform.

## Features

- 🚀 **Task Management**: Submit, monitor, and manage AI-powered development tasks
- 🤖 **Agent Orchestration**: Access multiple AI agents (Claude, GPT-4, Gemini, etc.)
- 📊 **Repository Analysis**: Automated code quality analysis and refactoring suggestions
- 🔧 **Async Support**: Full async/await support for high-performance applications
- 🛡️ **Type Safety**: Full type hints and Pydantic models
- 📚 **Rich Documentation**: Comprehensive API documentation and examples

## Installation

```bash
pip install atlas-sdk
```

For development with extra dependencies:

```bash
pip install atlas-sdk[dev,analysis]
```

## Quick Start

```python
from atlas import AtlasClient, TaskType

# Initialize client
client = AtlasClient(api_key="your-api-key")

# Submit a code generation task
task = client.submit_task(
    type=TaskType.CODE_GENERATION,
    description="Create a REST API endpoint for user authentication in FastAPI",
    context={
        "language": "python",
        "framework": "fastapi"
    }
)

# Wait for completion
result = task.wait()
print("Generated code:")
print(result.result.code)
```

## Usage Examples

### Code Review

```python
# Review existing code
task = client.submit_task(
    type=TaskType.CODE_REVIEW,
    description="Review this authentication function for security issues",
    context={
        "language": "python",
        "code": "def authenticate_user(username, password): ..."
    }
)

result = task.wait()
print("Review comments:")
for suggestion in result.result.suggestions:
    print(f"- {suggestion}")
```

### Repository Analysis

```python
# Analyze a repository
analysis = client.start_analysis(
    repository_path="/path/to/your/repo",
    analysis_type="full"
)

# Wait for completion
analysis.wait()
print(f"Found {len(analysis.opportunities)} refactoring opportunities")
print(f"Average chaos score: {analysis.summary['avg_chaos_score']}")
```

### Async Operations

```python
import asyncio

async def main():
    async with AtlasClient(api_key="your-key") as client:
        # Submit multiple tasks concurrently
        tasks = await asyncio.gather(*[
            client.submit_task_async(TaskType.CODE_GENERATION, desc)
            for desc in task_descriptions
        ])

        # Wait for all to complete
        results = await asyncio.gather(*[
            task.wait_async() for task in tasks
        ])

asyncio.run(main())
```

## API Reference

### AtlasClient

The main client class for interacting with ATLAS.

#### Methods

- `submit_task(type, description, context=None, requirements=None)` - Submit a new task
- `get_task(task_id)` - Retrieve a task by ID
- `list_tasks(status=None, type=None, limit=50, offset=0)` - List tasks with filtering
- `list_agents(provider=None, capability=None, status=None)` - List available agents
- `start_analysis(repository_path=None, repository_url=None, **options)` - Start repository analysis
- `get_metrics(period='24h')` - Get system metrics

### Task

Represents a submitted task.

#### Properties

- `task_id` - Unique task identifier
- `status` - Current execution status (queued, running, completed, failed)
- `result` - Task result (when completed)
- `agent_id` - ID of agent that executed the task
- `cost_usd` - Cost incurred for execution

#### Methods

- `wait(timeout=None, poll_interval=2.0)` - Wait for task completion
- `cancel()` - Cancel a running task

### Task Types

- `TaskType.CODE_GENERATION` - Generate new code
- `TaskType.CODE_REVIEW` - Review existing code
- `TaskType.REFACTORING` - Refactor code for improvement
- `TaskType.DEBUGGING` - Debug and fix issues
- `TaskType.ANALYSIS` - Analyze code quality

## Configuration

### Environment Variables

```bash
export ATLAS_API_KEY="your-api-key"
export ATLAS_API_URL="https://api.atlas-platform.com/v1"
```

### Programmatic Configuration

```python
client = AtlasClient(
    api_key="your-api-key",
    api_url="https://custom-api-url.com/v1",
    timeout=60
)
```

## Error Handling

```python
from atlas import AtlasError, AuthenticationError, RateLimitError

try:
    task = client.submit_task(TaskType.CODE_GENERATION, "Generate API")
    result = task.wait()
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, please try again later")
except AtlasError as e:
    print(f"ATLAS error: {e}")
```

## Advanced Usage

### Custom Task Requirements

```python
from atlas import TaskRequirements

requirements = TaskRequirements(
    required_capabilities=["code_generation", "security_analysis"],
    max_tokens=4000,
    timeout_seconds=300,
    priority="high",
    max_cost_usd=0.50,
    preferred_providers=["anthropic", "openai"]
)

task = client.submit_task(
    type=TaskType.CODE_GENERATION,
    description="Create secure authentication endpoint",
    requirements=requirements
)
```

### Metrics and Monitoring

```python
# Get system metrics
metrics = client.get_metrics(period="24h")
print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Total tasks: {metrics.total_tasks}")
print(f"Total cost: ${metrics.total_cost_usd}")

# Agent-specific metrics
for agent_id, agent_metrics in metrics.agents.items():
    print(f"{agent_id}: {agent_metrics['success_rate']:.1%} success rate")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Development

```bash
# Clone the repository
git clone https://github.com/your-org/atlas-python-sdk.git
cd atlas-python-sdk

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
flake8 atlas/
mypy atlas/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://atlas-sdk.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/your-org/atlas-python-sdk/issues)
- 💬 [Community Forum](https://community.atlas-platform.com)
- 📧 [Enterprise Support](mailto:enterprise@atlas-platform.com)

---

Built with ❤️ for the ATLAS platform. Automate your development workflow with AI-powered orchestration.
