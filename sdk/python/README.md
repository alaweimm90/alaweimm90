# ATLAS Python SDK

Python client library for the ATLAS Multi-Agent Orchestration Platform.

## Installation

```bash
pip install atlas-sdk
```

Or install from source:

```bash
cd sdk/python
pip install -e .
```

## Quick Start

```python
from atlas import AtlasClient

# Initialize client
client = AtlasClient(
    base_url="http://localhost:3000",
    api_key="your-api-key"
)

# Check health
health = client.health()
print(f"Status: {health.status}")

# Generate code
code = client.generate(
    "Write a function to calculate fibonacci numbers",
    language="python"
)
print(code)

# Review code
review = client.review(
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    language="python"
)
print(review)

# Chat with an agent
response = client.chat("What is dependency injection?")
print(response)
```

## API Reference

### AtlasClient

The main client class for interacting with ATLAS.

#### Constructor

```python
AtlasClient(
    base_url: str = "http://localhost:3000",
    api_key: Optional[str] = None,
    timeout: int = 120
)
```

#### Methods

##### health() -> HealthStatus

Check API health status.

```python
health = client.health()
print(health.status)  # "healthy"
print(health.version)  # "1.0.0"
```

##### list_agents() -> List[Agent]

List all available agents.

```python
agents = client.list_agents()
for agent in agents:
    print(f"{agent.id}: {agent.name}")
```

##### get_agent(agent_id: str) -> Agent

Get details of a specific agent.

```python
agent = client.get_agent("claude-agent")
print(agent.capabilities)
```

##### execute(task_type, description, ...) -> ExecutionResult

Execute a task using ATLAS orchestration.

```python
from atlas import TaskType, TaskPriority

result = client.execute(
    task_type=TaskType.CODE_GENERATION,
    description="Generate a REST API endpoint",
    priority=TaskPriority.HIGH
)
print(result.output)
```

##### generate(description, language?, context?) -> str

Generate code using ATLAS.

```python
code = client.generate(
    "Create a binary search function",
    language="python"
)
```

##### review(code, language?, context?) -> str

Review code using ATLAS.

```python
feedback = client.review(
    "def add(a, b): return a + b",
    language="python"
)
```

##### explain(code, language?) -> str

Get an explanation of code.

```python
explanation = client.explain(
    "lambda x: x * 2",
    language="python"
)
```

##### chat(message) -> str

Chat with an ATLAS agent.

```python
response = client.chat("How do I implement authentication?")
```

### Types

#### TaskType

```python
class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_EXPLANATION = "code_explanation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    GENERAL = "general"
```

#### TaskPriority

```python
class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### Exceptions

```python
from atlas.exceptions import (
    AtlasError,           # Base exception
    AuthenticationError,  # 401 errors
    RateLimitError,       # 429 errors
    ValidationError,      # 400 errors
    ConnectionError,      # Network errors
    TaskExecutionError,   # Task failures
)
```

## Error Handling

```python
from atlas import AtlasClient
from atlas.exceptions import AtlasError, RateLimitError

client = AtlasClient("http://localhost:3000")

try:
    result = client.generate("Create a hello world function")
except RateLimitError:
    print("Rate limited, please wait...")
except AtlasError as e:
    print(f"Error: {e}")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy atlas

# Linting
ruff check atlas
```

## License

MIT
