# MEZAN Testing Guide

## Overview

This guide covers the comprehensive testing infrastructure for the MEZAN/ORCHEX system, including unit tests, integration tests, performance benchmarks, and chaos engineering.

## Table of Contents

1. [Test Structure](#test-structure)
2. [Running Tests](#running-tests)
3. [Test Categories](#test-categories)
4. [Performance Testing](#performance-testing)
5. [Chaos Engineering](#chaos-engineering)
6. [CI/CD Integration](#cicd-integration)
7. [Best Practices](#best-practices)

## Test Structure

```
MEZAN/ORCHEX/ORCHEX-core/tests/
├── unit/                 # Unit tests
├── integration/          # Integration tests
│   ├── test_end_to_end_workflow.py
│   ├── test_distributed_execution.py
│   ├── test_api_gateway_integration.py
│   ├── test_monitoring_integration.py
│   ├── test_event_bus_integration.py
│   ├── test_auth_integration.py
│   └── test_failure_scenarios.py
├── performance/         # Performance tests
│   ├── benchmark_throughput.py
│   ├── benchmark_latency.py
│   ├── benchmark_scalability.py
│   ├── benchmark_resource_usage.py
│   ├── benchmark_distributed.py
│   └── load_test.py
├── chaos/              # Chaos engineering
│   ├── chaos_network.py
│   ├── chaos_node_failure.py
│   ├── chaos_resource_exhaustion.py
│   └── chaos_latency_injection.py
└── fixtures/           # Test fixtures and data
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_engine.py

# Run with coverage
pytest tests/unit/ --cov=atlas_core --cov-report=html

# Run with verbose output
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_engine.py::TestEngine::test_workflow_execution
```

### Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -m integration

# Run specific integration test suite
pytest tests/integration/test_end_to_end_workflow.py

# Run with timeout (integration tests can be slow)
pytest tests/integration/ --timeout=300

# Run in parallel
pytest tests/integration/ -n auto
```

### Performance Tests

```bash
# Run throughput benchmarks
python tests/performance/benchmark_throughput.py

# Run latency benchmarks
python tests/performance/benchmark_latency.py

# Run scalability tests
python tests/performance/benchmark_scalability.py

# Run resource profiling
python tests/performance/benchmark_resource_usage.py

# Run distributed benchmarks
python tests/performance/benchmark_distributed.py

# Run all benchmarks
make benchmark
```

### Load Testing with Locust

```bash
# Run load test with web UI
locust -f tests/performance/load_test.py --host=http://localhost:8080

# Run headless with specific user count
locust -f tests/performance/load_test.py \
    --host=http://localhost:8080 \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m

# Run distributed load test (master)
locust -f tests/performance/load_test.py \
    --host=http://localhost:8080 \
    --master

# Run distributed load test (worker)
locust -f tests/performance/load_test.py \
    --host=http://localhost:8080 \
    --worker \
    --master-host=localhost
```

### Chaos Engineering

```bash
# Run network chaos tests (requires root)
sudo python tests/chaos/chaos_network.py

# Run node failure tests
python tests/chaos/chaos_node_failure.py

# Run resource exhaustion tests
python tests/chaos/chaos_resource_exhaustion.py

# Run all chaos tests
make chaos-test
```

## Test Categories

### 1. Unit Tests

Unit tests focus on individual components in isolation:

- **Engine Tests**: Core engine functionality
- **Agent Tests**: Individual agent behavior
- **Blackboard Tests**: Data storage and retrieval
- **Utility Tests**: Helper functions and utilities

Example:
```python
def test_workflow_creation():
    engine = ATLASEngine()
    workflow = engine.create_workflow({"type": "research"})
    assert workflow.id is not None
    assert workflow.status == "pending"
```

### 2. Integration Tests

Integration tests verify component interactions:

- **End-to-End Workflows**: Complete workflow execution
- **Distributed Execution**: Multi-node coordination
- **API Gateway**: Request routing and authentication
- **Event Bus**: Event publishing and subscription
- **Monitoring**: Metrics and logging integration

Example:
```python
@pytest.mark.integration
async def test_complete_research_workflow(engine, test_workflow):
    result = await engine.execute_workflow(test_workflow)
    assert result["status"] == "completed"
    assert len(result["completed_steps"]) == len(test_workflow["steps"])
```

### 3. Performance Tests

Performance tests measure system capabilities:

- **Throughput**: Requests/workflows per second
- **Latency**: Response times and percentiles
- **Scalability**: Horizontal and vertical scaling
- **Resource Usage**: CPU, memory, disk, network
- **Distributed Performance**: Multi-node efficiency

Example output:
```
Workflow Throughput: 150 workflows/second
P95 Latency: 250ms
P99 Latency: 500ms
Memory Usage: 512MB
CPU Utilization: 65%
```

### 4. Chaos Engineering

Chaos tests verify resilience:

- **Network Failures**: Latency, packet loss, partitions
- **Node Failures**: Crashes, restarts, unresponsive nodes
- **Resource Exhaustion**: Memory, CPU, disk space
- **Byzantine Failures**: Malicious or corrupted nodes

## Performance Testing

### Benchmarking Strategy

1. **Baseline Establishment**
   - Run benchmarks on clean system
   - Record baseline metrics
   - Document hardware/software configuration

2. **Load Patterns**
   - Steady load: Constant request rate
   - Step load: Incremental increases
   - Spike load: Sudden traffic bursts
   - Soak test: Extended duration

3. **Metrics Collection**
   - Request throughput
   - Response latency (p50, p95, p99)
   - Error rates
   - Resource utilization
   - Queue depths

4. **Analysis**
   - Compare against baselines
   - Identify bottlenecks
   - Generate recommendations

### Load Test Scenarios

```python
# Custom load shape for stress testing
class StressTestShape(LoadTestShape):
    stages = [
        {"duration": 60, "users": 10},    # Warm-up
        {"duration": 120, "users": 50},   # Ramp-up
        {"duration": 180, "users": 100},  # Sustained
        {"duration": 60, "users": 200},   # Peak
        {"duration": 60, "users": 50},    # Cool-down
    ]
```

## Chaos Engineering

### Chaos Test Principles

1. **Start Small**: Begin with single-node failures
2. **Gradual Escalation**: Increase failure complexity
3. **Observe and Learn**: Monitor system behavior
4. **Automate Recovery**: Verify self-healing
5. **Document Findings**: Record failure modes

### Common Chaos Scenarios

1. **Network Chaos**
   - Inject 200ms latency
   - 10% packet loss
   - Network partitions
   - Bandwidth limitations

2. **Node Failures**
   - Kill random nodes
   - Resource starvation
   - Clock skew
   - Disk failures

3. **Application Chaos**
   - Inject exceptions
   - Corrupt data
   - Deadlocks
   - Memory leaks

### Running Chaos Tests Safely

```bash
# Always run in isolated environment first
export CHAOS_ENV=staging

# Enable safety checks
export CHAOS_SAFETY=true

# Set blast radius limits
export CHAOS_MAX_IMPACT=0.2  # Max 20% of nodes

# Run with monitoring
python tests/chaos/chaos_network.py --monitor
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: pytest tests/unit/ --cov

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start services
        run: docker-compose up -d
      - name: Run integration tests
        run: pytest tests/integration/ -m integration

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: make benchmark
      - name: Upload results
        uses: actions/upload-artifact@v2
```

### Test Automation

1. **Pre-commit Hooks**
   ```yaml
   - repo: local
     hooks:
       - id: pytest
         name: pytest
         entry: pytest tests/unit/
         language: system
         pass_filenames: false
   ```

2. **Nightly Performance Tests**
   ```yaml
   schedule:
     - cron: '0 2 * * *'  # 2 AM daily
   ```

3. **Chaos Tests (Weekly)**
   ```yaml
   schedule:
     - cron: '0 3 * * 0'  # Sunday 3 AM
   ```

## Best Practices

### 1. Test Organization

- Group related tests in classes
- Use descriptive test names
- Keep tests focused and atomic
- Avoid test interdependencies

### 2. Test Data Management

```python
@pytest.fixture
def test_workflow():
    """Reusable test workflow fixture."""
    return {
        "id": str(uuid.uuid4()),
        "type": "research",
        "tasks": ["analyze", "process", "synthesize"]
    }
```

### 3. Mocking and Stubbing

```python
@patch('atlas_core.engine.ATLASEngine.execute_workflow')
async def test_with_mock(mock_execute):
    mock_execute.return_value = {"status": "completed"}
    result = await run_test()
    assert result["status"] == "completed"
```

### 4. Performance Test Baselines

- Establish baselines for each environment
- Track performance over time
- Alert on significant regressions
- Document expected performance

### 5. Chaos Test Safety

- Never run in production without approval
- Start with read-only tests
- Implement circuit breakers
- Have rollback procedures

### 6. Test Coverage Goals

- Unit tests: 80% coverage minimum
- Integration tests: Critical paths covered
- Performance tests: All major operations
- Chaos tests: Common failure scenarios

### 7. Test Documentation

- Document test purposes
- Include setup requirements
- Provide troubleshooting guides
- Maintain test inventories

## Troubleshooting

### Common Issues

1. **Flaky Tests**
   - Add retries for network operations
   - Increase timeouts for slow operations
   - Use proper async/await patterns

2. **Resource Leaks**
   - Ensure proper cleanup in fixtures
   - Close connections and files
   - Clear caches between tests

3. **Test Isolation**
   - Use separate databases/namespaces
   - Reset global state
   - Mock external dependencies

### Debug Commands

```bash
# Run with debug output
pytest tests/ -vv --log-cli-level=DEBUG

# Run with pdb on failure
pytest tests/ --pdb

# Run specific test with output
pytest tests/unit/test_engine.py::test_name -s

# Profile test execution
pytest tests/ --profile

# Generate test report
pytest tests/ --html=report.html
```

## Metrics and Reporting

### Test Metrics

- **Pass Rate**: Percentage of passing tests
- **Coverage**: Code coverage percentage
- **Duration**: Test execution time
- **Flakiness**: Test reliability score

### Performance Metrics

- **Throughput**: Operations per second
- **Latency**: Response time percentiles
- **Error Rate**: Failed request percentage
- **Resource Usage**: CPU/Memory/Disk/Network

### Reporting Tools

1. **pytest-html**: HTML test reports
2. **allure**: Comprehensive test reporting
3. **coverage.py**: Code coverage reports
4. **locust**: Load test reports
5. **grafana**: Performance dashboards

## Continuous Improvement

1. **Regular Reviews**
   - Weekly test failure analysis
   - Monthly performance reviews
   - Quarterly chaos test updates

2. **Test Maintenance**
   - Update tests with code changes
   - Remove obsolete tests
   - Refactor complex tests

3. **Knowledge Sharing**
   - Document test patterns
   - Share failure analyses
   - Conduct test reviews

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Locust Documentation](https://docs.locust.io/)
- [Chaos Engineering Principles](https://principlesofchaos.org/)
- [Performance Testing Best Practices](https://www.performancetestingguide.com/)

---

For questions or support, contact the MEZAN team or refer to the main documentation.