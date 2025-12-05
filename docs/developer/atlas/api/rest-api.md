# REST API Reference

Complete REST API documentation for ORCHEX, including all endpoints, request/response formats, authentication, error handling, and practical examples.

---

## API Overview

The ORCHEX REST API provides programmatic access to all ORCHEX functionality, enabling integration with external systems, automation workflows, and custom applications.

**Base URL:** `https://api.orchex-platform.com/v1`  
**Protocol:** HTTPS only  
**Authentication:** API Key or JWT Token  
**Rate Limiting:** 1000 requests per hour per API key

---

## Authentication

### API Key Authentication

Include your API key in the `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.orchex-platform.com/v1/tasks
```

### JWT Token Authentication

For session-based authentication:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     https://api.orchex-platform.com/v1/tasks
```

### Obtaining API Keys

1. **Via Web Interface:** Log into ORCHEX dashboard and generate API keys
2. **Via CLI:** `ORCHEX config get api.key`
3. **Via API:** Use the authentication endpoints

---

## Response Format

All API responses follow a consistent JSON structure:

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "req_123456",
    "timestamp": "2025-11-29T21:00:00Z",
    "version": "v1"
  }
}
```

### Error Responses

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "description",
      "issue": "required"
    }
  },
  "meta": {
    "request_id": "req_123456",
    "timestamp": "2025-11-29T21:00:00Z"
  }
}
```

### Common Error Codes

| Code                   | Description                    | HTTP Status |
| ---------------------- | ------------------------------ | ----------- |
| `VALIDATION_ERROR`     | Invalid request parameters     | 400         |
| `AUTHENTICATION_ERROR` | Invalid or missing credentials | 401         |
| `AUTHORIZATION_ERROR`  | Insufficient permissions       | 403         |
| `NOT_FOUND`            | Resource not found             | 404         |
| `RATE_LIMIT_EXCEEDED`  | Too many requests              | 429         |
| `INTERNAL_ERROR`       | Server error                   | 500         |

---

## Agent Management

### Register Agent

Register a new AI agent with the system.

```http
POST /agents
```

**Request Body:**

```json
{
  "agent_id": "claude-sonnet-4",
  "name": "Claude Sonnet 4",
  "provider": "anthropic",
  "model": "claude-sonnet-4.5",
  "capabilities": ["code_generation", "code_review", "refactoring", "debugging"],
  "constraints": {
    "max_tokens": 200000,
    "max_concurrent_tasks": 5,
    "rate_limit_per_minute": 50,
    "cost_per_1k_tokens": 0.003
  },
  "config": {
    "base_weight": 1.0,
    "retry_policy": {
      "max_retries": 3,
      "backoff_multiplier": 2.0
    }
  }
}
```

**Response (201 Created):**

```json
{
  "success": true,
  "data": {
    "agent_id": "claude-sonnet-4",
    "status": "registered",
    "health": "healthy",
    "registered_at": "2025-11-29T21:00:00Z"
  }
}
```

**Curl Example:**

```bash
curl -X POST https://api.orchex-platform.com/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "claude-sonnet-4",
    "name": "Claude Sonnet 4",
    "provider": "anthropic",
    "model": "claude-sonnet-4.5",
    "capabilities": ["code_generation", "code_review"]
  }'
```

### List Agents

Retrieve all registered agents with optional filtering.

```http
GET /agents
```

**Query Parameters:**

- `provider` - Filter by provider (anthropic, openai, google)
- `capability` - Filter by capability
- `status` - Filter by status (active, inactive)
- `limit` - Maximum number of results (default: 50)
- `offset` - Pagination offset (default: 0)

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "agent_id": "claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "provider": "anthropic",
        "status": "active",
        "health": {
          "status": "healthy",
          "uptime_percentage": 99.5,
          "avg_response_time_ms": 1250
        },
        "performance": {
          "success_rate": 0.95,
          "avg_quality_score": 87.3,
          "total_tasks_completed": 1523
        }
      }
    ],
    "total": 1,
    "limit": 50,
    "offset": 0
  }
}
```

### Get Agent Details

Retrieve detailed information about a specific agent.

```http
GET /agents/{agent_id}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "agent_id": "claude-sonnet-4",
    "name": "Claude Sonnet 4",
    "provider": "anthropic",
    "model": "claude-sonnet-4.5",
    "capabilities": ["code_generation", "code_review", "refactoring", "debugging"],
    "constraints": {
      "max_tokens": 200000,
      "max_concurrent_tasks": 5,
      "rate_limit_per_minute": 50,
      "cost_per_1k_tokens": 0.003
    },
    "health": {
      "status": "healthy",
      "last_check": "2025-11-29T20:55:00Z",
      "uptime_percentage": 99.5,
      "avg_response_time_ms": 1250,
      "error_rate": 0.02
    },
    "performance": {
      "success_rate": 0.95,
      "avg_quality_score": 87.3,
      "total_tasks_completed": 1523,
      "total_failures": 76,
      "avg_cost_per_task": 0.12
    },
    "registered_at": "2025-11-29T10:00:00Z",
    "last_seen": "2025-11-29T20:55:00Z"
  }
}
```

### Update Agent

Update agent configuration or metadata.

```http
PUT /agents/{agent_id}
```

**Request Body:**

```json
{
  "name": "Updated Claude Agent",
  "constraints": {
    "max_concurrent_tasks": 10
  },
  "config": {
    "base_weight": 1.5
  }
}
```

### Delete Agent

Remove an agent from the system.

```http
DELETE /agents/{agent_id}
```

**Response (204 No Content)**

---

## Task Management

### Submit Task

Submit a new task for execution.

```http
POST /tasks
```

**Request Body:**

```json
{
  "type": "code_generation",
  "description": "Create a REST API endpoint for user authentication in Express.js",
  "context": {
    "repository": "my-app",
    "files": ["src/auth.js"],
    "language": "javascript",
    "framework": "express"
  },
  "requirements": {
    "required_capabilities": ["code_generation", "security_analysis"],
    "max_tokens": 4000,
    "timeout_seconds": 300,
    "priority": "high"
  },
  "constraints": {
    "max_cost_usd": 0.5,
    "preferred_providers": ["anthropic", "openai"],
    "excluded_agents": []
  },
  "metadata": {
    "user_id": "user123",
    "project": "authentication-service"
  }
}
```

**Response (202 Accepted):**

```json
{
  "success": true,
  "data": {
    "task_id": "task_abc123def456",
    "status": "queued",
    "estimated_completion": "2025-11-29T21:05:00Z",
    "queue_position": 2
  }
}
```

**Curl Example:**

```bash
curl -X POST https://api.orchex-platform.com/v1/tasks \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "code_generation",
    "description": "Create a user registration endpoint",
    "context": {
      "language": "javascript",
      "framework": "express"
    },
    "requirements": {
      "required_capabilities": ["code_generation"]
    }
  }'
```

### Get Task Status

Retrieve the current status and results of a task.

```http
GET /tasks/{task_id}
```

**Response (200 OK) - Running:**

```json
{
  "success": true,
  "data": {
    "task_id": "task_abc123def456",
    "status": "running",
    "progress": {
      "percentage": 75,
      "stage": "executing",
      "message": "Generating code with Claude Sonnet 4"
    },
    "agent_id": "claude-sonnet-4",
    "started_at": "2025-11-29T21:00:00Z",
    "estimated_completion": "2025-11-29T21:02:00Z"
  }
}
```

**Response (200 OK) - Completed:**

```json
{
  "success": true,
  "data": {
    "task_id": "task_abc123def456",
    "status": "completed",
    "result": {
      "code": "const express = require('express');\nconst router = express.Router();\n\nconst authenticateUser = async (req, res) => {\n  try {\n    const { email, password } = req.body;\n    \n    // Authentication logic here\n    \n    res.json({ success: true, token: 'jwt_token' });\n  } catch (error) {\n    res.status(401).json({ error: 'Authentication failed' });\n  }\n};\n\nrouter.post('/login', authenticateUser);\n\nmodule.exports = router;",
      "explanation": "Created a basic authentication endpoint with error handling",
      "suggestions": ["Add input validation", "Implement JWT token generation", "Add rate limiting"]
    },
    "agent_id": "claude-sonnet-4",
    "tier": 0,
    "attempts": 1,
    "duration_ms": 3240,
    "cost_usd": 0.012,
    "completed_at": "2025-11-29T21:01:00Z"
  }
}
```

### List Tasks

Retrieve a list of tasks with filtering and pagination.

```http
GET /tasks
```

**Query Parameters:**

- `status` - Filter by status (queued, running, completed, failed)
- `type` - Filter by task type
- `agent_id` - Filter by agent
- `user_id` - Filter by user
- `since` - Filter tasks created after timestamp
- `until` - Filter tasks created before timestamp
- `limit` - Maximum results (default: 50, max: 100)
- `offset` - Pagination offset

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "task_id": "task_abc123",
        "type": "code_generation",
        "status": "completed",
        "description": "Create authentication endpoint",
        "agent_id": "claude-sonnet-4",
        "created_at": "2025-11-29T21:00:00Z",
        "completed_at": "2025-11-29T21:01:00Z",
        "cost_usd": 0.012
      }
    ],
    "total": 1,
    "limit": 50,
    "offset": 0
  }
}
```

### Cancel Task

Cancel a queued or running task.

```http
DELETE /tasks/{task_id}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "task_id": "task_abc123",
    "status": "cancelled",
    "cancelled_at": "2025-11-29T21:00:30Z"
  }
}
```

### Retry Task

Retry a failed task with the same or different agent.

```http
POST /tasks/{task_id}/retry
```

**Request Body (optional):**

```json
{
  "agent_id": "gpt-4-turbo",
  "priority": "high"
}
```

**Response (202 Accepted):**

```json
{
  "success": true,
  "data": {
    "task_id": "task_abc123",
    "retry_task_id": "task_def456",
    "status": "queued"
  }
}
```

---

## Repository Analysis

### Start Analysis

Initiate a repository analysis.

```http
POST /analyze
```

**Request Body:**

```json
{
  "repository_path": "/path/to/repo",
  "repository_url": "https://github.com/user/repo",
  "branch": "main",
  "commit": "abc123...",
  "analysis_type": "full",
  "options": {
    "include_metrics": true,
    "identify_opportunities": true,
    "max_opportunities": 20,
    "include_patterns": ["**/*.js", "**/*.ts"],
    "exclude_patterns": ["**/node_modules/**", "**/dist/**"]
  }
}
```

**Response (202 Accepted):**

```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis_xyz789",
    "status": "running",
    "estimated_completion": "2025-11-29T21:20:00Z"
  }
}
```

### Get Analysis Results

Retrieve analysis results.

```http
GET /analyze/{analysis_id}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis_xyz789",
    "status": "completed",
    "repository": "https://github.com/user/repo",
    "branch": "main",
    "commit": "abc123...",
    "summary": {
      "total_files": 45,
      "files_analyzed": 42,
      "avg_chaos_score": 42.3,
      "high_chaos_files": 8,
      "total_opportunities": 23,
      "estimated_debt_hours": 67.5
    },
    "chaos_metrics": [
      {
        "file_path": "src/utils/helpers.js",
        "total_score": 78.5,
        "complexity": { "cyclomatic": 15, "cognitive": 12 },
        "duplication": { "percentage": 0.25 },
        "size": { "lines_of_code": 250 }
      }
    ],
    "opportunities": [
      {
        "opportunity_id": "opp_123",
        "type": "extract_function",
        "file_path": "src/utils/helpers.js",
        "description": "Complex function should be broken down",
        "impact": { "complexity_reduction": 40 },
        "risk": { "level": "medium" }
      }
    ],
    "completed_at": "2025-11-29T21:15:00Z",
    "duration_ms": 900000
  }
}
```

### List Analyses

Retrieve analysis history.

```http
GET /analyze
```

**Query Parameters:**

- `repository` - Filter by repository
- `status` - Filter by status
- `since` - Filter by start date
- `limit` - Pagination limit

---

## Refactoring Operations

### Apply Refactoring

Apply a specific refactoring operation.

```http
POST /refactor
```

**Request Body:**

```json
{
  "opportunity_id": "opp_123",
  "dry_run": false,
  "create_pr": true,
  "pr_options": {
    "title": "Refactor complex helper function",
    "body": "Automated refactoring to improve code maintainability",
    "branch": "refactor/helpers"
  }
}
```

**Response (202 Accepted):**

```json
{
  "success": true,
  "data": {
    "refactoring_id": "refactor_456",
    "status": "running",
    "estimated_completion": "2025-11-29T21:10:00Z"
  }
}
```

### Get Refactoring Status

Check refactoring progress and results.

```http
GET /refactor/{refactoring_id}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "refactoring_id": "refactor_456",
    "status": "completed",
    "success": true,
    "changes": {
      "files_modified": 2,
      "lines_added": 15,
      "lines_removed": 42
    },
    "safety_report": {
      "overall_safety": "safe",
      "checks": [
        { "check": "syntax_validation", "passed": true },
        { "check": "type_checking", "passed": true },
        { "check": "test_execution", "passed": true }
      ]
    },
    "pull_request": {
      "url": "https://github.com/user/repo/pull/123",
      "number": 123,
      "title": "Refactor complex helper function"
    },
    "completed_at": "2025-11-29T21:08:00Z"
  }
}
```

---

## Metrics and Monitoring

### Get System Metrics

Retrieve system performance metrics.

```http
GET /metrics
```

**Query Parameters:**

- `period` - Time period (1h, 24h, 7d, 30d)
- `metric_type` - Type of metrics (performance, cost, quality)

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "period": "24h",
    "total_tasks": 342,
    "success_rate": 0.94,
    "avg_duration_ms": 2150,
    "total_cost_usd": 45.67,
    "agents": {
      "claude-sonnet-4": {
        "tasks": 156,
        "success_rate": 0.96,
        "avg_quality": 88.2,
        "cost_usd": 18.45
      }
    }
  }
}
```

### Get Agent Metrics

Retrieve metrics for a specific agent.

```http
GET /metrics/agents/{agent_id}
```

**Query Parameters:**

- `period` - Time period
- `detailed` - Include detailed breakdowns

---

## System Management

### Health Check

Check system health and component status.

```http
GET /health
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 86400,
    "components": {
      "router": "healthy",
      "registry": "healthy",
      "load_balancer": "healthy",
      "fallback_manager": "healthy",
      "analyzer": "healthy",
      "refactoring_engine": "healthy"
    },
    "agents": {
      "total": 4,
      "healthy": 3,
      "degraded": 1,
      "offline": 0
    }
  }
}
```

### System Status

Get detailed system status and queue information.

```http
GET /status
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "system": "healthy",
    "components": {
      "router": { "status": "healthy", "load": 0.3 },
      "registry": { "status": "healthy", "agents": 4 },
      "load_balancer": { "status": "healthy", "distribution": "balanced" }
    },
    "queue": {
      "queued_tasks": 5,
      "running_tasks": 12,
      "avg_wait_time_ms": 450
    },
    "performance": {
      "requests_per_second": 15.2,
      "avg_response_time_ms": 245,
      "error_rate": 0.02
    }
  }
}
```

---

## Webhooks

### Webhook Configuration

Configure webhooks for real-time notifications.

```http
POST /webhooks
```

**Request Body:**

```json
{
  "url": "https://my-app.com/webhooks/ORCHEX",
  "secret": "webhook_secret",
  "events": ["task.completed", "task.failed", "analysis.completed", "refactoring.applied"],
  "active": true
}
```

### Webhook Payloads

**Task Completed:**

```json
{
  "event": "task.completed",
  "timestamp": "2025-11-29T21:01:00Z",
  "data": {
    "task_id": "task_abc123",
    "type": "code_generation",
    "status": "completed",
    "result": { ... },
    "agent_id": "claude-sonnet-4",
    "cost_usd": 0.012
  }
}
```

**Analysis Completed:**

```json
{
  "event": "analysis.completed",
  "timestamp": "2025-11-29T21:15:00Z",
  "data": {
    "analysis_id": "analysis_xyz789",
    "repository": "user/repo",
    "summary": { ... },
    "opportunities_count": 23
  }
}
```

---

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Authenticated requests:** 1000 requests per hour
- **Anonymous requests:** 100 requests per hour
- **Burst limit:** 50 requests per minute

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1638201600
X-RateLimit-Retry-After: 3600
```

---

## SDK Examples

### Python SDK

```python
from ORCHEX import ORCHEX

# Initialize client
ORCHEX = ORCHEX(api_key="your_api_key")

# Register agent
agent = ORCHEX.agents.register(
    agent_id="claude-sonnet-4",
    name="Claude Sonnet 4",
    provider="anthropic",
    capabilities=["code_generation", "code_review"]
)

# Submit task
task = ORCHEX.tasks.submit(
    type="code_generation",
    description="Create authentication endpoint",
    context={"language": "javascript", "framework": "express"}
)

# Wait for completion
result = task.wait()
print(result.code)
```

### JavaScript SDK

```javascript
const { ORCHEX } = require('@ORCHEX/sdk');

const ORCHEX = new ORCHEX({ apiKey: 'your_api_key' });

// Submit task
const result = await ORCHEX.tasks.submit({
  type: 'code_generation',
  description: 'Create user registration endpoint',
  context: { language: 'javascript', framework: 'express' },
});

console.log(result.code);
```

### cURL Examples

```bash
# Register agent
curl -X POST https://api.orchex-platform.com/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "claude-sonnet-4", "name": "Claude", "provider": "anthropic"}'

# Submit task
curl -X POST https://api.orchex-platform.com/v1/tasks \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type": "code_generation", "description": "Create API endpoint"}'

# Check status
curl https://api.orchex-platform.com/v1/tasks/task_123 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Error Handling

### Common Error Scenarios

**Invalid API Key:**

```json
{
  "success": false,
  "error": {
    "code": "AUTHENTICATION_ERROR",
    "message": "Invalid API key"
  }
}
```

**Rate Limit Exceeded:**

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "retry_after": 3600,
      "limit": 1000,
      "remaining": 0
    }
  }
}
```

**Validation Error:**

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "type",
      "issue": "must be one of: code_generation, code_review, ..."
    }
  }
}
```

---

## Best Practices

1. **Use appropriate timeouts** for long-running operations
2. **Implement retry logic** with exponential backoff
3. **Monitor rate limits** and handle 429 responses
4. **Validate responses** before processing
5. **Use webhooks** for real-time notifications instead of polling
6. **Cache results** when appropriate to reduce API calls
7. **Handle errors gracefully** with proper fallback logic

This comprehensive API reference provides everything needed to integrate ORCHEX into your applications and workflows. For additional support, visit our [developer community](https://community.orchex-platform.com) or contact [enterprise support](mailto:enterprise@ORCHEX-platform.com).</instructions>
