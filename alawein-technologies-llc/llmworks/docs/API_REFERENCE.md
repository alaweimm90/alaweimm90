# LLM Works API Reference

## Overview

LLM Works provides a comprehensive API for programmatic access to LLM evaluation capabilities. This document covers all available endpoints, data structures, and integration patterns.

## Table of Contents

- [Authentication](#authentication)
- [Core Concepts](#core-concepts)
- [Evaluation API](#evaluation-api)
- [Arena API](#arena-api)
- [Benchmark API](#benchmark-api)
- [Results API](#results-api)
- [Models API](#models-api)
- [System API](#system-api)
- [WebSocket Events](#websocket-events)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)
- [SDK Examples](#sdk-examples)

## Authentication

LLM Works uses API keys for authentication. All requests must include a valid API key in the header.

```http
Authorization: Bearer your-api-key-here
Content-Type: application/json
```

### Getting an API Key

1. Navigate to Settings → API Keys
2. Click "Generate New Key"
3. Copy and securely store your key
4. Keys can be revoked at any time

## Core Concepts

### Evaluation Types

- **Arena Evaluations**: Interactive, comparative testing between models
- **Benchmark Evaluations**: Standardized tests (MMLU, TruthfulQA, GSM8K)
- **Custom Evaluations**: User-defined test suites

### Model Configuration

```typescript
interface ModelConfig {
  id: string;
  name: string;
  provider: 'openai' | 'anthropic' | 'google' | 'meta' | 'custom';
  endpoint?: string;
  apiKey: string;
  parameters: {
    temperature: number;
    maxTokens: number;
    topP?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
  };
}
```

### Evaluation Result

```typescript
interface EvaluationResult {
  id: string;
  type: 'arena' | 'benchmark' | 'custom';
  status: 'pending' | 'running' | 'completed' | 'failed';
  createdAt: string;
  completedAt?: string;
  models: ModelConfig[];
  metrics: {
    accuracy?: number;
    latency: number;
    tokenUsage: number;
    cost: number;
  };
  results: any[];
  metadata: {
    version: string;
    environment: string;
    duration: number;
  };
}
```

## Evaluation API

### Create Evaluation

Start a new evaluation with specified models and configuration.

```http
POST /api/v1/evaluations
```

**Request Body:**

```json
{
  "type": "arena",
  "name": "GPT-4 vs Claude Comparison",
  "description": "Comparative evaluation of reasoning capabilities",
  "models": [
    {
      "id": "gpt-4",
      "provider": "openai",
      "apiKey": "sk-...",
      "parameters": {
        "temperature": 0.7,
        "maxTokens": 4096
      }
    },
    {
      "id": "claude-3-sonnet",
      "provider": "anthropic",
      "apiKey": "sk-ant-...",
      "parameters": {
        "temperature": 0.7,
        "maxTokens": 4096
      }
    }
  ],
  "config": {
    "rounds": 10,
    "categories": ["reasoning", "creativity", "factual"],
    "judgeModel": "gpt-4",
    "randomSeed": 42
  }
}
```

**Response:**

```json
{
  "id": "eval_abc123",
  "status": "pending",
  "estimatedDuration": 300,
  "createdAt": "2024-01-15T10:30:00Z",
  "websocketUrl": "wss://api.llmworks.dev/ws/eval_abc123"
}
```

### Get Evaluation Status

```http
GET /api/v1/evaluations/{evaluationId}
```

**Response:**

```json
{
  "id": "eval_abc123",
  "status": "running",
  "progress": 0.65,
  "currentStep": "Running debate round 7/10",
  "results": {
    "completed": 6,
    "total": 10,
    "current": {
      "gpt-4": 0.72,
      "claude-3-sonnet": 0.68
    }
  }
}
```

### List Evaluations

```http
GET /api/v1/evaluations?page=1&limit=20&type=arena&status=completed
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Results per page (default: 20, max: 100)
- `type`: Filter by evaluation type
- `status`: Filter by status
- `model`: Filter by model ID
- `createdAfter`: ISO date string
- `createdBefore`: ISO date string

## Arena API

### Start Arena Battle

```http
POST /api/v1/arena/battles
```

**Request Body:**

```json
{
  "modelA": "gpt-4",
  "modelB": "claude-3-sonnet",
  "prompt": "Explain quantum computing to a 10-year-old",
  "judgeModel": "gpt-4",
  "criteria": [
    "clarity",
    "accuracy",
    "age-appropriateness"
  ]
}
```

### Get Arena Leaderboard

```http
GET /api/v1/arena/leaderboard
```

**Response:**

```json
{
  "models": [
    {
      "id": "gpt-4",
      "name": "GPT-4",
      "eloRating": 1247,
      "wins": 156,
      "losses": 89,
      "draws": 23,
      "winRate": 0.582,
      "categories": {
        "reasoning": 1289,
        "creativity": 1205,
        "factual": 1247
      }
    }
  ],
  "updatedAt": "2024-01-15T10:30:00Z"
}
```

## Benchmark API

### Run Standard Benchmark

```http
POST /api/v1/benchmarks/run
```

**Request Body:**

```json
{
  "benchmark": "mmlu",
  "models": ["gpt-4", "claude-3-sonnet"],
  "subset": ["high_school_mathematics", "college_chemistry"],
  "config": {
    "fewShot": 5,
    "temperature": 0,
    "maxTokens": 1024
  }
}
```

### Get Benchmark Results

```http
GET /api/v1/benchmarks/{benchmarkId}/results
```

**Response:**

```json
{
  "id": "bench_xyz789",
  "benchmark": "mmlu",
  "completedAt": "2024-01-15T11:45:00Z",
  "results": {
    "gpt-4": {
      "overall": 0.847,
      "subjects": {
        "high_school_mathematics": 0.832,
        "college_chemistry": 0.891
      },
      "details": {
        "correct": 423,
        "total": 500,
        "accuracy": 0.846
      }
    }
  }
}
```

### List Available Benchmarks

```http
GET /api/v1/benchmarks
```

**Response:**

```json
{
  "benchmarks": [
    {
      "id": "mmlu",
      "name": "Massive Multitask Language Understanding",
      "description": "57 academic subjects across STEM, humanities, and social sciences",
      "categories": ["reasoning", "knowledge", "comprehension"],
      "questionCount": 15459,
      "estimatedTime": 45,
      "subjects": [
        "abstract_algebra",
        "anatomy",
        "astronomy"
      ]
    }
  ]
}
```

## Results API

### Export Results

```http
GET /api/v1/results/{evaluationId}/export?format=json
```

**Query Parameters:**
- `format`: Export format (`json`, `csv`, `xlsx`, `pdf`)
- `includeRaw`: Include raw model outputs (default: false)
- `includeMetadata`: Include evaluation metadata (default: true)

### Get Result Analytics

```http
GET /api/v1/results/{evaluationId}/analytics
```

**Response:**

```json
{
  "summary": {
    "totalQuestions": 100,
    "averageLatency": 2.3,
    "totalCost": 0.45,
    "duration": 847
  },
  "modelComparison": {
    "winner": "gpt-4",
    "confidence": 0.73,
    "margin": 0.12
  },
  "categoryBreakdown": {
    "reasoning": {
      "gpt-4": 0.82,
      "claude-3-sonnet": 0.79
    }
  }
}
```

## Models API

### List Configured Models

```http
GET /api/v1/models
```

### Add Model Configuration

```http
POST /api/v1/models
```

**Request Body:**

```json
{
  "name": "Custom GPT-4",
  "provider": "openai",
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "apiKey": "sk-...",
  "parameters": {
    "temperature": 0.7,
    "maxTokens": 4096,
    "topP": 0.9
  },
  "metadata": {
    "version": "gpt-4-0613",
    "description": "GPT-4 with custom parameters for evaluation"
  }
}
```

### Test Model Connection

```http
POST /api/v1/models/{modelId}/test
```

**Response:**

```json
{
  "status": "success",
  "latency": 1.23,
  "response": "Test successful",
  "metadata": {
    "model": "gpt-4-0613",
    "usage": {
      "promptTokens": 10,
      "completionTokens": 5,
      "totalTokens": 15
    }
  }
}
```

## System API

### Get System Status

```http
GET /api/v1/system/status
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.2.0",
  "uptime": 86400,
  "services": {
    "evaluationEngine": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "fileStorage": "healthy"
  },
  "statistics": {
    "totalEvaluations": 1247,
    "activeEvaluations": 3,
    "avgResponseTime": 1.2
  }
}
```

### Get API Usage

```http
GET /api/v1/system/usage
```

**Response:**

```json
{
  "current": {
    "requests": 1247,
    "evaluations": 23,
    "tokens": 45678
  },
  "limits": {
    "requests": 10000,
    "evaluations": 100,
    "tokens": 1000000
  },
  "resetDate": "2024-02-01T00:00:00Z"
}
```

## WebSocket Events

Connect to the WebSocket endpoint provided when creating an evaluation to receive real-time updates.

### Connection

```javascript
const ws = new WebSocket('wss://api.llmworks.dev/ws/eval_abc123?token=your-api-key');
```

### Event Types

#### Evaluation Progress

```json
{
  "type": "evaluation.progress",
  "data": {
    "evaluationId": "eval_abc123",
    "progress": 0.45,
    "currentStep": "Running benchmark 3/10",
    "estimatedTimeRemaining": 180
  }
}
```

#### Evaluation Complete

```json
{
  "type": "evaluation.complete",
  "data": {
    "evaluationId": "eval_abc123",
    "status": "completed",
    "results": {
      "winner": "gpt-4",
      "scores": {
        "gpt-4": 0.72,
        "claude-3-sonnet": 0.68
      }
    }
  }
}
```

#### Error Event

```json
{
  "type": "evaluation.error",
  "data": {
    "evaluationId": "eval_abc123",
    "error": "Model API timeout",
    "code": "MODEL_TIMEOUT",
    "retryable": true
  }
}
```

## Error Handling

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_MODEL_CONFIG",
    "message": "Model configuration is invalid",
    "details": {
      "field": "apiKey",
      "reason": "API key format is invalid"
    },
    "requestId": "req_abc123"
  }
}
```

### Common Error Codes

- `UNAUTHORIZED`: Invalid or missing API key
- `RATE_LIMITED`: Rate limit exceeded
- `INVALID_REQUEST`: Malformed request body
- `MODEL_UNAVAILABLE`: Specified model is not accessible
- `EVALUATION_NOT_FOUND`: Evaluation ID does not exist
- `INSUFFICIENT_CREDITS`: Not enough credits for evaluation

## Rate Limits

- **Standard**: 1000 requests/hour, 50 evaluations/day
- **Pro**: 5000 requests/hour, 200 evaluations/day
- **Enterprise**: Custom limits

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 956
X-RateLimit-Reset: 1642694400
```

## SDK Examples

### JavaScript/TypeScript

```typescript
import { LLMWorksClient } from '@llmworks/sdk';

const client = new LLMWorksClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.llmworks.dev'
});

// Start an arena battle
const battle = await client.arena.createBattle({
  modelA: 'gpt-4',
  modelB: 'claude-3-sonnet',
  prompt: 'Explain quantum computing',
  judgeModel: 'gpt-4'
});

// Monitor progress
battle.on('progress', (data) => {
  console.log(`Progress: ${data.progress * 100}%`);
});

battle.on('complete', (results) => {
  console.log('Winner:', results.winner);
});
```

### Python

```python
from llmworks import LLMWorksClient

client = LLMWorksClient(api_key='your-api-key')

# Run MMLU benchmark
evaluation = client.benchmarks.run(
    benchmark='mmlu',
    models=['gpt-4', 'claude-3-sonnet'],
    config={
        'few_shot': 5,
        'temperature': 0
    }
)

# Wait for completion
results = evaluation.wait_for_completion()
print(f"GPT-4 Score: {results['gpt-4']['overall']}")
```

### cURL Examples

```bash
# Create evaluation
curl -X POST https://api.llmworks.dev/api/v1/evaluations \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "arena",
    "models": [
      {
        "id": "gpt-4",
        "provider": "openai",
        "apiKey": "sk-..."
      }
    ]
  }'

# Get results
curl -X GET https://api.llmworks.dev/api/v1/evaluations/eval_abc123 \
  -H "Authorization: Bearer your-api-key"
```

## Webhook Integration

Configure webhooks to receive evaluation results automatically.

```http
POST /api/v1/webhooks
```

**Request Body:**

```json
{
  "url": "https://your-app.com/webhook",
  "events": [
    "evaluation.complete",
    "evaluation.failed"
  ],
  "secret": "webhook-secret-for-verification"
}
```

Webhook payloads include a signature header for verification:

```
X-LLMWorks-Signature: sha256=abc123...
```

For more examples and integration guides, visit our [Developer Portal](https://docs.llmworks.dev).