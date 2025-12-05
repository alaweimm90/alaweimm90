# @atlas/cli

ATLAS Multi-Agent Orchestration Platform - A production-ready system for routing tasks to AI agents (Claude, GPT-4, Gemini) with automatic failover, circuit breakers, and intelligent load balancing.

**Directory Structure**:

- **Implementation**: `tools/atlas/` (tracked in git)
- **Runtime Data**: `.atlas/` (gitignored - reports, metrics, evidence)

## Features

- **Multi-Agent Routing** - Route tasks to the best available AI agent based on capability, cost, or latency
- **Circuit Breaker Pattern** - Automatic failover with configurable thresholds
- **LLM Adapters** - Native support for Anthropic (Claude), OpenAI (GPT-4), and Google (Gemini)
- **REST API** - Native HTTP server with authentication
- **Storage Backends** - JSON files or SQLite with migration support
- **Docker Ready** - Production-ready containerization

## Installation

```bash
npm install @atlas/cli
```

## Quick Start

### CLI Usage

```bash
# Start the API server
atlas-api

# Or use the CLI directly
atlas execute --type code_generation --description "Write a hello world function"
```

### Programmatic Usage

```typescript
import { executeTask, agentRegistry } from '@atlas/cli';

// Execute a task with automatic agent selection
const result = await executeTask({
  id: 'task-1',
  type: 'code_generation',
  description: 'Generate a hello world function in TypeScript',
  context: { language: 'typescript' },
  priority: 'medium',
  status: 'pending',
  createdAt: new Date().toISOString(),
});

console.log(result);
```

### API Server

```typescript
import { startServer } from '@atlas/cli/api';

// Start server on port 3000
const server = await startServer({ port: 3000 });

// Or with API key authentication
const server = await startServer({
  port: 3000,
  apiKey: process.env.ATLAS_API_KEY,
});
```

## Configuration

Set environment variables for API keys:

```bash
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key
export GOOGLE_AI_API_KEY=your-key
export ATLAS_API_KEY=your-atlas-api-key  # For API authentication
```

## API Endpoints

| Endpoint      | Method | Description            |
| ------------- | ------ | ---------------------- |
| `/health`     | GET    | Health check           |
| `/status`     | GET    | Detailed system status |
| `/agents`     | GET    | List all agents        |
| `/agents/:id` | GET    | Get agent details      |
| `/execute`    | POST   | Execute a task         |
| `/generate`   | POST   | Generate code          |
| `/review`     | POST   | Review code            |
| `/explain`    | POST   | Explain code           |
| `/chat`       | POST   | Chat with an agent     |

## Storage

ATLAS supports pluggable storage backends:

```typescript
import { initializeStorage } from '@atlas/cli/storage';

// Use JSON files (default)
await initializeStorage({ backend: 'json', basePath: '.atlas/data' });

// Use SQLite
await initializeStorage({ backend: 'sqlite', connectionString: '.atlas/atlas.db' });
```

### Migration

Migrate data between backends:

```bash
npx atlas-migrate json sqlite --verbose
```

## Docker

```bash
# Build and run
docker-compose -f docker/atlas/docker-compose.yml up --build

# Development mode with hot reload
docker-compose -f docker/atlas/docker-compose.yml --profile dev up
```

## Routing Strategies

- **capability** - Match agent capabilities to task type (default)
- **load_balance** - Distribute tasks evenly across agents
- **cost** - Select cheapest capable agent
- **latency** - Select fastest agent based on historical data

## License

MIT
