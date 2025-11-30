# Monitor

> Continuous monitoring with circuit breakers for fault tolerance

**Category:** infrastructure

## Commands

### `npm run ai:monitor`

Monitor CLI

### `npm run ai:monitor:status`

Show monitor status and circuit breaker states

### `npm run ai:monitor:check`

Check for changes and trigger actions

## Configuration

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| debounceMs | number | Debounce time for changes | 2000 |
| maxFrequencyMs | number | Minimum time between triggers | 30000 |
| failureThreshold | number | Circuit breaker failure threshold | 3 |
| resetTimeoutMs | number | Circuit breaker reset timeout | 60000 |

## Exports

```typescript
import { monitor, circuitBreaker } from 'tools/ai/monitor';
```

## Examples

### Check monitor status

```bash
npm run ai:monitor:status
```

