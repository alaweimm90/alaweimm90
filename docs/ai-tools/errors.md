# Errors

> Structured error handling with automatic recovery strategies

**Category:** infrastructure

## Commands

### `npm run ai:errors`

Error handler CLI

### `npm run ai:errors:list`

List recent errors

**Arguments:**

| Name     | Type   | Required | Description                                      | Default |
| -------- | ------ | -------- | ------------------------------------------------ | ------- |
| severity | string | No       | Filter by severity (low, medium, high, critical) | -       |

### `npm run ai:errors:stats`

Show error statistics

## Exports

```typescript
import { errorHandler, ErrorCodes, AIOperationError } from 'tools/ai/errors';
```

## Examples

### List critical errors

```bash
npm run ai:errors:list critical
```

### View error stats

```bash
npm run ai:errors:stats
```
