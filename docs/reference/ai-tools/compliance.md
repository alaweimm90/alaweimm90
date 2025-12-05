# Compliance

> Policy-based validation with quantitative scoring and recommendations

**Category:** governance

## Commands

### `npm run ai:compliance`

Compliance CLI

### `npm run ai:compliance:check`

Run compliance check on files

**Arguments:**

| Name  | Type     | Required | Description    | Default |
| ----- | -------- | -------- | -------------- | ------- |
| files | string[] | No       | Files to check | -       |

### `npm run ai:compliance:score`

Quick compliance score check

## Exports

```typescript
import { compliance } from 'tools/ai/compliance';
```

## Examples

### Run compliance check

```bash
npm run ai:compliance:check src/api.ts
```

### Quick score

```bash
npm run ai:compliance:score
```
