# Cache

> Multi-layer caching with semantic similarity for AI operations

**Category:** infrastructure

## Commands

### `npm run ai:cache`

Cache management CLI

### `npm run ai:cache:stats`

Show cache statistics

### `npm run ai:cache:clear`

Clear cache entries

**Arguments:**

| Name | Type | Required | Description | Default |
|------|------|----------|-------------|---------|
| layer | string | No | Cache layer to clear (semantic, template, result, analysis) | - |

## Configuration

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| maxEntries | number | Maximum cache entries | 1000 |
| maxSizeBytes | number | Maximum cache size in bytes | 52428800 |
| defaultTtlMs | number | Default TTL in milliseconds | 3600000 |
| enableSemanticSimilarity | boolean | Enable semantic matching | true |
| similarityThreshold | number | Similarity threshold (0-1) | 0.85 |

## Exports

```typescript
import { cache } from 'tools/ai/cache';
```

## Examples

### View cache stats

```bash
npm run ai:cache:stats
```

### Clear semantic layer

```bash
npm run ai:cache:clear semantic
```

