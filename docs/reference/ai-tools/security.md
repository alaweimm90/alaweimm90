# Security

> Security scanning for secrets, vulnerabilities, and license compliance

**Category:** governance

## Commands

### `npm run ai:security`

Security scanner CLI

### `npm run ai:security:scan`

Run full security scan

### `npm run ai:security:secrets`

Scan for exposed secrets

### `npm run ai:security:vulns`

Scan for vulnerabilities

## Exports

```typescript
import { securityScanner } from 'tools/ai/security';
```

## Examples

### Run full scan

```bash
npm run ai:security:scan
```

### Check for secrets

```bash
npm run ai:security:secrets src/
```
