# 🏥 Repo Health

Automated repository health checks that run in parallel without affecting other tasks.

## Quick Start

```bash
# Run all health checks
npx ts-node tools/health/repo-health.ts

# Or use the npm script
npm run health
```

## Checks Performed

| Check | Description | Auto-fixable |
|-------|-------------|--------------|
| **Dependencies** | Security vulnerabilities, outdated packages | ✅ `npm audit fix` |
| **Dead Code** | Unused exports, orphaned test files | ❌ Manual |
| **Documentation** | Broken links, stale docs (90+ days) | ❌ Manual |
| **Secrets** | Exposed API keys, passwords, tokens | ❌ Manual |
| **TypeScript** | Compilation errors, `any` type usage | ❌ Manual |
| **Git Health** | Large files in history, uncommitted changes | ⚠️ Partial |
| **Assets** | Unoptimized images (>500KB) | ✅ Compress |

## CI/CD Integration

The health check runs automatically:
- **Weekly**: Every Sunday at 2am UTC
- **On-demand**: Via GitHub Actions workflow dispatch

## Reports

Health reports are saved to `.orchex/health-report.json`:

```json
{
  "timestamp": "2025-12-06T07:30:00.000Z",
  "score": 85,
  "results": [
    { "name": "Dependencies", "status": "pass", "message": "All secure" },
    { "name": "TypeScript", "status": "warn", "message": "42 any types" }
  ]
}
```

## Adding New Checks

Add a new async function in `repo-health.ts`:

```typescript
async function checkMyThing(): Promise<HealthResult> {
  // Your check logic
  return {
    name: 'My Check',
    status: 'pass' | 'warn' | 'fail',
    message: 'Description',
    details: ['Optional', 'details'],
    fixable: true,
  };
}
```

Then add it to the `checks` array in `runAllChecks()`.
