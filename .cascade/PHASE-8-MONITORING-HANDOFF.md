# 📊 PHASE 8: MONITORING & OBSERVABILITY - CASCADE HANDOFF

## Mission
Implement comprehensive monitoring, logging, and observability. AI-accelerated: 30-45 minutes.

## Context
- Token tracking already exists at `tools/ai/tokens.ts`
- Cost tracking at `alawein-technologies-llc/talai/cost_tracker.py`
- Need unified telemetry across all projects

---

## Tasks (Execute in Order)

### 1. Create Unified Telemetry Config (10 min)
Create `.config/telemetry/config.yaml`:
```yaml
version: "1.0"
telemetry:
  enabled: true
  
  # Log destinations
  destinations:
    file:
      enabled: true
      path: ".logs/"
      rotation: "daily"
      retention_days: 30
    
    console:
      enabled: true
      level: "info"
  
  # Metrics collection
  metrics:
    ai_tokens:
      enabled: true
      log_file: ".config/ai/logs/token-usage.jsonl"
    
    build_times:
      enabled: true
      log_file: ".logs/build-metrics.jsonl"
    
    test_results:
      enabled: true
      log_file: ".logs/test-metrics.jsonl"

  # Alerts
  alerts:
    token_budget_exceeded:
      threshold: 100000
      channel: "console"
    
    build_time_exceeded:
      threshold_seconds: 300
      channel: "console"
```

### 2. Create Telemetry CLI (15 min)
Create `tools/telemetry/index.ts`:
```typescript
#!/usr/bin/env tsx
/**
 * Unified telemetry CLI
 * Usage: npm run telemetry <command>
 */
import { readFileSync, existsSync, mkdirSync, appendFileSync } from 'fs';
import { join } from 'path';

const LOGS_DIR = '.logs';
const METRICS_FILE = join(LOGS_DIR, 'metrics.jsonl');

interface MetricEntry {
  timestamp: string;
  type: string;
  data: Record<string, unknown>;
}

function ensureLogsDir() {
  if (!existsSync(LOGS_DIR)) mkdirSync(LOGS_DIR, { recursive: true });
}

function logMetric(type: string, data: Record<string, unknown>) {
  ensureLogsDir();
  const entry: MetricEntry = {
    timestamp: new Date().toISOString(),
    type,
    data
  };
  appendFileSync(METRICS_FILE, JSON.stringify(entry) + '\n');
  console.log(`📊 Logged: ${type}`);
}

function showDashboard() {
  console.log('\n📊 TELEMETRY DASHBOARD\n');
  console.log('='.repeat(50));
  
  // Token usage
  const tokenLog = '.config/ai/logs/token-usage.jsonl';
  if (existsSync(tokenLog)) {
    const lines = readFileSync(tokenLog, 'utf-8').trim().split('\n');
    const total = lines.reduce((sum, line) => {
      try { return sum + JSON.parse(line).tokens; } catch { return sum; }
    }, 0);
    console.log(`🤖 AI Tokens Used: ${total.toLocaleString()}`);
  }
  
  // Build metrics
  if (existsSync(METRICS_FILE)) {
    const lines = readFileSync(METRICS_FILE, 'utf-8').trim().split('\n');
    console.log(`📈 Total Metrics Logged: ${lines.length}`);
  }
  
  console.log('='.repeat(50));
}

const [,, cmd, ...args] = process.argv;
switch (cmd) {
  case 'log':
    logMetric(args[0] || 'generic', { value: args[1] });
    break;
  case 'dashboard':
  case 'show':
    showDashboard();
    break;
  default:
    console.log('Usage: npm run telemetry <log|dashboard>');
}
```

### 3. Add Health Check Endpoints (10 min)
Create `tools/health/check.ts`:
```typescript
#!/usr/bin/env tsx
/**
 * Health check for all services
 */
const SERVICES = [
  { name: 'AI Proxy', url: 'http://localhost:4000/health' },
  { name: 'REPZ API', url: 'http://localhost:3000/api/health' },
];

async function checkHealth() {
  console.log('🏥 Health Check\n');
  for (const svc of SERVICES) {
    try {
      const res = await fetch(svc.url, { signal: AbortSignal.timeout(2000) });
      console.log(`  ${svc.name}: ${res.ok ? '✅ OK' : '❌ DOWN'}`);
    } catch {
      console.log(`  ${svc.name}: ⚪ Not running`);
    }
  }
}

checkHealth();
```

### 4. GitHub Actions Status Badge (5 min)
Update root `README.md` to include workflow badges (if not present):
```markdown
![CI](https://github.com/alawein/meta-governance/actions/workflows/ci.yml/badge.svg)
![Security](https://github.com/alawein/meta-governance/actions/workflows/security.yml/badge.svg)
```

### 5. Add npm Scripts (5 min)
Add to `package.json`:
```json
"telemetry": "tsx tools/telemetry/index.ts",
"telemetry:dashboard": "tsx tools/telemetry/index.ts dashboard",
"health": "tsx tools/health/check.ts"
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `.config/telemetry/config.yaml` | Create |
| `tools/telemetry/index.ts` | Create |
| `tools/health/check.ts` | Create |
| `package.json` | Add scripts |
| `.gitignore` | Add `.logs/` |

---

## Success Criteria

- [ ] `npm run telemetry dashboard` shows metrics
- [ ] `npm run health` checks service status
- [ ] `.logs/` directory for local logs
- [ ] Telemetry config YAML created

---

## Commit
`feat(monitoring): Complete Phase 8 monitoring & observability`

