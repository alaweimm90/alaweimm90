# ⚡ PHASE 7: PERFORMANCE OPTIMIZATION - CASCADE HANDOFF

## Mission
Optimize build performance, bundle sizes, and runtime efficiency. AI-accelerated: 20-40 minutes.

## Context
- Multiple TypeScript/React apps (REPZ, LiveItIconic, Attributa)
- Python packages (Librex, research projects)
- Current builds not optimized for production

---

## Tasks (Execute in Order)

### 1. Optimize TypeScript Build (10 min)
Update root `tsconfig.json`:
```json
{
  "compilerOptions": {
    "incremental": true,
    "tsBuildInfoFile": ".tsbuildinfo",
    "skipLibCheck": true,
    "moduleResolution": "bundler"
  }
}
```

### 2. Add Build Caching to CI (5 min)
Update `.github/workflows/ci.yml`:
```yaml
      - name: Cache node modules
        uses: actions/cache@v4
        with:
          path: |
            node_modules
            ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-node-

      - name: Cache Python packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

### 3. Bundle Analysis Script (10 min)
Create `tools/performance/bundle-analyzer.ts`:
```typescript
#!/usr/bin/env tsx
/**
 * Bundle size analyzer for all web apps
 * Usage: npm run perf:bundle
 */
import { execSync } from 'child_process';
import { readdirSync, statSync } from 'fs';
import { join } from 'path';

const WEB_APPS = [
  'repz-llc/repz',
  'live-it-iconic-llc/liveiticonic',
  'alawein-technologies-llc/attributa'
];

function analyzeBundles() {
  console.log('📦 Bundle Size Analysis\n');
  
  for (const app of WEB_APPS) {
    const distPath = join(app, 'dist');
    try {
      const files = readdirSync(distPath, { recursive: true }) as string[];
      let totalSize = 0;
      
      for (const file of files) {
        const filePath = join(distPath, file);
        const stat = statSync(filePath);
        if (stat.isFile()) totalSize += stat.size;
      }
      
      console.log(`${app}: ${(totalSize / 1024).toFixed(1)} KB`);
    } catch {
      console.log(`${app}: No dist folder`);
    }
  }
}

analyzeBundles();
```

### 4. Python Performance Baseline (5 min)
Create `tools/performance/python-benchmark.py`:
```python
#!/usr/bin/env python3
"""Quick performance benchmark for Python packages."""
import time
import sys

def benchmark_import(module: str) -> float:
    start = time.perf_counter()
    try:
        __import__(module)
        return time.perf_counter() - start
    except ImportError:
        return -1

if __name__ == "__main__":
    modules = ["equilibria", "mezan", "maglogic", "scicomp"]
    print("🐍 Python Import Times\n")
    for mod in modules:
        t = benchmark_import(mod)
        status = f"{t*1000:.1f}ms" if t > 0 else "not installed"
        print(f"  {mod}: {status}")
```

### 5. Add Turbo/nx for Monorepo (10 min)
Create `turbo.json` at root:
```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "build/**"]
    },
    "lint": {
      "dependsOn": ["^lint"]
    },
    "test": {
      "dependsOn": ["build"],
      "inputs": ["src/**", "tests/**"]
    }
  }
}
```

Add to `package.json`:
```json
"scripts": {
  "perf:bundle": "tsx tools/performance/bundle-analyzer.ts",
  "perf:python": "python tools/performance/python-benchmark.py"
}
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `tsconfig.json` | Add incremental build |
| `.github/workflows/ci.yml` | Add caching |
| `tools/performance/bundle-analyzer.ts` | Create |
| `tools/performance/python-benchmark.py` | Create |
| `turbo.json` | Create |
| `package.json` | Add perf scripts |

---

## Success Criteria

- [ ] TypeScript builds use incremental compilation
- [ ] CI caches node_modules and pip packages
- [ ] `npm run perf:bundle` works
- [ ] `npm run perf:python` works
- [ ] turbo.json configured for monorepo

---

## Commit
`feat(perf): Complete Phase 7 performance optimization`

