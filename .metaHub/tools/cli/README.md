# mh - MetaHub Unified CLI

Single entry point for all portfolio tools. **13 command groups | 8 templates | 30+ subcommands**

## Installation

```powershell
# Windows - add to PATH or use directly
pip install -r .metaHub/tools/cli/requirements.txt

# Run from repo root
python .metaHub/tools/cli/mh.py --help
# OR
.\.metaHub\tools\mh.cmd --help
```

## Commands Quick Reference

| Command | Purpose | Subcommands |
|---------|---------|-------------|
| `govern` | Portfolio governance | enforce, catalog, audit |
| `validate` | Validation | schema, structure, docker |
| `security` | Security scanning | scan-all, scan-secrets, scan-deps |
| `config` | Configuration | load, merge |
| `ci` | CI/CD workflows | validate, list-workflows |
| `log` | Logging utilities | setup |
| `supabase` | Supabase scaffolding | init, gen-auth, gen-types |
| `analytics` | Analytics setup | init (ga4/gtm/plausible) |
| `stripe` | Payment integration | gen-service |
| `template` | Code generation | list, apply |
| `ecommerce` | E-commerce patterns | init, gen-cart, gen-order, gen-checkout |
| `predict` | ML pipelines | init, train |
| `orchestrate` | Workflow orchestration | checkpoint, telemetry, recover, status, degrade |

---

## Governance
```bash
mh govern enforce ./organizations/my-org/   # Enforce policies
mh govern catalog --format json             # Generate catalog
mh govern audit --output report.md          # AI audit
```

## Validation
```bash
mh validate schema config.yaml --schema schema.json
mh validate structure ./my-repo --tier 2
mh validate docker ./Dockerfile
```

## Security
```bash
mh security scan-all ./                     # All scans
mh security scan-secrets ./                 # Secrets only
mh security scan-deps ./                    # Dependencies
```

## Configuration
```bash
mh config load config.yaml                  # Load & display
mh config merge base.yaml env.yaml -o out.yaml
```

## CI/CD
```bash
mh ci validate .github/workflows/ci.yml
mh ci list-workflows ./
```

## Logging
```bash
mh log setup --level INFO                   # Python snippet
mh log setup --level DEBUG --json           # JSON format
```

## Supabase
```bash
mh supabase init ./src/lib                  # Create client
mh supabase gen-auth ./src/services         # Auth service
mh supabase gen-types -p PROJECT_ID         # TypeScript types
```

## Analytics
```bash
mh analytics init ./src/lib                 # GA4 setup
mh analytics init ./src/lib --provider gtm  # GTM setup
mh analytics init ./src/lib --provider plausible
```

## Stripe
```bash
mh stripe gen-service ./src/services        # Payment service
```

## Templates
```bash
mh template list                            # Show all templates
mh template apply supabase/authService ./out.ts -v PROJECT=myapp
```

## E-commerce
```bash
mh ecommerce init ./src                     # All e-commerce services
mh ecommerce gen-cart ./src/services        # Cart service only
mh ecommerce gen-order ./src/services       # Order service only
mh ecommerce gen-checkout ./src/hooks       # Checkout hook only
```

## ML Prediction
```bash
mh predict init ./ml                        # Create pipeline scaffold
mh predict train data.csv -t target -m rf   # Train model
```

## Orchestration
```bash
mh orchestrate status                       # Show workflow status
mh orchestrate checkpoint create -w deploy  # Create checkpoint
mh orchestrate checkpoint restore --id abc  # Restore checkpoint
mh orchestrate recover -w deploy            # Recover failed workflow
mh orchestrate telemetry report --period 7d # Metrics report
mh orchestrate degrade --level 2            # Set degradation level
```

---

## Templates Available

```
supabase/
  - client.ts          (Supabase client init)
  - authService.ts     (Authentication service)
analytics/
  - analytics.ts       (GA4/GTM/Plausible)
stripe/
  - paymentService.ts  (Stripe integration)
ecommerce/
  - cartService.ts     (Shopping cart)
  - orderService.ts    (Order management)
  - useCheckout.ts     (React checkout hook)
predict/
  - pipeline.py        (ML training pipeline)
```

## Adding New Commands

1. Add command group in `cli/mh.py`
2. Import from existing modules in `tools/cli/` or `tools/lib/`
3. Test with `python .metaHub/tools/cli/mh.py <command> --help`

## Related Tools

| Tool | Location | Status |
|------|----------|--------|
| governance.py | tools/cli/governance.py | ✅ Integrated |
| validation.py | tools/lib/validation.py | ✅ Integrated |
| security scripts | tools/security/*.sh | ✅ Wrapped |
| orchestration | tools/legacy/orchestration/*.py | ✅ Wrapped |
