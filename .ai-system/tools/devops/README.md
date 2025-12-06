# DevOps Tools

DevOps automation and CLI tools for template deployment and infrastructure management.

## Purpose

This directory contains DevOps-specific tooling that builds on the shared library (`../lib/`).

## Related

- Shared utilities: `tools/lib/config.ts`, `tools/lib/fs.ts`
- Templates: `templates/devops/`
- Policies: `.metaHub/policies/`

## Usage

DevOps functionality is primarily accessed through npm scripts:

```bash
# Validate templates
npm run devops:validate

# Deploy templates
npm run devops:deploy

# List available templates
npm run devops:list
```

## Note

Core utilities have been consolidated into `tools/lib/` for reuse across the codebase.
