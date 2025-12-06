# Deployment Troubleshooting Guide

This guide helps resolve common deployment issues on Lovable.dev and other hosting platforms.

## Fixed Issues

### ✅ Environment Variable Compatibility
**Issue**: Using `process.env.NODE_ENV` in Vite/browser environment
**Fix**: Replaced with `import.meta.env.PROD` and `import.meta.env.DEV`

**Files Modified**:
- `src/lib/monitoring.ts`
- `src/lib/security.ts`

### ✅ Node.js Version Specification
**Issue**: Build platform using different Node.js version
**Fix**: Added version constraints

**Files Added**:
- `.nvmrc` - Specifies Node.js 20.16.0
- `package.json` - Added engines specification

### ✅ Environment Variables Documentation
**Issue**: Missing environment variable documentation
**Fix**: Created comprehensive environment variable guide

**Files Added**:
- `.env.example` - Documents all optional environment variables

### ✅ Dependency Installation
**Issue**: Build failing due to dependency issues
**Fix**: Tested clean installation and verified all dependencies

## Common Build Issues and Solutions

### 1. Environment Variable Issues

**Symptoms**:
- Build fails with "import.meta.env.VITE_X is undefined"
- Runtime errors about missing configuration

**Solutions**:
- All environment variables are optional in this project
- Check `.env.example` for available variables
- Use default values for missing variables

### 2. Node.js Version Mismatch

**Symptoms**:
- Build fails with syntax errors
- "Unsupported engine" errors

**Solutions**:
- Project requires Node.js >= 18.0.0
- Lovable.dev should use Node.js 20.16.0 (specified in `.nvmrc`)
- Local development: Use `nvm use` to switch to correct version

### 3. Import Path Issues

**Symptoms**:
- "Module not found" errors
- Case sensitivity issues

**Solutions**:
- All imports use `@/` path alias
- File paths are case-sensitive on production
- Verify component file names match import statements

### 4. Build Memory/Time Limits

**Symptoms**:
- Build timeout errors
- Out of memory errors

**Solutions**:
- Build is optimized with manual chunking
- Average build time: ~50 seconds
- If timeout occurs, retry deployment

## Verification Checklist

Before deployment, verify:

- [ ] `npm install` succeeds
- [ ] `npm run build` succeeds
- [ ] No TypeScript errors
- [ ] All imports use correct casing
- [ ] Environment variables are optional
- [ ] Node.js version is compatible

## Platform-Specific Notes

### Lovable.dev
- Uses GitHub sync for deployments
- Automatically detects build command from package.json
- Environment variables configured in dashboard
- Uses `.nvmrc` for Node.js version

### Vercel
- Set `VITE_ENVIRONMENT=production` in dashboard
- Configure Node.js version to 20.x

### Netlify
- Build command: `npm run build`
- Publish directory: `dist`
- Node.js version: 20.16.0

## Environment Variables Reference

All environment variables are **optional** with sensible defaults:

```bash
# API Configuration
VITE_API_URL=               # Defaults to environment-specific URLs
VITE_CDN_URL=               # Optional CDN for assets
VITE_WS_URL=                # WebSocket URL for real-time features

# Security
VITE_API_KEY_ENCRYPTION_SALT=  # For API key encryption (has default)

# Monitoring
VITE_SENTRY_DSN=            # Error tracking (optional)
VITE_VAPID_PUBLIC_KEY=      # Push notifications (optional)

# Environment
VITE_ENVIRONMENT=           # Auto-detected from build mode
```

## Build Performance

- **Average build time**: 45-55 seconds
- **Bundle size**: ~1.2MB (gzipped)
- **Chunk strategy**: Vendor splitting for optimal caching
- **Target**: ES2020 for modern browser support

## Getting Help

If deployment still fails:

1. Check the full build logs on Lovable.dev
2. Compare local build with production environment
3. Verify all environment variables are configured
4. Ensure Git repository is up to date

## Recent Fixes Applied

- ✅ Fixed environment variable compatibility
- ✅ Added Node.js version specification  
- ✅ Created environment variable documentation
- ✅ Tested clean dependency installation
- ✅ Verified import path consistency