# Deployment Guide

Comprehensive guide for deploying LiveItIconic to production environments.

## Table of Contents

- [Deployment Platforms](#deployment-platforms)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Environment Configuration](#environment-configuration)
- [Vercel Deployment](#vercel-deployment)
- [Netlify Deployment](#netlify-deployment)
- [Docker Deployment](#docker-deployment)
- [Database Setup](#database-setup)
- [SSL/TLS Configuration](#ssltls-configuration)
- [CDN and Caching](#cdn-and-caching)
- [Monitoring and Alerts](#monitoring-and-alerts)
- [Rollback Procedures](#rollback-procedures)
- [Performance Optimization](#performance-optimization)

## Deployment Platforms

### Recommended: Vercel

- Optimized for Next.js but works great with Vite
- Built-in edge functions and API routes
- Automatic SSL and CDN
- Git integration with automatic deployments
- Free tier available

### Alternative: Netlify

- Great for static sites and SPA
- Serverless functions support
- Form handling and identity service
- Continuous deployment from Git
- Free tier available

### Alternative: Docker/Self-Hosted

- Full control and flexibility
- Custom infrastructure
- Higher setup and maintenance cost
- Recommended for enterprise deployments

## Pre-Deployment Checklist

### Code Quality

- [ ] All tests passing: `npm run test`
- [ ] No TypeScript errors: `npm run type-check`
- [ ] Linting passes: `npm run lint`
- [ ] Code coverage above 80%
- [ ] No console errors or warnings
- [ ] Security vulnerabilities fixed: `npm audit`

### Functionality

- [ ] User authentication working
- [ ] Products display correctly
- [ ] Shopping cart functional
- [ ] Checkout process tested end-to-end
- [ ] Payment processing tested (staging)
- [ ] Email notifications working
- [ ] Admin dashboard functional

### Performance

- [ ] Lighthouse score > 90
- [ ] Load time < 3 seconds
- [ ] Images optimized
- [ ] Code bundled and minified
- [ ] API response times acceptable

### Security

- [ ] Environment variables secured
- [ ] HTTPS enabled
- [ ] CORS properly configured
- [ ] Rate limiting implemented
- [ ] Input validation in place
- [ ] XSS protection enabled
- [ ] CSRF tokens in place

### Documentation

- [ ] README updated
- [ ] API documentation complete
- [ ] Environment variables documented
- [ ] Deployment instructions clear
- [ ] Runbooks created

## Environment Configuration

### Environment Variables

Create `.env.production` with production values:

```env
# Supabase (Production)
VITE_SUPABASE_URL=https://your-prod-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_prod_anon_key

# Stripe (Production)
VITE_STRIPE_PUBLIC_KEY=pk_live_xxxxxxxxxx

# API Configuration
VITE_API_URL=https://api.liviticonic.com

# Feature Flags
VITE_ENABLE_AI_AGENTS=true
VITE_ENABLE_PWA=true

# Analytics
VITE_ANALYTICS_ID=G-XXXXXXX

# Sentry Error Tracking
VITE_SENTRY_DSN=https://xxxxx@sentry.io/xxxxx

# Feature Toggles
VITE_FEATURE_CHECKOUT_V2=true
VITE_FEATURE_SUBSCRIPTIONS=false
```

### Build Configuration

Optimize Vite build settings in `vite.config.ts`:

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
  plugins: [react()],
  build: {
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-select']
        }
      }
    }
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify('production')
  }
});
```

## Vercel Deployment

### Setup

1. **Connect Repository**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Select your GitHub repository
   - Click Import

2. **Configure Project**
   - **Framework**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install --legacy-peer-deps`

3. **Environment Variables**
   - Add production environment variables
   - Use "Preview" and "Production" environments
   - Secrets won't be exposed to frontend (except those prefixed with `VITE_`)

### Deploy

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy to production
vercel --prod

# Preview deployment
vercel
```

### Configuration File

Create `vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install --legacy-peer-deps",
  "framework": "vite",
  "outputDirectory": "dist",
  "env": {
    "VITE_SUPABASE_URL": "@supabase_url",
    "VITE_STRIPE_PUBLIC_KEY": "@stripe_key"
  },
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

### Deploy on Git Push

Vercel automatically deploys when you push to main branch:

```bash
# Commit and push
git add .
git commit -m "feat: new feature"
git push origin main

# Vercel detects change and deploys automatically
# Check deployment status at https://vercel.com
```

## Netlify Deployment

### Setup

1. **Connect Repository**
   - Go to [netlify.com](https://app.netlify.com)
   - Click "New site from Git"
   - Select your GitHub repository

2. **Configure Build**
   - **Build command**: `npm run build`
   - **Publish directory**: `dist`
   - **Node version**: 18.x

3. **Add Env Variables**
   - Settings → Environment
   - Add all `VITE_*` variables

### Configuration File

Create `netlify.toml`:

```toml
[build]
  command = "npm run build"
  publish = "dist"
  environment = { NODE_VERSION = "18" }

[build.environment]
  VITE_SUPABASE_URL = "${VITE_SUPABASE_URL}"
  VITE_STRIPE_PUBLIC_KEY = "${VITE_STRIPE_PUBLIC_KEY}"

# Redirect all routes to index.html for SPA
[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

# Cache headers
[[headers]]
  for = "/dist/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

# API rewriting
[[redirects]]
  from = "/api/*"
  to = "https://api.liviticonic.com/:splat"
  status = 200
```

### Deploy

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy to production
netlify deploy --prod

# Preview deployment
netlify deploy
```

## Docker Deployment

### Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --legacy-peer-deps

COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

# Install serve to run the static site
RUN npm install -g serve

COPY --from=builder /app/dist ./dist

EXPOSE 3000

ENV NODE_ENV=production

CMD ["serve", "-s", "dist", "-l", "3000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - VITE_SUPABASE_URL=${VITE_SUPABASE_URL}
      - VITE_STRIPE_PUBLIC_KEY=${VITE_STRIPE_PUBLIC_KEY}
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - web
    restart: unless-stopped
```

### Build and Run

```bash
# Build Docker image
docker build -t liviticonic:latest .

# Run container
docker run -p 3000:3000 \
  -e VITE_SUPABASE_URL=your_url \
  -e VITE_STRIPE_PUBLIC_KEY=your_key \
  liviticonic:latest

# Using Docker Compose
docker-compose up -d
```

## Database Setup

### Supabase Production

1. **Create Production Project**
   - Go to supabase.com
   - Create new project
   - Configure security settings
   - Enable backups

2. **Run Migrations**
   ```bash
   # Update .env with production Supabase URL
   npm run migrate:production
   ```

3. **Seed Initial Data**
   ```bash
   npm run seed:production
   ```

4. **Security Rules**
   - Enable RLS (Row Level Security)
   - Configure policies
   - Restrict API access

### Backup Strategy

```sql
-- Daily backup scheduled in Supabase Dashboard
-- Retention: 30 days
-- Test restore procedures monthly
```

## SSL/TLS Configuration

### Automatic (Vercel/Netlify)

Both platforms automatically provision and renew SSL certificates:
- ✓ HTTPS enabled by default
- ✓ Auto-renewal 30 days before expiration
- ✓ Modern TLS 1.2+

### Custom Domain

```bash
# Vercel
vercel domains add liviticonic.com

# Netlify
netlify domains add liviticonic.com

# Update DNS records as instructed
```

### HSTS Header

Add HSTS header for security:

```typescript
// vite.config.ts
export default defineConfig({
  // ...
  server: {
    middlewares: [
      (req, res, next) => {
        res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
        next();
      }
    ]
  }
});
```

## CDN and Caching

### Static Asset Caching

```typescript
// In Vercel/Netlify headers configuration
{
  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    },
    {
      "source": "/",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=0, must-revalidate"
        }
      ]
    }
  ]
}
```

### API Caching

```typescript
// Cache API responses in service workers
if ('caches' in window) {
  const cacheName = 'api-v1';

  fetch('/api/products')
    .then(response => {
      if (response.ok) {
        caches.open(cacheName).then(cache => {
          cache.put('/api/products', response.clone());
        });
      }
      return response;
    })
    .catch(() => {
      return caches.match('/api/products');
    });
}
```

## Monitoring and Alerts

### Error Tracking (Sentry)

```typescript
// src/main.tsx
import * as Sentry from '@sentry/react';

Sentry.init({
  dsn: process.env.VITE_SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 0.1,
  release: `liviticonic@${APP_VERSION}`
});
```

### Performance Monitoring

```typescript
// Monitor Core Web Vitals
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

getCLS(console.log);
getFID(console.log);
getFCP(console.log);
getLCP(console.log);
getTTFB(console.log);
```

### Application Monitoring

Set up monitoring for:
- Server uptime and response times
- Database query performance
- API endpoint latency
- Error rates and types
- User session metrics

**Recommended Tools**:
- Sentry - Error tracking
- LogRocket - Session replay
- Datadog - Infrastructure monitoring
- New Relic - Application performance

### Alerts Configuration

Configure alerts for:
- Error rate > 1%
- API response time > 1000ms
- Database query time > 500ms
- Deployment failures
- SSL certificate expiration

## Rollback Procedures

### Rollback on Vercel

```bash
# View deployment history
vercel list

# Rollback to previous deployment
vercel rollback

# Or manually redeploy specific commit
git checkout <commit-hash>
git push origin main
```

### Rollback on Netlify

```bash
# View deployment history in Netlify Dashboard
# Settings → Deployments → Deployment History

# Click "Publish Deploy" on previous deployment
# Or use Netlify CLI
netlify deploy --prod --dir=dist --alias <deployment-id>
```

### Database Rollback

```bash
# Supabase - restore from backup
# Settings → Backups → Restore

# Or manually run migration rollback
npm run migrate:rollback
```

### Health Checks

After rollback, verify:

```bash
# Check API endpoints
curl https://api.liviticonic.com/health

# Verify database
npm run test:db-connection

# Check authentication
curl -X POST https://api.liviticonic.com/auth/me

# Test critical user flows
npm run test:e2e
```

## Performance Optimization

### Build Optimization

```bash
# Analyze bundle size
npm run build --analyze

# Check Lighthouse score
npm run test:lighthouse
```

### Image Optimization

```typescript
// Use next-gen formats with fallbacks
<picture>
  <source srcSet="image.webp" type="image/webp" />
  <source srcSet="image.jpg" type="image/jpeg" />
  <img src="image.jpg" alt="Description" loading="lazy" />
</picture>
```

### Code Splitting

```typescript
// Lazy load routes
const AdminDashboard = lazy(() =>
  import('@/pages/admin/Dashboard')
);

const Checkout = lazy(() =>
  import('@/pages/Checkout')
);
```

### Service Worker

```typescript
// Enable PWA with automatic updates
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js')
    .then(reg => {
      reg.addEventListener('updatefound', () => {
        const newWorker = reg.installing;
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed') {
            showUpdatePrompt(); // Notify user of update
          }
        });
      });
    });
}
```

## Deployment Checklist

Before going live:

- [ ] All tests passing
- [ ] No console errors
- [ ] Performance optimized (Lighthouse > 90)
- [ ] Security audit completed
- [ ] Environment variables set
- [ ] Database migrations run
- [ ] Backups configured
- [ ] Monitoring enabled
- [ ] Error tracking active
- [ ] SSL certificate valid
- [ ] CDN cache configured
- [ ] Rate limiting enabled
- [ ] API documentation updated
- [ ] Runbooks created
- [ ] Team trained
- [ ] Communication plan ready

## Deployment Timeline

### 1 Week Before
- Code freeze
- Testing phase
- Performance audit
- Security review

### 1 Day Before
- Final testing
- Backup verification
- Rollback testing
- Team notification

### Deployment Day
- 6:00 AM - Health checks
- 7:00 AM - Deploy to staging
- 8:00 AM - Staging verification
- 9:00 AM - Deploy to production
- 10:00 AM - Production verification
- 10:30 AM - Team debriefing

### After Deployment
- Monitor error rates
- Check key metrics
- Verify user reports
- Plan follow-up releases

## Related Documentation

- [Architecture Guide](./ARCHITECTURE.md)
- [Development Guide](./DEVELOPMENT.md)
- [API Documentation](./API.md)
- [Vercel Documentation](https://vercel.com/docs)
- [Netlify Documentation](https://docs.netlify.com)
- [Supabase Documentation](https://supabase.com/docs)

---

For deployment issues or questions, refer to the platform-specific documentation or contact your DevOps team.
