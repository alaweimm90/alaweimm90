# Backstage Dockerfile
# Lightweight standalone Backstage instance for development

FROM node:20-alpine AS base
WORKDIR /app

# Install dependencies
RUN apk add --no-cache python3 make g++ curl

# Create a minimal Backstage app
FROM base AS builder

# Install Backstage CLI
RUN npm install -g @backstage/create-app@latest

# Create Backstage app (this will be minimal for now)
RUN npx @backstage/create-app --skip-install backstage-app || echo "Using existing structure"

WORKDIR /app

# Copy our custom configuration
COPY app-config.yaml ./app-config.yaml
COPY catalog-info.yaml ./catalog-info.yaml

# Create minimal package.json for standalone mode
RUN cat > package.json <<EOF
{
  "name": "backstage-portal",
  "version": "1.0.0",
  "private": true,
  "engines": {
    "node": "18 || 20"
  },
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "js-yaml": "^4.1.0"
  }
}
EOF

# Create a simple Express server to serve the catalog
RUN cat > server.js <<'EOF'
const express = require('express');
const fs = require('fs');
const yaml = require('js-yaml');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Health check endpoint
app.get('/healthcheck', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// API endpoint for catalog
app.get('/api/catalog/entities', (req, res) => {
  try {
    const catalogData = yaml.load(fs.readFileSync('/app/catalog-info.yaml', 'utf8'));
    res.json(catalogData);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Service catalog compatibility
app.get('/api/services', (req, res) => {
  try {
    const serviceCatalog = JSON.parse(fs.readFileSync('/app/service-catalog.json', 'utf8'));
    res.json(serviceCatalog);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Root
app.get('/', (req, res) => {
  res.send(`
    <html>
      <head><title>Backstage Portal - Multi-Org Platform</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px;">
        <h1>🎯 Backstage Developer Portal</h1>
        <p>Welcome to the Multi-Organization Platform Service Catalog</p>

        <h2>Available Endpoints:</h2>
        <ul>
          <li><a href="/healthcheck">/healthcheck</a> - Health status</li>
          <li><a href="/api/catalog/entities">/api/catalog/entities</a> - Backstage catalog (YAML)</li>
          <li><a href="/api/services">/api/services</a> - Service catalog (JSON)</li>
        </ul>

        <h2>📊 System Status</h2>
        <p>Status: <strong style="color: green;">✅ Running</strong></p>
        <p>Port: ${PORT}</p>
        <p>Node: ${process.version}</p>

        <h2>🔗 Quick Links</h2>
        <ul>
          <li><a href="http://localhost:3000">SimCore</a></li>
          <li><a href="http://localhost:8080">Repz</a></li>
          <li><a href="http://localhost:8081">BenchBarrier</a></li>
          <li><a href="http://localhost:3001">Attributa</a></li>
          <li><a href="http://localhost:8085">AI Agent Demo</a></li>
          <li><a href="http://localhost:8086">API Gateway</a></li>
          <li><a href="http://localhost:8087">Dashboard</a></li>
          <li><a href="http://localhost:8088">Healthcare</a></li>
        </ul>

        <p style="margin-top: 40px; color: #666;">
          <em>Note: For full Backstage UI, install and run locally with <code>npx @backstage/create-app</code></em>
        </p>
      </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`Backstage Portal running on http://localhost:${PORT}`);
  console.log(`Health: http://localhost:${PORT}/healthcheck`);
  console.log(`Catalog: http://localhost:${PORT}/api/catalog/entities`);
});
EOF

# Install dependencies
RUN npm install --production

# Production stage
FROM node:20-alpine AS production
WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy from builder
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/server.js ./server.js

# Copy config files (will be mounted as volumes)
COPY app-config.yaml ./app-config.yaml
COPY catalog-info.yaml ./catalog-info.yaml

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nodejs && \
    chown -R nodejs:nodejs /app

USER nodejs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/healthcheck || exit 1

CMD ["node", "server.js"]
