const express = require('express');
const fs = require('fs');
const yaml = require('js-yaml');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Health check endpoint
app.get('/healthcheck', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    service: 'backstage-portal'
  });
});

// Catalog entities endpoint (YAML format)
app.get('/api/catalog/entities', (req, res) => {
  try {
    const catalogData = yaml.load(fs.readFileSync('/app/catalog-info.yaml', 'utf8'));
    res.setHeader('Content-Type', 'application/x-yaml');
    res.send(yaml.dump(catalogData));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Services endpoint (JSON format)
app.get('/api/services', (req, res) => {
  try {
    if (fs.existsSync('/app/service-catalog.json')) {
      const serviceData = JSON.parse(fs.readFileSync('/app/service-catalog.json', 'utf8'));
      res.json(serviceData);
    } else {
      // Fallback: parse catalog-info.yaml and convert to JSON
      const catalogData = yaml.load(fs.readFileSync('/app/catalog-info.yaml', 'utf8'));
      res.json(catalogData);
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'Backstage Developer Portal',
    version: '1.0.0',
    description: 'Lightweight service catalog and developer portal',
    endpoints: {
      healthcheck: '/healthcheck',
      catalog: '/api/catalog/entities',
      services: '/api/services'
    },
    documentation: 'https://backstage.io/docs'
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Backstage portal listening on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/healthcheck`);
  console.log(`Catalog API: http://localhost:${PORT}/api/catalog/entities`);
  console.log(`Services API: http://localhost:${PORT}/api/services`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});
