import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { createProxyMiddleware } from 'http-proxy-middleware';
import dotenv from 'dotenv';
import { authMiddleware } from './middleware/auth';
import { logger } from './utils/logger';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});

app.use(limiter);
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'api-gateway', timestamp: new Date() });
});

// Authentication endpoints (no auth middleware)
app.post('/api/v1/auth/login', async (req, res) => {
  // Login logic
  res.json({ success: true, token: 'jwt_token' });
});

app.post('/api/v1/auth/register', async (req, res) => {
  // Registration logic
  res.json({ success: true, message: 'User registered' });
});

app.post('/api/v1/auth/refresh', async (req, res) => {
  // Token refresh logic
  res.json({ success: true, token: 'new_jwt_token' });
});

// Protected routes with authentication
app.use('/api/v1', authMiddleware);

// Service routing
const services = {
  content: process.env.CONTENT_SERVICE_URL || 'http://localhost:3001',
  social: process.env.SOCIAL_SERVICE_URL || 'http://localhost:3002',
  email: process.env.EMAIL_SERVICE_URL || 'http://localhost:3003',
  visual: process.env.VISUAL_SERVICE_URL || 'http://localhost:3004',
  scheduling: process.env.SCHEDULING_SERVICE_URL || 'http://localhost:3005',
  analytics: process.env.ANALYTICS_SERVICE_URL || 'http://localhost:3006',
  integrations: process.env.INTEGRATIONS_SERVICE_URL || 'http://localhost:3007',
  personalization: process.env.PERSONALIZATION_SERVICE_URL || 'http://localhost:3008',
  compliance: process.env.COMPLIANCE_SERVICE_URL || 'http://localhost:3009'
};

// Proxy middleware for each service
Object.entries(services).forEach(([name, url]) => {
  app.use(
    `/api/v1/${name}`,
    createProxyMiddleware({
      target: url,
      changeOrigin: true,
      pathRewrite: {
        [`^/api/v1/${name}`]: '/api/v1'
      },
      onError: (err, req, res) => {
        logger.error(`Proxy error for ${name}:`, err);
        res.status(503).json({
          success: false,
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: `${name} service is currently unavailable`
          }
        });
      },
      onProxyReq: (proxyReq, req: any) => {
        // Forward user information from JWT
        if (req.user) {
          proxyReq.setHeader('X-User-Id', req.user.id);
          proxyReq.setHeader('X-Organization-Id', req.user.organizationId);
          proxyReq.setHeader('X-User-Role', req.user.role);
        }
      }
    })
  );
});

// Error handling
app.use((err: any, req: any, res: any, next: any) => {
  logger.error('Gateway error:', err);
  res.status(500).json({
    success: false,
    error: {
      code: 'INTERNAL_ERROR',
      message: process.env.NODE_ENV === 'production' ? 'An error occurred' : err.message
    }
  });
});

// Start server
app.listen(PORT, () => {
  logger.info(`API Gateway running on port ${PORT}`);
  logger.info('Service routes:');
  Object.entries(services).forEach(([name, url]) => {
    logger.info(`  - /${name} -> ${url}`);
  });
});

export default app;
