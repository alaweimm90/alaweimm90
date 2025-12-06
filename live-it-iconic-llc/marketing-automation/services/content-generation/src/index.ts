import express, { Application } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import dotenv from 'dotenv';
import { contentRouter } from './routes/content.routes';
import { logger } from './utils/logger';
import { errorHandler } from './middleware/error-handler';
import { rateLimiter } from './middleware/rate-limiter';

dotenv.config();

const app: Application = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors());
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
app.use(rateLimiter);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'content-generation', timestamp: new Date() });
});

// Routes
app.use('/api/v1/content', contentRouter);

// Error handling
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
  logger.info(`Content Generation Service running on port ${PORT}`);
});

export default app;
