import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import dotenv from 'dotenv';
import { socialMediaRouter } from './routes/social.routes';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3002;

app.use(helmet());
app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'social-media', timestamp: new Date() });
});

app.use('/api/v1/social', socialMediaRouter);

app.listen(PORT, () => {
  console.log(`Social Media Service running on port ${PORT}`);
});

export default app;
