import type { ErrorRequestHandler } from 'express';
import express from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { authenticate, authorizeRole } from './auth';
import getAthleteDashboardData from './data';
import { calculateRisk, generateRecommendations } from './risk';
import type { DashboardResponse } from './types';

const app = express();

app.use(express.json());
app.use(
  helmet({
    contentSecurityPolicy: false,
    hsts: true,
    referrerPolicy: { policy: 'no-referrer' },
  })
);

const limiter = rateLimit({
  windowMs: 60_000,
  max: 60,
});

app.use(limiter);

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'coaching-api' });
});

app.use(authenticate);

app.get('/v1/dashboard/:athleteId', authorizeRole(['coach', 'athlete', 'admin']), (req, res) => {
  const { athlete, metrics, weeklyPlan, sessions } = getAthleteDashboardData(req.params.athleteId);

  const risk = calculateRisk(athlete, metrics);
  const recommendations = generateRecommendations(athlete, metrics, risk);

  const payload: DashboardResponse = {
    athlete,
    metrics,
    weeklyPlan,
    sessions,
    risk,
    recommendations,
  };

  res.json(payload);
});

const port = Number(process.env.COACHING_API_PORT || 4100);

app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`coaching-api listening on port ${port}`);
});

const errorHandler: ErrorRequestHandler = (err, _req, res) => {
  res.status(500).json({ error: 'internal_error' });
};

app.use(errorHandler);
