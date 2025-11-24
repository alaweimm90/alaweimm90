import type { AthleteProfile, MetricSnapshot, RiskAssessment } from './types';

/**
 *
 * @param athlete
 * @param metrics
 */
export function calculateRisk(athlete: AthleteProfile, metrics: MetricSnapshot): RiskAssessment {
  let score = 0;
  const factors: string[] = [];

  if (metrics.adherence7d < 0.7) {
    score += 30;
    factors.push('low_adherence_7d');
  }

  if (athlete.sleepHours < 7) {
    score += 20;
    factors.push('low_sleep');
  }

  if (metrics.weeklyVolume > 9000 && athlete.riskTier !== 'low') {
    score += 25;
    factors.push('high_volume_relative_to_risk');
  }

  if (metrics.readinessScore < 6) {
    score += 25;
    factors.push('low_readiness');
  }

  let tier: RiskAssessment['tier'];
  if (score < 30) tier = 'low';
  else if (score < 60) tier = 'moderate';
  else tier = 'high';

  return {
    score,
    tier,
    contributingFactors: factors,
  };
}

/**
 *
 * @param athlete
 * @param metrics
 * @param risk
 */
export function generateRecommendations(
  athlete: AthleteProfile,
  metrics: MetricSnapshot,
  risk: RiskAssessment,
): string[] {
  const recs: string[] = [];

  if (metrics.adherence7d < 0.8) {
    recs.push(
      'Focus this week on hitting all planned sessions before adding extra volume or intensity.',
    );
  } else {
    recs.push('You are consistent—consider a small progressive overload on key lifts (2–5%).');
  }

  if (athlete.sleepHours < 7) {
    recs.push(
      'Aim for at least 7 hours of sleep; add 1–2 low-intensity recovery sessions and a fixed bedtime.',
    );
  }

  if (risk.tier === 'high') {
    recs.push(
      'Current fatigue risk is high—discuss a deload week with your coach and temporarily reduce intensity.',
    );
  } else if (risk.tier === 'moderate') {
    recs.push(
      'Fatigue risk is moderate—keep heavy sets around RPE 6–7 and monitor soreness and joint pain.',
    );
  }

  recs.push('Upload at least one bench press form video for coach review this week.');

  return recs;
}
