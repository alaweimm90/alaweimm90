import type {
  AthleteProfile,
  DashboardResponse,
  MetricSnapshot,
  PlanItem,
  SessionSummary,
} from './types';

const demoAthlete: AthleteProfile = {
  id: 'athlete-001',
  name: 'Demo Athlete',
  goal: 'Increase bench press strength and overall training consistency',
  program: 'BenchBarrier Strength Phase 1',
  riskTier: 'moderate',
  adherence: 0.78,
  sleepHours: 6.6,
  lastUpdated: new Date().toISOString(),
};

const demoMetrics: MetricSnapshot = {
  adherence7d: 0.78,
  readinessScore: 7.1,
  fatigueRisk: 'moderate',
  weeklyVolume: 8750,
};

const demoWeeklyPlan: PlanItem[] = [
  {
    day: 'Mon',
    session: 'Upper Body Strength',
    focus: 'Bench, rows, accessories',
    status: 'completed',
  },
  {
    day: 'Tue',
    session: 'Conditioning',
    focus: 'Intervals + mobility',
    status: 'planned',
  },
  {
    day: 'Thu',
    session: 'Heavy Bench Technique',
    focus: 'Low-rep top sets',
    status: 'upcoming',
  },
  {
    day: 'Sat',
    session: 'Full Body Hypertrophy',
    focus: 'Accessory volume + core',
    status: 'upcoming',
  },
];

const demoSessions: SessionSummary[] = [
  {
    id: 'sess-001',
    athleteId: demoAthlete.id,
    date: '2025-11-17',
    type: 'strength',
    focus: 'Bench 5x5 @ 75% + rows',
    completed: true,
    rpe: 7,
  },
  {
    id: 'sess-002',
    athleteId: demoAthlete.id,
    date: '2025-11-18',
    type: 'conditioning',
    focus: 'Bike intervals + mobility',
    completed: true,
    rpe: 6,
  },
  {
    id: 'sess-003',
    athleteId: demoAthlete.id,
    date: '2025-11-20',
    type: 'strength',
    focus: 'Heavy singles @ RPE 8',
    completed: false,
  },
];

/**
 *
 * @param athleteId
 */
export default function getAthleteDashboardData(
  athleteId: string
): Pick<DashboardResponse, 'athlete' | 'metrics' | 'weeklyPlan' | 'sessions'> {
  const sessionsForAthlete = demoSessions.filter((s) => s.athleteId === athleteId);
  return {
    athlete: demoAthlete,
    metrics: demoMetrics,
    weeklyPlan: demoWeeklyPlan,
    sessions: sessionsForAthlete.length ? sessionsForAthlete : demoSessions,
  };
}
