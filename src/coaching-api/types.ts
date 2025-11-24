export interface AthleteProfile {
  id: string;
  name: string;
  goal: string;
  program: string;
  riskTier: 'low' | 'moderate' | 'high';
  adherence: number; // 0–1 over the last 7 days
  sleepHours: number; // average hours per night
  lastUpdated: string; // ISO timestamp
}

export type SessionType = 'strength' | 'conditioning' | 'mobility' | 'recovery';

export interface SessionSummary {
  id: string;
  athleteId: string;
  date: string; // ISO date
  type: SessionType;
  focus: string;
  completed: boolean;
  rpe?: number;
}

export type PlanStatus = 'completed' | 'planned' | 'upcoming';

export interface PlanItem {
  day: string; // e.g. 'Mon'
  session: string;
  focus: string;
  status: PlanStatus;
}

export type FatigueTier = 'low' | 'moderate' | 'high';

export interface MetricSnapshot {
  adherence7d: number; // 0–1
  readinessScore: number; // 0–10
  fatigueRisk: FatigueTier;
  weeklyVolume: number; // arbitrary units or kg
}

export interface RiskAssessment {
  score: number; // 0–100
  tier: FatigueTier;
  contributingFactors: string[];
}

export interface DashboardResponse {
  athlete: AthleteProfile;
  metrics: MetricSnapshot;
  weeklyPlan: PlanItem[];
  sessions: SessionSummary[];
  risk: RiskAssessment;
  recommendations: string[];
}
