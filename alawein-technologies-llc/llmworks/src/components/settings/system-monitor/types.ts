export interface SystemStatus {
  service: string;
  status: "operational" | "degraded" | "down";
  responseTime: number;
  uptime: number;
}

export interface ActiveEvaluation {
  id: string;
  name: string;
  type: string;
  progress: number;
  estimatedTime: string;
  models: string[];
}

export interface Metrics {
  totalEvaluations: number;
  activeUsers: number;
  systemLoad: number;
  avgResponseTime: number;
}
