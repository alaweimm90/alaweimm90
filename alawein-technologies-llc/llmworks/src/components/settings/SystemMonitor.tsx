import { useState, useEffect } from "react";
import { OverviewCards } from "./system-monitor/OverviewCards";
import { ServiceStatus } from "./system-monitor/ServiceStatus";
import { ActiveEvaluationsCard } from "./system-monitor/ActiveEvaluations";
import { PerformanceTrends } from "./system-monitor/PerformanceTrends";
import type { SystemStatus, ActiveEvaluation } from "./system-monitor/types";

export const SystemMonitor = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus[]>([
    { service: "Arena Service", status: "operational", responseTime: 45, uptime: 99.9 },
    { service: "Bench Service", status: "operational", responseTime: 62, uptime: 99.8 },
    { service: "Model Gateway", status: "degraded", responseTime: 120, uptime: 98.5 },
    { service: "Database", status: "operational", responseTime: 23, uptime: 99.9 },
    { service: "Authentication", status: "operational", responseTime: 31, uptime: 100 }
  ]);

  const [activeEvaluations, setActiveEvaluations] = useState<ActiveEvaluation[]>([
    {
      id: "eval-1",
      name: "MMLU Full Suite",
      type: "benchmark",
      progress: 67,
      estimatedTime: "12 min remaining",
      models: ["GPT-4o", "Claude 3.5 Sonnet"]
    },
    {
      id: "eval-2", 
      name: "Climate Change Debate",
      type: "arena",
      progress: 34,
      estimatedTime: "8 min remaining",
      models: ["Gemini 1.5 Pro", "Llama 3.1 70B"]
    }
  ]);

  const [metrics, setMetrics] = useState({
    totalEvaluations: 2847,
    activeUsers: 156,
    systemLoad: 73,
    avgResponseTime: 58
  });

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setActiveEvaluations(prev => 
        prev.map(evaluation => ({
          ...evaluation,
          progress: Math.min(100, evaluation.progress + Math.random() * 5)
        }))
      );

      setMetrics(prev => ({
        ...prev,
        systemLoad: Math.max(30, Math.min(95, prev.systemLoad + (Math.random() - 0.5) * 10)),
        avgResponseTime: Math.max(20, Math.min(150, prev.avgResponseTime + (Math.random() - 0.5) * 20))
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <OverviewCards metrics={metrics} />

      <ServiceStatus systemStatus={systemStatus} />

      <ActiveEvaluationsCard activeEvaluations={activeEvaluations} />

      <PerformanceTrends />
    </div>
  );
};