import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Activity, Users, Zap, BarChart3 } from "lucide-react";
import type { Metrics } from "./types";

interface OverviewCardsProps {
  metrics: Metrics;
}

export const OverviewCards = ({ metrics }: OverviewCardsProps) => {
  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">System Load</p>
            <p className="text-2xl font-bold text-foreground">{metrics.systemLoad}%</p>
          </div>
          <Activity className="h-8 w-8 text-primary" />
        </div>
        <Progress value={metrics.systemLoad} className="mt-2" />
      </Card>

      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">Active Users</p>
            <p className="text-2xl font-bold text-foreground">{metrics.activeUsers}</p>
          </div>
          <Users className="h-8 w-8 text-accent" />
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">Avg Response</p>
            <p className="text-2xl font-bold text-foreground">{Math.round(metrics.avgResponseTime)}ms</p>
          </div>
          <Zap className="h-8 w-8 text-primary" />
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">Total Evals</p>
            <p className="text-2xl font-bold text-foreground">{metrics.totalEvaluations.toLocaleString()}</p>
          </div>
          <BarChart3 className="h-8 w-8 text-accent" />
        </div>
      </Card>
    </div>
  );
};
