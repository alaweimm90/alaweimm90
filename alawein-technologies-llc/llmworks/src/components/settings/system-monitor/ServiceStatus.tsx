import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Server, AlertTriangle, CheckCircle, Clock } from "lucide-react";
import type { SystemStatus } from "./types";

interface ServiceStatusProps {
  systemStatus: SystemStatus[];
}

export const ServiceStatus = ({ systemStatus }: ServiceStatusProps) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "operational":
        return "bg-accent/10 text-accent";
      case "degraded":
        return "bg-yellow-500/10 text-yellow-600";
      case "down":
        return "bg-destructive/10 text-destructive";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "operational":
        return CheckCircle;
      case "degraded":
        return AlertTriangle;
      case "down":
        return AlertTriangle;
      default:
        return Clock;
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center gap-2 mb-4">
        <Server className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-bold text-foreground">Service Status</h3>
      </div>

      <div className="space-y-3">
        {systemStatus.map((service, index) => {
          const StatusIcon = getStatusIcon(service.status);

          return (
            <div
              key={index}
              className="flex items-center justify-between p-4 rounded-lg border border-border"
            >
              <div className="flex items-center gap-3">
                <StatusIcon className="h-5 w-5" />
                <div>
                  <div className="font-medium text-foreground">{service.service}</div>
                  <div className="text-sm text-muted-foreground">
                    {service.responseTime}ms • {service.uptime}% uptime
                  </div>
                </div>
              </div>
              <Badge className={getStatusColor(service.status)}>{service.status}</Badge>
            </div>
          );
        })}
      </div>
    </Card>
  );
};
