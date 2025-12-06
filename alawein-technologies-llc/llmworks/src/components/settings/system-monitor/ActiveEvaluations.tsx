import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Activity } from "lucide-react";
import type { ActiveEvaluation } from "./types";

interface ActiveEvaluationsProps {
  activeEvaluations: ActiveEvaluation[];
}

export const ActiveEvaluationsCard = ({ activeEvaluations }: ActiveEvaluationsProps) => {
  return (
    <Card className="p-6">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="h-5 w-5 text-accent" />
        <h3 className="text-lg font-bold text-foreground">Active Evaluations</h3>
      </div>

      {activeEvaluations.length > 0 ? (
        <div className="space-y-4">
          {activeEvaluations.map((evaluation) => (
            <div key={evaluation.id} className="p-4 rounded-lg border border-border">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <div className="font-medium text-foreground">{evaluation.name}</div>
                  <div className="text-sm text-muted-foreground">
                    {evaluation.models.join(" vs ")} • {evaluation.type}
                  </div>
                </div>
                <Badge variant="outline">{evaluation.progress}%</Badge>
              </div>

              <div className="space-y-2">
                <Progress value={evaluation.progress} className="h-2" />
                <div className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>{evaluation.estimatedTime}</span>
                  <span>Running...</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-muted-foreground">
          <Activity className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>No active evaluations</p>
        </div>
      )}
    </Card>
  );
};
