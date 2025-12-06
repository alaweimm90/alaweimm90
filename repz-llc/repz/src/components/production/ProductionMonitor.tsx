import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Server,
  TrendingUp,
  Users,
  Zap
} from 'lucide-react';

interface SystemHealth {
  overall_status: 'healthy' | 'degraded' | 'critical';
  api_response_time: number;
  database_health: boolean;
  error_rate: number;
  active_users: number;
  last_updated: string;
}

export const ProductionMonitor = () => {
  const { toast } = useToast();
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchSystemHealth = async () => {
      try {
        const { data, error } = await supabase.functions.invoke('production-monitoring', {
          body: { action: 'get_system_health' }
        });

        if (error) throw error;
        setHealth(data);
      } catch (error) {
        console.error('Error fetching system health:', error);
        toast({
          title: "Monitoring Error",
          description: "Could not fetch system health",
          variant: "destructive"
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchSystemHealth();
    const interval = setInterval(fetchSystemHealth, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [toast]);

  if (isLoading) {
    return <div className="p-4">Loading system health...</div>;
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'degraded': return 'text-yellow-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return CheckCircle;
      case 'degraded': return AlertTriangle;
      case 'critical': return AlertTriangle;
      default: return Clock;
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Activity className="h-6 w-6" />
            Production System Health
          </CardTitle>
        </CardHeader>
        <CardContent>
          {health && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    {React.createElement(getStatusIcon(health.overall_status), {
                      className: `h-5 w-5 ${getStatusColor(health.overall_status)}`
                    })}
                    <span className="font-medium">Overall Status</span>
                  </div>
                  <Badge variant={health.overall_status === 'healthy' ? 'default' : 'destructive'}>
                    {health.overall_status.toUpperCase()}
                  </Badge>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Server className="h-5 w-5 text-blue-500" />
                    <span className="font-medium">API Response</span>
                  </div>
                  <div className="text-2xl font-bold">{health.api_response_time}ms</div>
                  <Progress value={Math.min(health.api_response_time / 50, 100)} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                    <span className="font-medium">Error Rate</span>
                  </div>
                  <div className="text-2xl font-bold">{health.error_rate}%</div>
                  <Progress value={health.error_rate} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Users className="h-5 w-5 text-green-500" />
                    <span className="font-medium">Active Users</span>
                  </div>
                  <div className="text-2xl font-bold">{health.active_users}</div>
                </CardContent>
              </Card>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
