import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Database, 
  RefreshCw,
  Server
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface SystemMonitoringProps {
  className?: string;
}

export const SystemMonitoring: React.FC<SystemMonitoringProps> = ({ className }) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  const refreshData = async () => {
    setIsRefreshing(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsRefreshing(false);
  };

  return (
    <div className={cn("space-y-6", className)}>
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">System Monitoring</h2>
          <p className="text-muted-foreground">Real-time system health and performance metrics</p>
        </div>
        <Button onClick={refreshData} disabled={isRefreshing} variant="outline">
          <RefreshCw className={cn("mr-2 h-4 w-4", isRefreshing && "animate-spin")} />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overall Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-2xl font-bold">Healthy</span>
            </div>
            <p className="text-xs text-muted-foreground">Uptime: 99.9%</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Database</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-2xl font-bold">Online</span>
            </div>
            <p className="text-xs text-muted-foreground">12 active connections</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Services</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-2xl font-bold">Online</span>
            </div>
            <p className="text-xs text-muted-foreground">120ms avg response</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">0.1%</div>
            <Progress value={0.1} className="h-2 mt-2" />
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="metrics" className="w-full">
        <TabsList>
          <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics">
          <Card>
            <CardHeader>
              <CardTitle>System Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>CPU Usage</span>
                  <span>45%</span>
                </div>
                <Progress value={45} />
                
                <div className="flex items-center justify-between">
                  <span>Memory Usage</span>
                  <span>62%</span>
                </div>
                <Progress value={62} />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts">
          <Card>
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center space-x-4 p-3 rounded-lg border">
                  <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Storage usage approaching 80% threshold</p>
                    <p className="text-xs text-muted-foreground">5 minutes ago</p>
                  </div>
                  <Badge variant="outline" className="text-yellow-600">Warning</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};