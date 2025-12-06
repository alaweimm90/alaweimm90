import React from 'react';
import { SystemMonitoring } from '@/components/production/SystemMonitoring';
import { SecurityMonitor } from '@/components/security/SecurityMonitor';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Shield, Server, Activity } from 'lucide-react';

const SystemHealth: React.FC = () => {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Shield className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">System Health</h1>
              <p className="text-muted-foreground">
                Monitor system performance and health metrics
              </p>
            </div>
          </div>
        </div>

        <Tabs defaultValue="monitoring" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="monitoring">System Monitoring</TabsTrigger>
            <TabsTrigger value="security">Security Monitor</TabsTrigger>
          </TabsList>
          
          <TabsContent value="monitoring" className="space-y-6">
            <SystemMonitoring />
          </TabsContent>
          
          <TabsContent value="security" className="space-y-6">
            <SecurityMonitor />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default SystemHealth;