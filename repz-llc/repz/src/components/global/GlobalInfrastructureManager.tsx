import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Globe, Server, Zap, Shield, TrendingUp, Activity, Users, MapPin, Wifi, Database, Cloud } from 'lucide-react';

interface Region {
  id: string;
  name: string;
  code: string;
  status: 'active' | 'deploying' | 'maintenance' | 'offline';
  users: number;
  latency: number;
  uptime: number;
  capacity: number;
}

interface DeploymentMetrics {
  totalRegions: number;
  activeUsers: number;
  globalLatency: number;
  totalUptime: number;
  dataTransfer: number;
  costOptimization: number;
}

export const GlobalInfrastructureManager: React.FC = () => {
  const [selectedRegion, setSelectedRegion] = useState<string>('us-east-1');
  const [isScaling, setIsScaling] = useState(false);

  const regions: Region[] = [
    { id: 'us-east-1', name: 'US East (N. Virginia)', code: 'USE1', status: 'active', users: 15420, latency: 12, uptime: 99.98, capacity: 87 },
    { id: 'us-west-2', name: 'US West (Oregon)', code: 'USW2', status: 'active', users: 8750, latency: 15, uptime: 99.97, capacity: 73 },
    { id: 'eu-west-1', name: 'Europe (Ireland)', code: 'EUW1', status: 'active', users: 12300, latency: 18, uptime: 99.95, capacity: 81 },
    { id: 'ap-southeast-1', name: 'Asia Pacific (Singapore)', code: 'APS1', status: 'deploying', users: 6890, latency: 22, uptime: 99.92, capacity: 65 },
    { id: 'ap-northeast-1', name: 'Asia Pacific (Tokyo)', code: 'APN1', status: 'active', users: 9840, latency: 19, uptime: 99.96, capacity: 78 },
    { id: 'sa-east-1', name: 'South America (São Paulo)', code: 'SAE1', status: 'maintenance', users: 3420, latency: 28, uptime: 99.89, capacity: 45 }
  ];

  const metrics: DeploymentMetrics = {
    totalRegions: 6,
    activeUsers: 56620,
    globalLatency: 18.7,
    totalUptime: 99.95,
    dataTransfer: 1247.8,
    costOptimization: 34.2
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'deploying': return 'bg-yellow-500';
      case 'maintenance': return 'bg-orange-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'active': return 'default';
      case 'deploying': return 'secondary';
      case 'maintenance': return 'outline';
      case 'offline': return 'destructive';
      default: return 'secondary';
    }
  };

  const handleAutoScale = () => {
    setIsScaling(true);
    setTimeout(() => setIsScaling(false), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Global Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Regions</CardTitle>
            <Globe className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalRegions}</div>
            <p className="text-xs text-muted-foreground">
              Across 4 continents
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Global Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.activeUsers.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              +12.3% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.globalLatency}ms</div>
            <p className="text-xs text-muted-foreground">
              -4.2ms improvement
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Global Uptime</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalUptime}%</div>
            <p className="text-xs text-muted-foreground">
              SLA: 99.9% maintained
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="regions" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="regions">
            <MapPin className="h-4 w-4 mr-2" />
            Regions
          </TabsTrigger>
          <TabsTrigger value="scaling">
            <TrendingUp className="h-4 w-4 mr-2" />
            Auto Scaling
          </TabsTrigger>
          <TabsTrigger value="network">
            <Wifi className="h-4 w-4 mr-2" />
            Network
          </TabsTrigger>
          <TabsTrigger value="data">
            <Database className="h-4 w-4 mr-2" />
            Data Sync
          </TabsTrigger>
        </TabsList>

        <TabsContent value="regions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Regional Deployment Status</CardTitle>
              <CardDescription>
                Monitor and manage global infrastructure across all regions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {regions.map((region) => (
                  <div key={region.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(region.status)}`} />
                      <div>
                        <div className="font-medium">{region.name}</div>
                        <div className="text-sm text-muted-foreground">{region.code}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-6">
                      <div className="text-center">
                        <div className="text-sm font-medium">{region.users.toLocaleString()}</div>
                        <div className="text-xs text-muted-foreground">Users</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm font-medium">{region.latency}ms</div>
                        <div className="text-xs text-muted-foreground">Latency</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm font-medium">{region.uptime}%</div>
                        <div className="text-xs text-muted-foreground">Uptime</div>
                      </div>
                      <div className="text-center min-w-[100px]">
                        <Progress value={region.capacity} className="h-2" />
                        <div className="text-xs text-muted-foreground mt-1">{region.capacity}% Capacity</div>
                      </div>
                      <Badge variant={getStatusVariant(region.status)}>
                        {region.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scaling" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Intelligent Auto Scaling</CardTitle>
              <CardDescription>
                AI-powered scaling based on traffic patterns and predictions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-2">Scaling Rules</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>CPU Threshold</span>
                      <span className="font-medium">75%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Memory Threshold</span>
                      <span className="font-medium">80%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Response Time</span>
                      <span className="font-medium">&lt;200ms</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Queue Length</span>
                      <span className="font-medium">&lt;100</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Current Status</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Active Instances</span>
                      <span className="font-medium">247</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Scaling Events (24h)</span>
                      <span className="font-medium">18</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Cost Optimization</span>
                      <span className="font-medium text-green-600">+{metrics.costOptimization}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Prediction Accuracy</span>
                      <span className="font-medium">94.7%</span>
                    </div>
                  </div>
                </div>
              </div>
              <Button 
                onClick={handleAutoScale} 
                disabled={isScaling}
                className="w-full"
              >
                {isScaling ? 'Scaling in Progress...' : 'Trigger Manual Scale'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Global Network Performance</CardTitle>
              <CardDescription>
                CDN and edge network optimization metrics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm">Cache Hit Rate</span>
                      <span className="text-sm font-medium">94.3%</span>
                    </div>
                    <Progress value={94.3} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm">Bandwidth Utilization</span>
                      <span className="text-sm font-medium">67.8%</span>
                    </div>
                    <Progress value={67.8} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm">Edge Coverage</span>
                      <span className="text-sm font-medium">98.1%</span>
                    </div>
                    <Progress value={98.1} className="h-2" />
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Total Data Transfer (24h)</span>
                    <span className="font-medium">{metrics.dataTransfer} TB</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Peak Throughput</span>
                    <span className="font-medium">2.3 Gbps</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Active Edge Locations</span>
                    <span className="font-medium">156</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Network Security Events</span>
                    <span className="font-medium text-green-600">0</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Global Data Synchronization</CardTitle>
              <CardDescription>
                Real-time data replication and consistency across regions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-4 border rounded-lg">
                    <Database className="h-8 w-8 mx-auto mb-2 text-primary" />
                    <div className="text-2xl font-bold">15.2ms</div>
                    <div className="text-sm text-muted-foreground">Avg Sync Time</div>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <Shield className="h-8 w-8 mx-auto mb-2 text-primary" />
                    <div className="text-2xl font-bold">99.99%</div>
                    <div className="text-sm text-muted-foreground">Data Integrity</div>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <Cloud className="h-8 w-8 mx-auto mb-2 text-primary" />
                    <div className="text-2xl font-bold">6/6</div>
                    <div className="text-sm text-muted-foreground">Regions Synced</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium">Replication Status</h4>
                  <div className="space-y-2">
                    {regions.filter(r => r.status === 'active').map((region) => (
                      <div key={region.id} className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm">{region.name}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full" />
                          <span className="text-sm text-muted-foreground">Synced</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};