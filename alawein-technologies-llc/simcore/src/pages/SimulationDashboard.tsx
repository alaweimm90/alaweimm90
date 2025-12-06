import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { useNavigate } from 'react-router-dom';
import { ScientificPlot } from '@/components/ScientificPlot';
import { MLAnalysis } from '@/components/MLAnalysis';
import { 
  Activity,
  BarChart3,
  Brain,
  Cpu,
  Database,
  Gauge,
  LineChart,
  Monitor,
  Zap,
  TrendingUp,
  Users,
  Globe,
  Clock,
  CheckCircle,
  AlertTriangle,
  XCircle
} from 'lucide-react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useSEO } from '@/hooks/use-seo';

interface SimulationMetrics {
  activeSimulations: number;
  totalComputations: number;
  cpuUsage: number;
  memoryUsage: number;
  successRate: number;
  averageExecutionTime: number;
}

interface SystemStatus {
  webWorkers: 'online' | 'offline' | 'degraded';
  webGPU: 'online' | 'offline' | 'degraded';
  mlEngine: 'online' | 'offline' | 'degraded';
  database: 'online' | 'offline' | 'degraded';
}

const SimulationDashboard = () => {
  const navigate = useNavigate();
  useSEO({ title: 'Simulation Dashboard – SimCore', description: 'Real-time monitoring and analytics for physics simulations.' });
  
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name: 'Simulation Dashboard',
    description: 'Real-time monitoring and analytics for physics simulations.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'simulation dashboard, analytics, monitoring',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  
  // Real-time metrics
  const [metrics, setMetrics] = useState<SimulationMetrics>({
    activeSimulations: 3,
    totalComputations: 1247,
    cpuUsage: 45,
    memoryUsage: 62,
    successRate: 98.5,
    averageExecutionTime: 2.3
  });
  
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    webWorkers: 'online',
    webGPU: 'online',
    mlEngine: 'online',
    database: 'online'
  });
  
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  
  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        activeSimulations: Math.max(0, prev.activeSimulations + (Math.random() - 0.5) * 2),
        totalComputations: prev.totalComputations + Math.floor(Math.random() * 5),
        cpuUsage: Math.max(10, Math.min(90, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(20, Math.min(95, prev.memoryUsage + (Math.random() - 0.5) * 8)),
        successRate: Math.max(95, Math.min(100, prev.successRate + (Math.random() - 0.5) * 2)),
        averageExecutionTime: Math.max(0.5, Math.min(10, prev.averageExecutionTime + (Math.random() - 0.5) * 0.5))
      }));
    }, 3000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Performance data for charts
  const performanceData = useMemo(() => {
    const timePoints = Array.from({ length: 24 }, (_, i) => i);
    const cpuData = timePoints.map(t => ({
      time: t,
      cpu: 30 + 20 * Math.sin(t * 0.3) + 10 * Math.random(),
      memory: 40 + 15 * Math.cos(t * 0.4) + 8 * Math.random(),
      throughput: 50 + 25 * Math.sin(t * 0.2) + 12 * Math.random()
    }));
    
    return {
      datasets: [
        {
          x: cpuData.map(d => d.time),
          y: cpuData.map(d => d.cpu),
          label: 'CPU Usage (%)',
          color: '#3b82f6',
          lineWidth: 2
        },
        {
          x: cpuData.map(d => d.time),
          y: cpuData.map(d => d.memory),
          label: 'Memory Usage (%)',
          color: '#10b981',
          lineWidth: 2
        },
        {
          x: cpuData.map(d => d.time),
          y: cpuData.map(d => d.throughput),
          label: 'Throughput',
          color: '#f59e0b',
          lineWidth: 2
        }
      ],
      xLabel: 'Time (hours)',
      yLabel: 'Usage (%)',
      title: 'System Performance (24h)'
    };
  }, []);
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-400';
      case 'degraded': return 'text-yellow-400';
      case 'offline': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online': return CheckCircle;
      case 'degraded': return AlertTriangle;
      case 'offline': return XCircle;
      default: return Clock;
    }
  };
  
  const recentSimulations = [
    { id: 1, type: 'Quantum Tunneling', status: 'completed', duration: '2.3s', accuracy: '99.2%' },
    { id: 2, type: 'Band Structure', status: 'running', duration: '1.7s', accuracy: '96.8%' },
    { id: 3, type: 'ML Training', status: 'completed', duration: '15.2s', accuracy: '98.5%' },
    { id: 4, type: 'LLG Dynamics', status: 'queued', duration: '-', accuracy: '-' },
    { id: 5, type: 'Ising Model', status: 'completed', duration: '0.9s', accuracy: '97.1%' }
  ];
  
  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Simulation Dashboard"
        description="Real-time monitoring and analytics for physics simulations"
        category="System Management"
        difficulty="Advanced"
        equation="\\sum_{i=1}^{n} S_i = \\int_{0}^{t} \\mathcal{P}(\\tau) d\\tau"
      />
      
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" />
                <div className="text-xs text-muted-foreground">Active</div>
              </div>
              <div className="text-2xl font-bold text-primary mt-1">{Math.round(metrics.activeSimulations)}</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-green-400" />
                <div className="text-xs text-muted-foreground">Total</div>
              </div>
              <div className="text-2xl font-bold text-primary mt-1">{metrics.totalComputations}</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-purple-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-purple-400" />
                <div className="text-xs text-muted-foreground">CPU</div>
              </div>
              <div className="text-2xl font-bold text-primary mt-1">{Math.round(metrics.cpuUsage)}%</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border-orange-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-orange-400" />
                <div className="text-xs text-muted-foreground">Memory</div>
              </div>
              <div className="text-2xl font-bold text-primary mt-1">{Math.round(metrics.memoryUsage)}%</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-teal-500/10 to-cyan-500/10 border-teal-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-teal-400" />
                <div className="text-xs text-muted-foreground">Success</div>
              </div>
              <div className="text-2xl font-bold text-primary mt-1">{metrics.successRate.toFixed(1)}%</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border-indigo-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                <div className="text-xs text-muted-foreground">Avg Time</div>
              </div>
              <div className="text-2xl font-bold text-primary mt-1">{metrics.averageExecutionTime.toFixed(1)}s</div>
            </CardContent>
          </Card>
        </div>
        
        {/* System Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Monitor className="w-5 h-5" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(systemStatus).map(([service, status]) => {
                const StatusIcon = getStatusIcon(status);
                return (
                  <div key={service} className="flex items-center gap-3 p-3 bg-card/30 rounded-lg">
                    <StatusIcon className={`w-5 h-5 ${getStatusColor(status)}`} />
                    <div>
                      <div className="font-medium capitalize">{service.replace(/([A-Z])/g, ' $1').trim()}</div>
                      <div className={`text-sm capitalize ${getStatusColor(status)}`}>{status}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LineChart className="w-5 h-5" />
                Performance Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ScientificPlot
                  title={performanceData.title}
                  data={performanceData}
                  plotType="2d"
                  xLabel={performanceData.xLabel}
                  yLabel={performanceData.yLabel}
                  showGrid={true}
                  showLegend={true}
                />
              </div>
            </CardContent>
          </Card>
          
          {/* Recent Simulations */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Gauge className="w-5 h-5" />
                Recent Simulations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentSimulations.map((sim) => (
                  <div key={sim.id} className="flex items-center justify-between p-3 bg-card/30 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        sim.status === 'completed' ? 'bg-green-400' :
                        sim.status === 'running' ? 'bg-blue-400 animate-pulse' :
                        'bg-yellow-400'
                      }`} />
                      <div>
                        <div className="font-medium text-sm">{sim.type}</div>
                        <div className="text-xs text-muted-foreground">
                          {sim.duration} • {sim.accuracy}
                        </div>
                      </div>
                    </div>
                    <Badge 
                      variant={
                        sim.status === 'completed' ? 'default' :
                        sim.status === 'running' ? 'secondary' : 'outline'
                      }
                      className="text-xs"
                    >
                      {sim.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
        
        {/* Detailed Analytics */}
        <Tabs value={selectedTimeRange} onValueChange={setSelectedTimeRange} className="w-full">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Advanced Analytics
                </CardTitle>
                <TabsList>
                  <TabsTrigger value="1h">1H</TabsTrigger>
                  <TabsTrigger value="24h">24H</TabsTrigger>
                  <TabsTrigger value="7d">7D</TabsTrigger>
                  <TabsTrigger value="30d">30D</TabsTrigger>
                </TabsList>
              </div>
            </CardHeader>
            <CardContent>
              <TabsContent value="1h" className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Last hour performance analysis and optimization recommendations.
                </div>
              </TabsContent>
              
              <TabsContent value="24h" className="space-y-4">
                <MLAnalysis
                  data={[
                    [metrics.cpuUsage, metrics.memoryUsage, metrics.successRate, metrics.averageExecutionTime],
                    [metrics.cpuUsage * 0.9, metrics.memoryUsage * 1.1, metrics.successRate * 0.98, metrics.averageExecutionTime * 1.2],
                    [metrics.cpuUsage * 1.1, metrics.memoryUsage * 0.95, metrics.successRate * 1.02, metrics.averageExecutionTime * 0.8]
                  ]}
                  parameters={{
                    cpuUsage: metrics.cpuUsage,
                    memoryUsage: metrics.memoryUsage,
                    successRate: metrics.successRate,
                    avgTime: metrics.averageExecutionTime
                  }}
                  simulationType="system_performance"
                />
              </TabsContent>
              
              <TabsContent value="7d" className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Weekly trends and pattern analysis for system optimization.
                </div>
              </TabsContent>
              
              <TabsContent value="30d" className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Monthly performance reports and capacity planning insights.
                </div>
              </TabsContent>
            </CardContent>
          </Card>
        </Tabs>
        
        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Quick Actions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Button 
                variant="outline" 
                className="h-auto p-4 flex flex-col items-center gap-2"
                onClick={() => navigate('/modules/ml-showcase')}
              >
                <Brain className="w-6 h-6" />
                <span className="text-sm">ML Showcase</span>
              </Button>
              
              <Button 
                variant="outline" 
                className="h-auto p-4 flex flex-col items-center gap-2"
                onClick={() => navigate('/performance-monitor')}
              >
                <Monitor className="w-6 h-6" />
                <span className="text-sm">Performance</span>
              </Button>
              
              <Button 
                variant="outline" 
                className="h-auto p-4 flex flex-col items-center gap-2"
                onClick={() => navigate('/advanced-simulation')}
              >
                <Gauge className="w-6 h-6" />
                <span className="text-sm">Advanced Sim</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </PhysicsModuleLayout>
  );
};

export default SimulationDashboard;