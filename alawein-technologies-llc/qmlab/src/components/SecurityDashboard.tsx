import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Activity,
  Eye,
  EyeOff,
  RefreshCw,
  Settings,
  Lock,
  Unlock,
  Zap,
  TrendingUp,
  TrendingDown,
  Clock,
  Users,
  Server,
  Bug
} from 'lucide-react';

// Mock hook for security data (would be implemented with actual threat detection)
const useSecurityData = () => {
  const [securityStatus, setSecurityStatus] = useState({
    isMonitoring: false,
    metrics: {
      threats_detected: 0,
      threats_blocked: 0,
      false_positives: 0,
      average_response_time: 0,
      security_score: 100,
      last_scan: Date.now()
    },
    recentThreats: 0,
    activeSignatures: 8,
    totalThreats: 0,
    lastScan: Date.now(),
    threatsByCategory: {},
    threatsBySeverity: {}
  });

  const [threats, setThreats] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Mock data generator for demonstration
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate security status updates
      setSecurityStatus(prev => ({
        ...prev,
        metrics: {
          ...prev.metrics,
          security_score: Math.max(85, Math.min(100, prev.metrics.security_score + (Math.random() - 0.5) * 2)),
          last_scan: Date.now()
        },
        recentThreats: Math.floor(Math.random() * 3),
        lastScan: Date.now()
      }));
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const toggleMonitoring = () => {
    setSecurityStatus(prev => ({
      ...prev,
      isMonitoring: !prev.isMonitoring
    }));
  };

  const refreshData = async () => {
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      setSecurityStatus(prev => ({
        ...prev,
        lastScan: Date.now()
      }));
    }, 1000);
  };

  return {
    securityStatus,
    threats,
    isLoading,
    actions: {
      toggleMonitoring,
      refreshData
    }
  };
};

// Security Score Indicator
const SecurityScoreIndicator: React.FC<{ score: number }> = ({ score }) => {
  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-500';
    if (score >= 70) return 'text-yellow-500';
    if (score >= 50) return 'text-orange-500';
    return 'text-red-500';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 90) return <ShieldCheck className="h-5 w-5" />;
    if (score >= 70) return <Shield className="h-5 w-5" />;
    return <ShieldAlert className="h-5 w-5" />;
  };

  const getScoreStatus = (score: number) => {
    if (score >= 90) return 'Excellent';
    if (score >= 70) return 'Good';
    if (score >= 50) return 'Fair';
    return 'Poor';
  };

  return (
    <div className={`flex items-center gap-3 ${getScoreColor(score)}`}>
      {getScoreIcon(score)}
      <div>
        <div className="text-3xl font-bold">{score}/100</div>
        <div className="text-sm opacity-80">{getScoreStatus(score)}</div>
      </div>
    </div>
  );
};

// Threat Item Component
const ThreatItem: React.FC<{ threat: any }> = ({ threat }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'secondary';
      case 'medium': return 'outline';
      default: return 'outline';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'high': return <AlertTriangle className="h-4 w-4 text-orange-500" />;
      case 'medium': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default: return <Activity className="h-4 w-4 text-blue-500" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'injection': return <Bug className="h-4 w-4" />;
      case 'xss': return <Zap className="h-4 w-4" />;
      case 'dos': return <Server className="h-4 w-4" />;
      case 'data-exfiltration': return <Lock className="h-4 w-4" />;
      case 'reconnaissance': return <Eye className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  return (
    <div className="flex items-center justify-between p-4 border rounded-lg bg-white dark:bg-gray-900">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          {getSeverityIcon(threat.severity)}
          {getCategoryIcon(threat.category)}
        </div>
        <div>
          <h4 className="font-semibold">{threat.name}</h4>
          <p className="text-sm text-gray-600">{threat.description}</p>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant={getSeverityColor(threat.severity) as any}>
              {threat.severity}
            </Badge>
            <span className="text-xs text-gray-500">
              {new Date(threat.timestamp).toLocaleString()}
            </span>
          </div>
        </div>
      </div>
      <div className="text-right">
        <div className="text-lg font-bold text-red-500">
          {threat.riskScore}/100
        </div>
        <div className="text-xs text-gray-500">Risk Score</div>
      </div>
    </div>
  );
};

// Security Metric Chart
const SecurityMetricChart: React.FC<{
  title: string;
  value: number;
  trend: 'up' | 'down' | 'stable';
  color: string;
  icon: React.ReactNode;
}> = ({ title, value, trend, color, icon }) => {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-red-500" />;
      case 'down': return <TrendingDown className="h-4 w-4 text-green-500" />;
      default: return <Activity className="h-4 w-4 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium">{title}</span>
        </div>
        {getTrendIcon()}
      </div>
      <div className={`text-2xl font-bold ${color}`}>
        {value}
      </div>
    </div>
  );
};

// Main Security Dashboard Component
export const SecurityDashboard: React.FC = () => {
  const { securityStatus, threats, isLoading, actions } = useSecurityData();
  const [selectedTab, setSelectedTab] = useState('overview');

  // Format time
  const formatTime = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  // Mock threat data for demonstration
  const mockThreats = [
    {
      id: '1',
      name: 'SQL Injection Attempt',
      description: 'Detected SQL injection pattern in query parameter',
      severity: 'high',
      category: 'injection',
      timestamp: Date.now() - 300000,
      riskScore: 85,
      source: 'web-request',
      blocked: true
    },
    {
      id: '2',
      name: 'Excessive API Requests',
      description: 'Unusual spike in API request rate detected',
      severity: 'medium',
      category: 'dos',
      timestamp: Date.now() - 600000,
      riskScore: 65,
      source: 'api-endpoint',
      blocked: false
    },
    {
      id: '3',
      name: 'Cross-Site Scripting',
      description: 'XSS attempt detected in form submission',
      severity: 'high',
      category: 'xss',
      timestamp: Date.now() - 900000,
      riskScore: 78,
      source: 'form-input',
      blocked: true
    }
  ];

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Security Dashboard</h1>
          <p className="text-gray-600">
            Real-time threat monitoring and security analysis
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            size="sm"
            variant="outline"
            onClick={actions.toggleMonitoring}
          >
            {securityStatus.isMonitoring ? (
              <>
                <Eye className="h-4 w-4 mr-2" />
                Monitoring Active
              </>
            ) : (
              <>
                <EyeOff className="h-4 w-4 mr-2" />
                Monitoring Inactive
              </>
            )}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={actions.refreshData}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Security Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Shield className="h-4 w-4" />
              Security Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <SecurityScoreIndicator score={securityStatus.metrics.security_score} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <AlertTriangle className="h-4 w-4" />
              Recent Threats
            </CardTitle>
          </CardHeader>
          <CardContent>
            <SecurityMetricChart
              title=""
              value={securityStatus.recentThreats}
              trend="down"
              color="text-orange-500"
              icon={<></>}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <ShieldCheck className="h-4 w-4" />
              Threats Blocked
            </CardTitle>
          </CardHeader>
          <CardContent>
            <SecurityMetricChart
              title=""
              value={securityStatus.metrics.threats_blocked}
              trend="up"
              color="text-green-500"
              icon={<></>}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4" />
              Active Signatures
            </CardTitle>
          </CardHeader>
          <CardContent>
            <SecurityMetricChart
              title=""
              value={securityStatus.activeSignatures}
              trend="stable"
              color="text-blue-500"
              icon={<></>}
            />
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="threats">Threats</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* System Status */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  System Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>Monitoring Status:</span>
                  <div className="flex items-center gap-2">
                    {securityStatus.isMonitoring ? (
                      <>
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span className="text-green-600">Active</span>
                      </>
                    ) : (
                      <>
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        <span className="text-red-600">Inactive</span>
                      </>
                    )}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span>Last Scan:</span>
                  <span className="text-sm text-gray-500">
                    {formatTime(securityStatus.lastScan)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Response Time:</span>
                  <span className="text-sm font-mono">
                    {securityStatus.metrics.average_response_time}ms
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Security Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>Total Threats Detected:</span>
                  <span className="font-mono font-bold">
                    {securityStatus.metrics.threats_detected}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Threats Blocked:</span>
                  <span className="font-mono font-bold text-green-600">
                    {securityStatus.metrics.threats_blocked}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>False Positives:</span>
                  <span className="font-mono font-bold text-yellow-600">
                    {securityStatus.metrics.false_positives}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Threats */}
          {mockThreats.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Recent Security Events
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockThreats.slice(0, 3).map(threat => (
                    <ThreatItem key={threat.id} threat={threat} />
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="threats" className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Threat Detection Log</CardTitle>
              <div className="flex gap-2">
                <Button size="sm" variant="outline">
                  Export Log
                </Button>
                <Button size="sm" variant="outline">
                  Filter
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockThreats.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <ShieldCheck className="h-12 w-12 mx-auto mb-4 text-green-500" />
                    <p>No threats detected</p>
                    <p className="text-sm">Your system is secure</p>
                  </div>
                ) : (
                  mockThreats.map(threat => (
                    <ThreatItem key={threat.id} threat={threat} />
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Detection Rules</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {[
                  { name: 'SQL Injection Detection', enabled: true, severity: 'high' },
                  { name: 'XSS Prevention', enabled: true, severity: 'high' },
                  { name: 'Rate Limiting', enabled: true, severity: 'medium' },
                  { name: 'Quantum Parameter Validation', enabled: true, severity: 'medium' },
                  { name: 'Reconnaissance Detection', enabled: false, severity: 'low' }
                ].map((rule, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        rule.enabled ? 'bg-green-500' : 'bg-gray-300'
                      }`}></div>
                      <span className="font-medium">{rule.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {rule.severity}
                      </Badge>
                    </div>
                    <Button size="sm" variant="ghost">
                      {rule.enabled ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                    </Button>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Alert Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Alert Threshold</label>
                  <select className="w-full p-2 border rounded">
                    <option>Critical and High only</option>
                    <option>Medium and above</option>
                    <option>All severities</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Notification Method</label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked />
                      <span className="text-sm">Browser notifications</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" />
                      <span className="text-sm">Email alerts</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" />
                      <span className="text-sm">Console logging</span>
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Security Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Security Mode</label>
                  <select className="w-full p-2 border rounded">
                    <option>Standard Protection</option>
                    <option>High Security</option>
                    <option>Maximum Security</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Scan Frequency</label>
                  <select className="w-full p-2 border rounded">
                    <option>Every 30 seconds</option>
                    <option>Every minute</option>
                    <option>Every 5 minutes</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input type="checkbox" defaultChecked />
                    <span className="text-sm">Enable automatic threat blocking</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" defaultChecked />
                    <span className="text-sm">Log all security events</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" />
                    <span className="text-sm">Enable detailed forensics</span>
                  </label>
                </div>
              </div>

              <div className="pt-4 border-t">
                <Button className="mr-2">Save Configuration</Button>
                <Button variant="outline">Reset to Defaults</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SecurityDashboard;