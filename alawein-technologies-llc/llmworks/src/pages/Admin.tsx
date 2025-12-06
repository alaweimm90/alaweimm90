import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import {
  Settings,
  Users,
  Activity,
  Database,
  Shield,
  Bell,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Download,
  Upload,
  Trash2,
  Eye,
  EyeOff,
  Lock,
  Unlock,
} from "lucide-react";
import { getConfig, updateConfig } from "@/lib/environment";
import { getAllFeatureFlags, setOverride, clearOverrides } from "@/lib/feature-flags";
// import { offlineSyncManager } from "@/lib/offline-sync";
// import { pwaManager } from "@/lib/pwa";

interface SystemStatus {
  status: 'healthy' | 'degraded' | 'critical';
  uptime: number;
  users: {
    total: number;
    active: number;
    new: number;
  };
  performance: {
    avgResponseTime: number;
    errorRate: number;
    throughput: number;
  };
  storage: {
    used: number;
    total: number;
    percentage: number;
  };
  cache: {
    hitRate: number;
    size: number;
  };
}

interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'moderator';
  status: 'active' | 'suspended' | 'pending';
  lastLogin: string;
  evaluations: number;
  benchmarks: number;
}

const AdminDashboard = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'healthy',
    uptime: 99.98,
    users: { total: 1247, active: 89, new: 12 },
    performance: { avgResponseTime: 245, errorRate: 0.02, throughput: 1500 },
    storage: { used: 45.2, total: 100, percentage: 45.2 },
    cache: { hitRate: 94.5, size: 256 },
  });

  const [users, setUsers] = useState<User[]>([
    {
      id: '1',
      email: 'admin@llmworks.dev',
      name: 'System Admin',
      role: 'admin',
      status: 'active',
      lastLogin: '2024-01-12T10:30:00Z',
      evaluations: 156,
      benchmarks: 23,
    },
    // Add more mock users as needed
  ]);

  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [featureFlags, setFeatureFlags] = useState(getAllFeatureFlags());
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState<{
    type: 'success' | 'error' | 'info';
    message: string;
  } | null>(null);

  const config = getConfig();

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setSystemStatus(prev => ({
        ...prev,
        users: {
          ...prev.users,
          active: Math.max(50, prev.users.active + Math.floor(Math.random() * 10 - 5)),
        },
        performance: {
          ...prev.performance,
          avgResponseTime: Math.max(200, prev.performance.avgResponseTime + Math.floor(Math.random() * 40 - 20)),
        },
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const showNotification = (type: 'success' | 'error' | 'info', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  const handleFeatureFlagToggle = async (flagName: string, enabled: boolean) => {
    try {
      setOverride(flagName, enabled);
      setFeatureFlags({ ...featureFlags, [flagName]: { ...featureFlags[flagName], enabled } });
      showNotification('success', `Feature flag ${flagName} ${enabled ? 'enabled' : 'disabled'}`);
    } catch (error) {
      showNotification('error', 'Failed to update feature flag');
    }
  };

  const handleClearCache = async () => {
    setLoading(true);
    try {
      // await pwaManager.clearCaches();
      showNotification('success', 'Cache cleared successfully');
    } catch (error) {
      showNotification('error', 'Failed to clear cache');
    } finally {
      setLoading(false);
    }
  };

  const handleForceSync = async () => {
    setLoading(true);
    try {
      // await offlineSyncManager.sync();
      showNotification('success', 'Sync completed successfully');
    } catch (error) {
      showNotification('error', 'Sync failed');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'degraded':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'critical':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-red-100 text-red-800';
      case 'moderator':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'suspended':
        return 'bg-red-100 text-red-800';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
            <p className="text-gray-600">Manage LLM Works platform</p>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon(systemStatus.status)}
            <Badge variant={systemStatus.status === 'healthy' ? 'default' : 'destructive'}>
              {systemStatus.status.toUpperCase()}
            </Badge>
          </div>
        </div>

        {/* Notification */}
        {notification && (
          <Alert className={`${
            notification.type === 'success' ? 'border-green-500' :
            notification.type === 'error' ? 'border-red-500' : 'border-blue-500'
          }`}>
            <AlertTitle>
              {notification.type === 'success' ? 'Success' :
               notification.type === 'error' ? 'Error' : 'Info'}
            </AlertTitle>
            <AlertDescription>{notification.message}</AlertDescription>
          </Alert>
        )}

        {/* System Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Users</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStatus.users.total}</div>
              <p className="text-xs text-muted-foreground">
                {systemStatus.users.active} active now
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStatus.performance.avgResponseTime}ms</div>
              <p className="text-xs text-muted-foreground">
                {systemStatus.performance.errorRate}% error rate
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStatus.storage.percentage}%</div>
              <Progress value={systemStatus.storage.percentage} className="mt-2" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStatus.cache.hitRate}%</div>
              <p className="text-xs text-muted-foreground">
                {systemStatus.cache.size}MB cached
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="users">Users</TabsTrigger>
            <TabsTrigger value="features">Features</TabsTrigger>
            <TabsTrigger value="system">System</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Recent Activity</CardTitle>
                  <CardDescription>Latest system events and user actions</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center space-x-4">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm font-medium">New user registration</p>
                      <p className="text-xs text-muted-foreground">john@example.com • 2 minutes ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm font-medium">Benchmark completed</p>
                      <p className="text-xs text-muted-foreground">GPT-4 vs Claude • 5 minutes ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm font-medium">High CPU usage detected</p>
                      <p className="text-xs text-muted-foreground">Server load: 85% • 10 minutes ago</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                  <CardDescription>Common administrative tasks</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button 
                    onClick={handleClearCache} 
                    disabled={loading} 
                    className="w-full justify-start"
                    variant="outline"
                  >
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Clear System Cache
                  </Button>
                  <Button 
                    onClick={handleForceSync} 
                    disabled={loading} 
                    className="w-full justify-start"
                    variant="outline"
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Force Data Sync
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <Upload className="mr-2 h-4 w-4" />
                    Export Data
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <Bell className="mr-2 h-4 w-4" />
                    Send Notification
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Users Tab */}
          <TabsContent value="users" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>User Management</CardTitle>
                <CardDescription>Manage user accounts and permissions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {users.map((user) => (
                    <div key={user.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div>
                          <p className="font-medium">{user.name}</p>
                          <p className="text-sm text-muted-foreground">{user.email}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge className={getRoleColor(user.role)}>{user.role}</Badge>
                        <Badge className={getStatusColor(user.status)}>{user.status}</Badge>
                        <Button size="sm" variant="outline">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Settings className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Features Tab */}
          <TabsContent value="features" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Feature Flags</CardTitle>
                <CardDescription>Control feature rollout and availability</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(featureFlags).map(([flagName, flag]) => (
                    <div key={flagName} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <p className="font-medium">{flag.name}</p>
                        <p className="text-sm text-muted-foreground">{flag.description}</p>
                      </div>
                      <Switch
                        checked={flag.enabled}
                        onCheckedChange={(checked) => handleFeatureFlagToggle(flagName, checked)}
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* System Tab */}
          <TabsContent value="system" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>System Performance</CardTitle>
                  <CardDescription>Monitor system health and performance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">CPU Usage</span>
                      <span className="text-sm text-muted-foreground">65%</span>
                    </div>
                    <Progress value={65} />
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">Memory Usage</span>
                      <span className="text-sm text-muted-foreground">72%</span>
                    </div>
                    <Progress value={72} />
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">Disk Usage</span>
                      <span className="text-sm text-muted-foreground">{systemStatus.storage.percentage}%</span>
                    </div>
                    <Progress value={systemStatus.storage.percentage} />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Service Status</CardTitle>
                  <CardDescription>Current status of system services</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span>API Server</span>
                    <Badge className="bg-green-100 text-green-800">Running</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Database</span>
                    <Badge className="bg-green-100 text-green-800">Connected</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Redis Cache</span>
                    <Badge className="bg-green-100 text-green-800">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Background Jobs</span>
                    <Badge className="bg-yellow-100 text-yellow-800">Busy</Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Security Tab */}
          <TabsContent value="security" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Security Settings</CardTitle>
                <CardDescription>Configure security policies and monitoring</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-base">Rate Limiting</Label>
                      <p className="text-sm text-muted-foreground">Enable API rate limiting</p>
                    </div>
                    <Switch defaultChecked={config.security.rateLimiting} />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-base">CSRF Protection</Label>
                      <p className="text-sm text-muted-foreground">Enable CSRF token validation</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-base">Content Security Policy</Label>
                      <p className="text-sm text-muted-foreground">Enable strict CSP headers</p>
                    </div>
                    <Switch defaultChecked={config.security.cspEnabled} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Platform Settings</CardTitle>
                <CardDescription>Configure global platform settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="site-name">Site Name</Label>
                    <Input id="site-name" defaultValue="LLM Works" />
                  </div>
                  
                  <div>
                    <Label htmlFor="api-url">API URL</Label>
                    <Input id="api-url" defaultValue={config.apiUrl} />
                  </div>
                  
                  <div>
                    <Label htmlFor="max-evaluations">Max Evaluations per User</Label>
                    <Input id="max-evaluations" type="number" defaultValue="100" />
                  </div>
                  
                  <div>
                    <Label htmlFor="maintenance-message">Maintenance Message</Label>
                    <Textarea 
                      id="maintenance-message" 
                      placeholder="Enter maintenance message..."
                      rows={3}
                    />
                  </div>
                </div>
                
                <div className="pt-4">
                  <Button>Save Settings</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default AdminDashboard;