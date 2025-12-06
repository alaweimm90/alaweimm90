import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Input } from '@/ui/atoms/Input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { 
  AlertTriangle, 
  XCircle, 
  Clock,
  TrendingUp,
  TrendingDown,
  Search,
  Filter,
  Bug,
  Code,
  Globe,
  Smartphone,
  RefreshCw
} from 'lucide-react';

interface ErrorLog {
  id: string;
  timestamp: string;
  level: 'error' | 'warning' | 'info';
  message: string;
  source: 'client' | 'server' | 'api' | 'database';
  category: 'javascript' | 'network' | 'authentication' | 'validation' | 'system';
  userAgent?: string;
  userId?: string;
  url?: string;
  stack?: string;
  count: number;
  firstSeen: string;
  lastSeen: string;
  resolved: boolean;
}

interface ErrorTrend {
  date: string;
  errors: number;
  warnings: number;
  total: number;
}

export const ErrorTracker = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLevel, setSelectedLevel] = useState<string>('all');
  const [selectedSource, setSelectedSource] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  const errorLogs: ErrorLog[] = useMemo(() => [
    {
      id: 'err-1',
      timestamp: '2024-01-15T14:30:00Z',
      level: 'error',
      message: 'TypeError: Cannot read property \'id\' of undefined',
      source: 'client',
      category: 'javascript',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      userId: 'user-123',
      url: '/dashboard',
      stack: 'at UserProfile.render (/src/components/UserProfile.tsx:45:12)',
      count: 127,
      firstSeen: '2024-01-14T08:15:00Z',
      lastSeen: '2024-01-15T14:30:00Z',
      resolved: false
    },
    {
      id: 'err-2',
      timestamp: '2024-01-15T14:25:00Z',
      level: 'error',
      message: 'Network request failed: 500 Internal Server Error',
      source: 'api',
      category: 'network',
      url: '/api/users/profile',
      count: 43,
      firstSeen: '2024-01-15T12:00:00Z',
      lastSeen: '2024-01-15T14:25:00Z',
      resolved: false
    },
    {
      id: 'err-3',
      timestamp: '2024-01-15T14:20:00Z',
      level: 'warning',
      message: 'Deprecated API endpoint used: /api/v1/users',
      source: 'api',
      category: 'system',
      count: 89,
      firstSeen: '2024-01-10T00:00:00Z',
      lastSeen: '2024-01-15T14:20:00Z',
      resolved: false
    },
    {
      id: 'err-4',
      timestamp: '2024-01-15T14:15:00Z',
      level: 'error',
      message: 'Authentication token expired',
      source: 'server',
      category: 'authentication',
      userId: 'user-456',
      count: 12,
      firstSeen: '2024-01-15T13:00:00Z',
      lastSeen: '2024-01-15T14:15:00Z',
      resolved: true
    },
    {
      id: 'err-5',
      timestamp: '2024-01-15T14:10:00Z',
      level: 'error',
      message: 'Validation failed: Email is required',
      source: 'client',
      category: 'validation',
      url: '/signup',
      count: 23,
      firstSeen: '2024-01-15T10:00:00Z',
      lastSeen: '2024-01-15T14:10:00Z',
      resolved: false
    }
  ], []);

  const errorTrends: ErrorTrend[] = useMemo(() => {
    const hours = timeRange === '1h' ? 1 : timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720;
    const data: ErrorTrend[] = [];
    
    for (let i = hours; i >= 0; i--) {
      const date = new Date();
      date.setHours(date.getHours() - i);
      
      const errors = Math.floor(Math.random() * 50) + 10;
      const warnings = Math.floor(Math.random() * 30) + 5;
      
      data.push({
        date: timeRange === '1h' || timeRange === '24h' ? 
          date.toLocaleTimeString() : 
          date.toLocaleDateString(),
        errors,
        warnings,
        total: errors + warnings
      });
    }
    
    return data;
  }, [timeRange]);

  const filteredErrors = useMemo(() => {
    return errorLogs.filter(error => {
      const matchesSearch = searchTerm === '' || 
        error.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        error.category.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesLevel = selectedLevel === 'all' || error.level === selectedLevel;
      const matchesSource = selectedSource === 'all' || error.source === selectedSource;
      
      return matchesSearch && matchesLevel && matchesSource;
    });
  }, [errorLogs, searchTerm, selectedLevel, selectedSource]);

  const getErrorStats = () => {
    const total = errorLogs.length;
    const errors = errorLogs.filter(e => e.level === 'error').length;
    const warnings = errorLogs.filter(e => e.level === 'warning').length;
    const resolved = errorLogs.filter(e => e.resolved).length;
    const unresolved = total - resolved;
    
    return { total, errors, warnings, resolved, unresolved };
  };

  const getLevelColor = (level: ErrorLog['level']) => {
    switch (level) {
      case 'error': return 'destructive';
      case 'warning': return 'secondary';
      case 'info': return 'outline';
    }
  };

  const getSourceIcon = (source: ErrorLog['source']) => {
    switch (source) {
      case 'client': return <Globe className="h-4 w-4" />;
      case 'server': return <Code className="h-4 w-4" />;
      case 'api': return <RefreshCw className="h-4 w-4" />;
      case 'database': return <Bug className="h-4 w-4" />;
    }
  };

  const stats = getErrorStats();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Error Tracker</h2>
          <p className="text-muted-foreground">Monitor and track application errors and warnings</p>
        </div>
        <div className="flex gap-2">
          {(['1h', '24h', '7d', '30d'] as const).map((range) => (
            <Button
              key={range}
              variant={timeRange === range ? 'default' : 'outline'}
              size="sm"
              onClick={() => setTimeRange(range)}
            >
              {range}
            </Button>
          ))}
        </div>
      </div>

      {/* Error Statistics */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{stats.total}</div>
            <p className="text-xs text-muted-foreground">Total Errors</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-red-600">{stats.errors}</div>
            <p className="text-xs text-muted-foreground">Critical Errors</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-yellow-600">{stats.warnings}</div>
            <p className="text-xs text-muted-foreground">Warnings</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-green-600">{stats.resolved}</div>
            <p className="text-xs text-muted-foreground">Resolved</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-orange-600">{stats.unresolved}</div>
            <p className="text-xs text-muted-foreground">Unresolved</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="errors" className="w-full">
        <TabsList>
          <TabsTrigger value="errors">Error Logs</TabsTrigger>
          <TabsTrigger value="trends">Error Trends</TabsTrigger>
          <TabsTrigger value="analysis">Error Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="errors">
          <div className="space-y-4">
            {/* Filters */}
            <div className="flex gap-4 flex-wrap">
              <div className="flex-1 min-w-64">
                <Input
                  placeholder="Search errors..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full"
                />
              </div>
              <Select value={selectedLevel} onValueChange={setSelectedLevel}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="info">Info</SelectItem>
                </SelectContent>
              </Select>
              <Select value={selectedSource} onValueChange={setSelectedSource}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sources</SelectItem>
                  <SelectItem value="client">Client</SelectItem>
                  <SelectItem value="server">Server</SelectItem>
                  <SelectItem value="api">API</SelectItem>
                  <SelectItem value="database">Database</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Error List */}
            <div className="space-y-3">
              {filteredErrors.map((error) => (
                <Card key={error.id} className={error.resolved ? 'opacity-60' : ''}>
                  <CardContent className="pt-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          {getSourceIcon(error.source)}
                          <Badge variant={getLevelColor(error.level)}>
                            {error.level.toUpperCase()}
                          </Badge>
                          <Badge variant="outline" className="capitalize">
                            {error.category}
                          </Badge>
                          <Badge variant="outline">
                            {error.source}
                          </Badge>
                          <span className="text-sm text-muted-foreground">
                            {error.count} occurrences
                          </span>
                          {error.resolved && (
                            <Badge variant="default">Resolved</Badge>
                          )}
                        </div>
                        
                        <h4 className="font-medium mb-1">{error.message}</h4>
                        
                        <div className="text-sm text-muted-foreground space-y-1">
                          {error.url && <p>URL: {error.url}</p>}
                          {error.userId && <p>User: {error.userId}</p>}
                          {error.stack && <p>Stack: {error.stack}</p>}
                          <p>
                            First seen: {new Date(error.firstSeen).toLocaleString()}
                            {' | '}
                            Last seen: {new Date(error.lastSeen).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex gap-2 ml-4">
                        <Button size="sm" variant="outline" onClick={() => console.log("ErrorTracker button clicked")}>
                          View Details
                        </Button>
                        {!error.resolved && (
                          <Button size="sm" onClick={() => console.log("ErrorTracker button clicked")}>
                            Mark Resolved
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="trends">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Error Trends Over Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={errorTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="errors" 
                      stroke="hsl(var(--destructive))" 
                      strokeWidth={2}
                      name="Errors"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="warnings" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      name="Warnings"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Errors by Category</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[
                      { category: 'JavaScript', count: 127 },
                      { category: 'Network', count: 89 },
                      { category: 'Authentication', count: 43 },
                      { category: 'Validation', count: 23 },
                      { category: 'System', count: 12 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="category" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Errors by Source</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[
                      { source: 'Client', count: 150 },
                      { source: 'API', count: 132 },
                      { source: 'Server', count: 89 },
                      { source: 'Database', count: 23 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="source" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};