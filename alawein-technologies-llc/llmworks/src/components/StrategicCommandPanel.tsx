import { memo, useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Terminal, 
  Settings, 
  Play, 
  Pause, 
  Square, 
  RotateCcw, 
  Zap, 
  Shield, 
  Eye, 
  Users,
  Cpu,
  HardDrive,
  Network,
  ChevronRight,
  AlertTriangle,
  CheckCircle,
  Clock,
  Search,
  Filter,
  Download,
  Upload,
  Database,
  Code,
  Layers,
  Lock,
  Unlock,
  Power,
  Volume2,
  Bell
} from "lucide-react";

interface CommandModule {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'maintenance' | 'error';
  cpu: number;
  memory: number;
  uptime: string;
  version: string;
  lastUpdate: string;
}

interface ActiveSession {
  id: string;
  user: string;
  model: string;
  task: string;
  duration: string;
  progress: number;
  priority: 'critical' | 'high' | 'medium' | 'low';
}

interface SystemLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  module: string;
  message: string;
}

const StrategicCommandPanelComponent = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [commandInput, setCommandInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [notifications, setNotifications] = useState(3);
  const terminalRef = useRef<HTMLDivElement>(null);

  const commandModules: CommandModule[] = [
    { 
      id: 'eval-engine', 
      name: 'Evaluation Engine', 
      status: 'online', 
      cpu: 67, 
      memory: 45, 
      uptime: '7d 14h 23m', 
      version: 'v2.1.4',
      lastUpdate: '2 hours ago'
    },
    { 
      id: 'model-registry', 
      name: 'Model Registry', 
      status: 'online', 
      cpu: 23, 
      memory: 18, 
      uptime: '12d 8h 45m', 
      version: 'v1.8.2',
      lastUpdate: '6 hours ago'
    },
    { 
      id: 'result-analyzer', 
      name: 'Result Analyzer', 
      status: 'maintenance', 
      cpu: 0, 
      memory: 12, 
      uptime: '0h 0m', 
      version: 'v1.5.1',
      lastUpdate: '1 hour ago'
    },
    { 
      id: 'security-monitor', 
      name: 'Security Monitor', 
      status: 'online', 
      cpu: 12, 
      memory: 8, 
      uptime: '15d 3h 12m', 
      version: 'v3.0.1',
      lastUpdate: '3 minutes ago'
    },
  ];

  const activeSessions: ActiveSession[] = [
    { 
      id: '1', 
      user: 'admin@llmworks.dev', 
      model: 'GPT-4 Turbo', 
      task: 'Strategic Analysis Benchmark',
      duration: '14:23',
      progress: 78,
      priority: 'high'
    },
    { 
      id: '2', 
      user: 'researcher@university.edu', 
      model: 'Claude-3 Opus', 
      task: 'Creative Problem Solving',
      duration: '06:45',
      progress: 34,
      priority: 'medium'
    },
    { 
      id: '3', 
      user: 'devops@company.com', 
      model: 'Gemini Ultra', 
      task: 'Performance Stress Test',
      duration: '28:17',
      progress: 92,
      priority: 'critical'
    },
  ];

  const systemLogs: SystemLog[] = [
    { 
      id: '1', 
      timestamp: '14:23:45', 
      level: 'info', 
      module: 'EVAL_ENGINE', 
      message: 'Strategic evaluation batch completed successfully - 247 models processed'
    },
    { 
      id: '2', 
      timestamp: '14:22:31', 
      level: 'warning', 
      module: 'SECURITY', 
      message: 'Unusual request pattern detected from IP 192.168.1.45 - rate limiting applied'
    },
    { 
      id: '3', 
      timestamp: '14:21:18', 
      level: 'success', 
      module: 'REGISTRY', 
      message: 'New model registered: LLaMA-3-70B-Instruct with strategic capabilities'
    },
    { 
      id: '4', 
      timestamp: '14:20:09', 
      level: 'error', 
      module: 'ANALYZER', 
      message: 'Memory threshold exceeded in result processing - automatic cleanup initiated'
    },
    { 
      id: '5', 
      timestamp: '14:19:42', 
      level: 'info', 
      module: 'NETWORK', 
      message: 'Load balancer health check passed - all nodes operational'
    },
  ];

  const executeCommand = async () => {
    if (!commandInput.trim()) return;
    
    setIsExecuting(true);
    setCommandHistory(prev => [...prev, `$ ${commandInput}`]);
    
    // Simulate command execution
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Mock responses
    const mockResponses = {
      'status': 'System Status: All modules operational ✓\nEvaluation Engine: Online (67% CPU)\nModel Registry: Online (23% CPU)\nSecurity Monitor: Online (12% CPU)',
      'help': 'Available commands:\n  status - Show system status\n  restart <module> - Restart a module\n  logs <level> - Show filtered logs\n  sessions - List active sessions\n  metrics - Display performance metrics',
      'logs error': 'Error logs from last hour:\n[14:20:09] ANALYZER: Memory threshold exceeded\n[13:45:23] NETWORK: Connection timeout to node-3',
      'metrics': 'Performance Metrics:\nCPU Usage: 45.7%\nMemory Usage: 62.3%\nNetwork I/O: 1.2 GB/s\nActive Sessions: 247\nToday\'s Evaluations: 12,847',
    };

    const response = mockResponses[commandInput.toLowerCase() as keyof typeof mockResponses] || 
      `Command executed: ${commandInput}\nResponse: Operation completed successfully`;
    
    setCommandHistory(prev => [...prev, response, '']);
    setCommandInput('');
    setIsExecuting(false);

    // Auto-scroll to bottom
    setTimeout(() => {
      if (terminalRef.current) {
        terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
      }
    }, 100);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-500 bg-green-500/10 border-green-500/30';
      case 'offline': return 'text-gray-500 bg-gray-500/10 border-gray-500/30';
      case 'maintenance': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30';
      case 'error': return 'text-red-500 bg-red-500/10 border-red-500/30';
      default: return 'text-muted-foreground bg-muted/10 border-muted/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'offline': return <Power className="h-4 w-4 text-gray-500" />;
      case 'maintenance': return <Settings className="h-4 w-4 text-yellow-500" />;
      case 'error': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'info': return 'text-blue-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      case 'success': return 'text-green-400';
      default: return 'text-muted-foreground';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'rank-platinum strategic-rank';
      case 'high': return 'rank-gold strategic-rank';
      case 'medium': return 'rank-silver strategic-rank';
      case 'low': return 'rank-bronze strategic-rank';
      default: return 'performance-standard strategic-rank';
    }
  };

  return (
    <div className="glass-panel w-full max-w-7xl mx-auto">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="glass-subtle p-2 rounded-xl">
              <Terminal className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="heading-refined text-lg">Strategic Command Panel</CardTitle>
              <p className="text-xs text-muted-foreground">
                Advanced system control and monitoring interface
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="glass-minimal">
              <Bell className="h-4 w-4 mr-2" />
              <span className="text-xs">{notifications}</span>
            </Button>
            <Badge className="performance-elite strategic-rank">AUTHORIZED</Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="glass-subtle p-1 rounded-xl grid grid-cols-4 w-full max-w-md">
            <TabsTrigger value="overview" className="text-xs">Overview</TabsTrigger>
            <TabsTrigger value="modules" className="text-xs">Modules</TabsTrigger>
            <TabsTrigger value="sessions" className="text-xs">Sessions</TabsTrigger>
            <TabsTrigger value="terminal" className="text-xs">Terminal</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {/* System Overview */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* System Health */}
              <Card className="glass-subtle border-border/20">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Shield className="h-4 w-4 text-primary" />
                    System Health
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Overall Status</span>
                    <Badge className="performance-elite strategic-rank">OPTIMAL</Badge>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>CPU Usage</span>
                      <span>45.7%</span>
                    </div>
                    <Progress value={45.7} className="h-1" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>Memory</span>
                      <span>62.3%</span>
                    </div>
                    <Progress value={62.3} className="h-1" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>Network I/O</span>
                      <span>1.2 GB/s</span>
                    </div>
                    <Progress value={78} className="h-1" />
                  </div>
                </CardContent>
              </Card>

              {/* Active Operations */}
              <Card className="glass-subtle border-border/20">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Zap className="h-4 w-4 text-primary" />
                    Active Operations
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Evaluations Today</span>
                    <span className="text-sm font-bold text-primary">12,847</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Active Sessions</span>
                    <span className="text-sm font-bold text-secondary">247</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Models Online</span>
                    <span className="text-sm font-bold text-accent">47</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Queue Depth</span>
                    <span className="text-sm font-bold text-green-500">23</span>
                  </div>
                </CardContent>
              </Card>

              {/* Security Status */}
              <Card className="glass-subtle border-border/20">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Lock className="h-4 w-4 text-primary" />
                    Security Status
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Threat Level</span>
                    <Badge className="performance-superior strategic-rank">LOW</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Failed Attempts</span>
                    <span className="text-sm font-bold text-red-500">0</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Active Firewalls</span>
                    <span className="text-sm font-bold text-green-500">3</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Last Scan</span>
                    <span className="text-xs text-muted-foreground">2m ago</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="modules" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="heading-refined text-sm">System Modules</h3>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="glass-minimal">
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
                <Button variant="outline" size="sm" className="glass-minimal">
                  <Settings className="h-4 w-4 mr-2" />
                  Configure
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {commandModules.map((module) => (
                <Card key={module.id} className="glass-subtle border-border/20 hover:shadow-md transition-all duration-300">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        {getStatusIcon(module.status)}
                        <div>
                          <div className="text-sm font-medium">{module.name}</div>
                          <div className="text-xs text-muted-foreground">{module.version}</div>
                        </div>
                      </div>
                      <Badge className={`text-xs ${getStatusColor(module.status)}`}>
                        {module.status.toUpperCase()}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-3">
                      <div>
                        <div className="text-xs text-muted-foreground">CPU</div>
                        <div className="flex items-center gap-2">
                          <Progress value={module.cpu} className="h-1 flex-1" />
                          <span className="text-xs font-mono">{module.cpu}%</span>
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-muted-foreground">Memory</div>
                        <div className="flex items-center gap-2">
                          <Progress value={module.memory} className="h-1 flex-1" />
                          <span className="text-xs font-mono">{module.memory}%</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Uptime: {module.uptime}</span>
                      <span>Updated: {module.lastUpdate}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="sessions" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="heading-refined text-sm">Active Sessions</h3>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="glass-minimal">
                  <Search className="h-4 w-4 mr-2" />
                  Search
                </Button>
                <Button variant="outline" size="sm" className="glass-minimal">
                  <Filter className="h-4 w-4 mr-2" />
                  Filter
                </Button>
              </div>
            </div>

            <div className="space-y-3">
              {activeSessions.map((session) => (
                <Card key={session.id} className="glass-subtle border-border/20 hover:shadow-md transition-all duration-300">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <Badge className={getPriorityColor(session.priority)}>
                          {session.priority}
                        </Badge>
                        <div>
                          <div className="text-sm font-medium">{session.task}</div>
                          <div className="text-xs text-muted-foreground">
                            {session.user} • {session.model}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-primary">{session.progress}%</div>
                        <div className="text-xs text-muted-foreground">{session.duration}</div>
                      </div>
                    </div>
                    <Progress value={session.progress} className="h-2" />
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="terminal" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="heading-refined text-sm">Strategic Command Terminal</h3>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="glass-minimal">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
                <Button variant="outline" size="sm" className="glass-minimal">
                  <Code className="h-4 w-4 mr-2" />
                  Clear
                </Button>
              </div>
            </div>

            {/* Terminal */}
            <Card className="glass-subtle border-border/20 bg-black/20">
              <CardContent className="p-0">
                <div 
                  ref={terminalRef}
                  className="font-mono text-xs text-green-400 p-4 h-64 overflow-y-auto scrollbar-elegant"
                  style={{ backgroundColor: 'rgba(0, 0, 0, 0.4)' }}
                >
                  <div className="text-primary mb-2">LLM Works Strategic Command Terminal v2.1.4</div>
                  <div className="text-muted-foreground mb-4">Type 'help' for available commands</div>
                  
                  {commandHistory.map((line, index) => (
                    <div key={index} className={line.startsWith('$') ? 'text-blue-400' : 'text-green-400'}>
                      {line}
                    </div>
                  ))}
                  
                  <div className="flex items-center mt-2">
                    <span className="text-blue-400 mr-2">$</span>
                    <Input
                      value={commandInput}
                      onChange={(e) => setCommandInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          executeCommand();
                        }
                      }}
                      className="border-0 bg-transparent text-green-400 p-0 h-auto focus-visible:ring-0"
                      placeholder="Enter command..."
                      disabled={isExecuting}
                    />
                    {isExecuting && (
                      <div className="ml-2 text-yellow-400 animate-pulse">executing...</div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* System Logs */}
            <Card className="glass-subtle border-border/20">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Eye className="h-4 w-4 text-primary" />
                  System Logs
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {systemLogs.map((log) => (
                  <div key={log.id} className="flex items-start gap-3 text-xs">
                    <span className="text-muted-foreground font-mono">{log.timestamp}</span>
                    <span className={`font-mono ${getLogLevelColor(log.level)} min-w-[4rem]`}>
                      [{log.level.toUpperCase()}]
                    </span>
                    <span className="text-primary font-mono">{log.module}:</span>
                    <span className="text-muted-foreground flex-1">{log.message}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </div>
  );
};

export const StrategicCommandPanel = memo(StrategicCommandPanelComponent);