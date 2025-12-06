import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { useToast } from '@/hooks/use-toast';
import { 
  GitBranch, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  Play,
  Pause,
  RotateCcw,
  GitCommit,
  GitMerge,
  Users,
  Timer
} from 'lucide-react';

interface Pipeline {
  id: string;
  name: string;
  repository: string;
  branch: string;
  commit: string;
  commitMessage: string;
  author: string;
  status: 'queued' | 'running' | 'success' | 'failed' | 'cancelled';
  startTime: Date;
  endTime?: Date;
  duration?: number;
  stages: PipelineStage[];
  triggeredBy: 'push' | 'pull_request' | 'manual' | 'scheduled';
}

interface PipelineStage {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  logs?: string[];
  coverage?: number;
  testResults?: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
  };
}

export const CICDMonitor = () => {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [activePipeline, setActivePipeline] = useState<Pipeline | null>(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const { toast } = useToast();

  const mockPipelines: Pipeline[] = [
    {
      id: 'pipeline-1',
      name: 'main-pipeline',
      repository: 'repz-fitness/main',
      branch: 'main',
      commit: 'a1b2c3d',
      commitMessage: 'feat: add comprehensive testing suite',
      author: 'developer@repz.com',
      status: 'running',
      startTime: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
      stages: [
        {
          id: 'build',
          name: 'Build & Compile',
          status: 'success',
          startTime: new Date(Date.now() - 1000 * 60 * 15),
          endTime: new Date(Date.now() - 1000 * 60 * 12),
          duration: 180000
        },
        {
          id: 'test',
          name: 'Unit Tests',
          status: 'running',
          startTime: new Date(Date.now() - 1000 * 60 * 12),
          coverage: 87,
          testResults: {
            total: 245,
            passed: 230,
            failed: 3,
            skipped: 12
          }
        },
        {
          id: 'security',
          name: 'Security Scan',
          status: 'pending'
        },
        {
          id: 'deploy',
          name: 'Deploy to Staging',
          status: 'pending'
        }
      ],
      triggeredBy: 'push'
    },
    {
      id: 'pipeline-2',
      name: 'feature-branch',
      repository: 'repz-fitness/main',
      branch: 'feature/qa-dashboard',
      commit: 'x9y8z7w',
      commitMessage: 'refactor: improve test performance',
      author: 'qa@repz.com',
      status: 'success',
      startTime: new Date(Date.now() - 1000 * 60 * 60), // 1 hour ago
      endTime: new Date(Date.now() - 1000 * 60 * 45), // 45 minutes ago
      duration: 900000, // 15 minutes
      stages: [
        {
          id: 'build',
          name: 'Build & Compile',
          status: 'success',
          duration: 165000,
          coverage: 92
        },
        {
          id: 'test',
          name: 'Unit Tests',
          status: 'success',
          duration: 420000,
          testResults: {
            total: 245,
            passed: 242,
            failed: 0,
            skipped: 3
          }
        },
        {
          id: 'integration',
          name: 'Integration Tests',
          status: 'success',
          duration: 315000
        }
      ],
      triggeredBy: 'pull_request'
    },
    {
      id: 'pipeline-3',
      name: 'hotfix-pipeline',
      repository: 'repz-fitness/main',
      branch: 'hotfix/security-patch',
      commit: 'h0t1f2x',
      commitMessage: 'fix: security vulnerability in auth',
      author: 'security@repz.com',
      status: 'failed',
      startTime: new Date(Date.now() - 1000 * 60 * 120), // 2 hours ago
      endTime: new Date(Date.now() - 1000 * 60 * 110), // 1h 50m ago
      duration: 600000, // 10 minutes
      stages: [
        {
          id: 'build',
          name: 'Build & Compile',
          status: 'success',
          duration: 145000
        },
        {
          id: 'security',
          name: 'Security Scan',
          status: 'failed',
          duration: 280000
        }
      ],
      triggeredBy: 'manual'
    }
  ];

  const startMonitoring = () => {
    setIsMonitoring(true);
    setPipelines(mockPipelines);
    setActivePipeline(mockPipelines.find(p => p.status === 'running') || null);

    toast({
      title: "CI/CD Monitoring Started",
      description: "Now monitoring pipeline status",
    });
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    setActivePipeline(null);
    
    toast({
      title: "CI/CD Monitoring Stopped",
      description: "Pipeline monitoring has been disabled",
    });
  };

  const triggerPipeline = (type: 'main' | 'feature' | 'hotfix') => {
    const newPipeline: Pipeline = {
      id: `pipeline-${Date.now()}`,
      name: `${type}-pipeline`,
      repository: 'repz-fitness/main',
      branch: type === 'main' ? 'main' : type === 'feature' ? 'feature/new-feature' : 'hotfix/urgent-fix',
      commit: Math.random().toString(36).substring(2, 8),
      commitMessage: `${type}: triggered manually`,
      author: 'user@repz.com',
      status: 'queued',
      startTime: new Date(),
      stages: [
        { id: 'build', name: 'Build & Compile', status: 'pending' },
        { id: 'test', name: 'Unit Tests', status: 'pending' },
        { id: 'security', name: 'Security Scan', status: 'pending' },
        { id: 'deploy', name: 'Deploy', status: 'pending' }
      ],
      triggeredBy: 'manual'
    };

    setPipelines(prev => [newPipeline, ...prev]);
    setActivePipeline(newPipeline);

    toast({
      title: "Pipeline Triggered",
      description: `${type} pipeline has been queued`,
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-600" />;
      case 'running': return <div className="h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />;
      case 'queued': return <Clock className="h-4 w-4 text-gray-600" />;
      case 'cancelled': return <XCircle className="h-4 w-4 text-gray-600" />;
      default: return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-600 bg-green-50 border-green-200';
      case 'failed': return 'text-red-600 bg-red-50 border-red-200';
      case 'running': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'queued': return 'text-gray-600 bg-gray-50 border-gray-200';
      case 'cancelled': return 'text-gray-600 bg-gray-50 border-gray-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getTriggerIcon = (trigger: string) => {
    switch (trigger) {
      case 'push': return <GitCommit className="h-4 w-4" />;
      case 'pull_request': return <GitMerge className="h-4 w-4" />;
      case 'manual': return <Play className="h-4 w-4" />;
      case 'scheduled': return <Timer className="h-4 w-4" />;
      default: return <GitBranch className="h-4 w-4" />;
    }
  };

  const formatDuration = (ms: number) => {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const getOverallStats = () => {
    const total = pipelines.length;
    const successful = pipelines.filter(p => p.status === 'success').length;
    const failed = pipelines.filter(p => p.status === 'failed').length;
    const running = pipelines.filter(p => p.status === 'running').length;
    const successRate = total > 0 ? Math.round((successful / total) * 100) : 0;

    return { total, successful, failed, running, successRate };
  };

  const stats = getOverallStats();

  useEffect(() => {
    if (isMonitoring && activePipeline?.status === 'running') {
      const interval = setInterval(() => {
        // Simulate pipeline progress
        setActivePipeline(prev => {
          if (!prev) return null;
          
          const updatedStages = prev.stages.map(stage => {
            if (stage.status === 'running') {
              // Continue running
              return stage;
            } else if (stage.status === 'pending') {
              // Start next pending stage
              return { ...stage, status: 'running' as const, startTime: new Date() };
            }
            return stage;
          });

          return { ...prev, stages: updatedStages };
        });
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [isMonitoring, activePipeline?.status]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitBranch className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">CI/CD Pipeline Monitor</h2>
        </div>
        <div className="flex gap-2">
          {!isMonitoring ? (
            <Button onClick={startMonitoring} className="gap-2">
              <Play className="h-4 w-4" />
              Start Monitoring
            </Button>
          ) : (
            <Button onClick={stopMonitoring} variant="outline" className="gap-2">
              <Pause className="h-4 w-4" />
              Stop Monitoring
            </Button>
          )}
        </div>
      </div>

      {/* Pipeline Stats */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{stats.total}</div>
            <p className="text-xs text-muted-foreground">Total Pipelines</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-green-600">{stats.successful}</div>
            <p className="text-xs text-muted-foreground">Successful</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">{stats.failed}</div>
            <p className="text-xs text-muted-foreground">Failed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-blue-600">{stats.running}</div>
            <p className="text-xs text-muted-foreground">Running</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{stats.successRate}%</div>
            <p className="text-xs text-muted-foreground">Success Rate</p>
          </CardContent>
        </Card>
      </div>

      {/* Manual Pipeline Triggers */}
      <Card>
        <CardHeader>
          <CardTitle>Trigger Pipeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button onClick={() => triggerPipeline('main')} variant="outline">
              <GitCommit className="h-4 w-4 mr-2" />
              Main Branch
            </Button>
            <Button onClick={() => triggerPipeline('feature')} variant="outline">
              <GitBranch className="h-4 w-4 mr-2" />
              Feature Branch
            </Button>
            <Button onClick={() => triggerPipeline('hotfix')} variant="outline">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Hotfix
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Active Pipeline */}
      {activePipeline && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(activePipeline.status)}
                Active Pipeline: {activePipeline.name}
                <Badge className={getStatusColor(activePipeline.status)}>
                  {activePipeline.status.toUpperCase()}
                </Badge>
              </div>
              <div className="text-sm text-muted-foreground">
                {activePipeline.branch} • {activePipeline.commit}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-sm text-muted-foreground">
              <div className="flex items-center gap-2 mb-2">
                <Users className="h-4 w-4" />
                {activePipeline.author} • {activePipeline.commitMessage}
              </div>
              <div className="flex items-center gap-2">
                {getTriggerIcon(activePipeline.triggeredBy)}
                Triggered by {activePipeline.triggeredBy}
              </div>
            </div>

            {/* Pipeline Stages */}
            <div className="space-y-3">
              {activePipeline.stages.map((stage, index) => (
                <div key={stage.id} className="space-y-2">
                  <div className="flex items-center justify-between p-3 border rounded">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(stage.status)}
                      <div>
                        <div className="font-medium">{stage.name}</div>
                        {stage.duration && (
                          <div className="text-sm text-muted-foreground">
                            Duration: {formatDuration(stage.duration)}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="text-right text-sm">
                      {stage.coverage && (
                        <div className="text-green-600">Coverage: {stage.coverage}%</div>
                      )}
                      {stage.testResults && (
                        <div className="text-muted-foreground">
                          Tests: {stage.testResults.passed}/{stage.testResults.total}
                        </div>
                      )}
                    </div>
                  </div>

                  {stage.testResults && (
                    <div className="ml-6 space-y-2">
                      <Progress 
                        value={(stage.testResults.passed / stage.testResults.total) * 100} 
                        className="h-2"
                      />
                      <div className="grid grid-cols-4 gap-2 text-xs">
                        <div className="text-green-600">
                          Passed: {stage.testResults.passed}
                        </div>
                        <div className="text-red-600">
                          Failed: {stage.testResults.failed}
                        </div>
                        <div className="text-yellow-600">
                          Skipped: {stage.testResults.skipped}
                        </div>
                        <div className="text-muted-foreground">
                          Total: {stage.testResults.total}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pipeline History */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Pipelines</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {pipelines.slice(0, 10).map((pipeline) => (
              <div key={pipeline.id} className="flex items-center justify-between p-3 border rounded">
                <div className="flex items-center gap-3">
                  {getStatusIcon(pipeline.status)}
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      {pipeline.name}
                      <Badge variant="outline">{pipeline.branch}</Badge>
                      {getTriggerIcon(pipeline.triggeredBy)}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {pipeline.commitMessage} • {pipeline.author}
                    </div>
                  </div>
                </div>
                <div className="text-right text-sm">
                  <div>{pipeline.startTime.toLocaleString()}</div>
                  {pipeline.duration && (
                    <div className="text-muted-foreground">
                      {formatDuration(pipeline.duration)}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Pipeline Configuration Tips */}
      <Card>
        <CardHeader>
          <CardTitle>CI/CD Best Practices</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">🚀 Pipeline Optimization</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Use parallel stages to reduce pipeline duration</li>
              <li>• Cache dependencies between builds</li>
              <li>• Fail fast with early security and quality checks</li>
              <li>• Use conditional deployments based on branch</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">🔍 Monitoring & Alerts</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Set up notifications for failed pipelines</li>
              <li>• Monitor pipeline success rates and trends</li>
              <li>• Track deployment frequency and lead time</li>
              <li>• Implement automatic rollback on failure</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">🛡️ Security Integration</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Include SAST/DAST scans in every pipeline</li>
              <li>• Scan dependencies for vulnerabilities</li>
              <li>• Use signed commits and verify signatures</li>
              <li>• Implement environment-specific approvals</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};