import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { 
  Workflow,
  Clock,
  CheckCircle,
  Zap,
  Users,
  Calendar,
  Mail,
  Bell,
  FileText,
  TrendingUp,
  Settings,
  Play
} from 'lucide-react';

interface AutomationRule {
  id: string;
  name: string;
  description: string;
  trigger: string;
  action: string;
  status: 'active' | 'paused' | 'draft';
  executions: number;
  successRate: number;
  category: 'user-engagement' | 'health-monitoring' | 'business-ops' | 'content-management';
  lastExecution: Date;
}

interface WorkflowStep {
  id: string;
  name: string;
  type: 'trigger' | 'condition' | 'action' | 'delay';
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration: number;
  details: string;
}

interface AutomationMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  change: number;
  target: number;
  category: string;
}

export const EnterpriseAutomationEngine: React.FC = () => {
  const [automationRules, setAutomationRules] = useState<AutomationRule[]>([]);
  const [activeWorkflows, setActiveWorkflows] = useState<WorkflowStep[]>([]);
  const [automationMetrics, setAutomationMetrics] = useState<AutomationMetric[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState(0);

  useEffect(() => {
    initializeAutomationRules();
    initializeMetrics();
    simulateWorkflows();
  }, []);

  const initializeAutomationRules = () => {
    const rules: AutomationRule[] = [
      {
        id: 'engagement-1',
        name: 'Smart User Re-engagement',
        description: 'Automatically re-engage users who haven\'t logged in for 3 days',
        trigger: 'User inactive for 3 days',
        action: 'Send personalized motivation email + in-app notification',
        status: 'active',
        executions: 2847,
        successRate: 94.2,
        category: 'user-engagement',
        lastExecution: new Date(Date.now() - 2 * 60 * 60 * 1000)
      },
      {
        id: 'health-1',
        name: 'Health Alert System',
        description: 'Monitor health metrics and alert coaches for concerning trends',
        trigger: 'Health metric deviation > 15%',
        action: 'Alert coach + schedule check-in',
        status: 'active',
        executions: 1523,
        successRate: 98.7,
        category: 'health-monitoring',
        lastExecution: new Date(Date.now() - 45 * 60 * 1000)
      },
      {
        id: 'business-1',
        name: 'Automated Billing Management',
        description: 'Handle subscription changes and billing notifications',
        trigger: 'Billing event or subscription change',
        action: 'Update user tier + send confirmation + sync features',
        status: 'active',
        executions: 856,
        successRate: 99.1,
        category: 'business-ops',
        lastExecution: new Date(Date.now() - 30 * 60 * 1000)
      },
      {
        id: 'content-1',
        name: 'Dynamic Content Personalization',
        description: 'Automatically customize content based on user preferences and progress',
        trigger: 'User behavior pattern detected',
        action: 'Update content recommendations + adjust UI',
        status: 'active',
        executions: 4521,
        successRate: 91.8,
        category: 'content-management',
        lastExecution: new Date(Date.now() - 10 * 60 * 1000)
      },
      {
        id: 'engagement-2',
        name: 'Goal Achievement Celebration',
        description: 'Celebrate milestones and encourage continued progress',
        trigger: 'User achieves fitness milestone',
        action: 'Send celebration + share achievement + suggest next goal',
        status: 'active',
        executions: 1247,
        successRate: 96.3,
        category: 'user-engagement',
        lastExecution: new Date(Date.now() - 1 * 60 * 60 * 1000)
      },
      {
        id: 'health-2',
        name: 'Intelligent Rest Day Scheduler',
        description: 'Automatically schedule rest days based on recovery metrics',
        trigger: 'Recovery score < 70% for 2 days',
        action: 'Schedule rest day + send recovery plan',
        status: 'active',
        executions: 892,
        successRate: 93.5,
        category: 'health-monitoring',
        lastExecution: new Date(Date.now() - 3 * 60 * 60 * 1000)
      }
    ];

    setAutomationRules(rules);
  };

  const initializeMetrics = () => {
    const metrics: AutomationMetric[] = [
      { id: 'automation-rate', name: 'Automation Rate', value: 87.3, unit: '%', change: 12.5, target: 90, category: 'efficiency' },
      { id: 'time-saved', name: 'Time Saved/Day', value: 14.7, unit: 'hrs', change: 23.1, target: 16, category: 'productivity' },
      { id: 'user-satisfaction', name: 'User Satisfaction', value: 4.7, unit: '/5', change: 8.2, target: 4.8, category: 'experience' },
      { id: 'error-rate', name: 'Error Rate', value: 0.8, unit: '%', change: -45.2, target: 1, category: 'reliability' },
      { id: 'cost-reduction', name: 'Cost Reduction', value: 34.2, unit: '%', change: 18.7, target: 40, category: 'financial' },
      { id: 'response-time', name: 'Avg Response Time', value: 1.2, unit: 's', change: -28.3, target: 1, category: 'performance' }
    ];

    setAutomationMetrics(metrics);
  };

  const simulateWorkflows = () => {
    const workflows: WorkflowStep[] = [
      { id: 'w1', name: 'User Inactive Detection', type: 'trigger', status: 'completed', duration: 150, details: 'Detected 47 inactive users' },
      { id: 'w2', name: 'Engagement Score Calculation', type: 'condition', status: 'completed', duration: 89, details: 'Calculated personalized scores' },
      { id: 'w3', name: 'Email Campaign Trigger', type: 'action', status: 'running', duration: 0, details: 'Sending 23 personalized emails' },
      { id: 'w4', name: 'Health Metric Analysis', type: 'trigger', status: 'completed', duration: 234, details: 'Analyzed 156 user health profiles' },
      { id: 'w5', name: 'Coach Alert System', type: 'action', status: 'pending', duration: 0, details: 'Queued 3 coach notifications' }
    ];

    setActiveWorkflows(workflows);

    // Simulate workflow progression
    setInterval(() => {
      setActiveWorkflows(prev => prev.map(step => {
        if (step.status === 'pending' && Math.random() > 0.7) {
          return { ...step, status: 'running' as const };
        }
        if (step.status === 'running') {
          if (Math.random() > 0.8) {
            return { ...step, status: 'completed' as const, duration: step.duration + 50 };
          } else {
            return { ...step, duration: step.duration + 100 };
          }
        }
        return step;
      }));
    }, 2000);
  };

  const executeAutomation = async () => {
    setIsExecuting(true);
    setExecutionProgress(0);

    const executionSteps = [
      'Initializing automation engine',
      'Scanning user activity patterns',
      'Evaluating automation triggers',
      'Executing priority workflows',
      'Monitoring execution results',
      'Optimizing performance',
      'Updating success metrics',
      'Generating execution report'
    ];

    for (let i = 0; i < executionSteps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 600));
      setExecutionProgress(((i + 1) / executionSteps.length) * 100);
    }

    // Update metrics after execution
    setAutomationMetrics(prev => prev.map(metric => ({
      ...metric,
      value: metric.id === 'error-rate' ? 
        Math.max(0.1, metric.value * 0.95) : 
        metric.value * (1 + (Math.random() * 0.05))
    })));

    setTimeout(() => {
      setIsExecuting(false);
      setExecutionProgress(0);
    }, 1000);
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'user-engagement': return <Users className="h-4 w-4" />;
      case 'health-monitoring': return <TrendingUp className="h-4 w-4" />;
      case 'business-ops': return <FileText className="h-4 w-4" />;
      case 'content-management': return <Mail className="h-4 w-4" />;
      default: return <Workflow className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500/10 text-green-600 border-green-500/20';
      case 'paused': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      case 'draft': return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
      default: return '';
    }
  };

  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'running': return <div className="h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
      case 'failed': return <div className="h-4 w-4 bg-red-500 rounded-full" />;
      default: return <div className="h-4 w-4 border border-gray-300 rounded-full" />;
    }
  };

  const overallEfficiency = automationMetrics.find(m => m.id === 'automation-rate')?.value || 0;

  return (
    <div className="space-y-6">
      {/* Automation Engine Header */}
      <Card className="glass-card border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Workflow className="h-6 w-6 text-primary" />
              Enterprise Automation Engine
            </div>
            <Badge 
              variant="default" 
              className="bg-gradient-to-r from-blue-500 to-cyan-600 text-white text-lg px-4 py-2"
            >
              EFFICIENCY: {overallEfficiency.toFixed(1)}%
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{automationRules.filter(r => r.status === 'active').length}</div>
              <div className="text-sm text-muted-foreground">Active Rules</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-adaptive">{automationRules.reduce((acc, r) => acc + r.executions, 0).toLocaleString()}</div>
              <div className="text-sm text-muted-foreground">Executions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-performance">{(automationRules.reduce((acc, r) => acc + r.successRate, 0) / automationRules.length).toFixed(1)}%</div>
              <div className="text-sm text-muted-foreground">Success Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-longevity">{automationMetrics.find(m => m.id === 'time-saved')?.value.toFixed(1)}</div>
              <div className="text-sm text-muted-foreground">Hours Saved/Day</div>
            </div>
          </div>

          <div className="flex gap-4">
            <Button 
              onClick={executeAutomation}
              disabled={isExecuting}
              className="flex items-center gap-2"
            >
              {isExecuting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Executing...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Execute Automation
                </>
              )}
            </Button>
            
            {isExecuting && (
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span>Automation Progress</span>
                  <span>{executionProgress.toFixed(0)}%</span>
                </div>
                <Progress value={executionProgress} className="h-2" />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Automation Metrics */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-adaptive" />
            Automation Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {automationMetrics.map((metric) => (
              <div key={metric.id} className="bg-card/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">{metric.name}</span>
                  <Badge 
                    variant={metric.change > 0 ? 'default' : 'destructive'}
                    className="text-xs"
                  >
                    {metric.change > 0 ? '+' : ''}{metric.change.toFixed(1)}%
                  </Badge>
                </div>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">
                    {metric.value.toFixed(1)}
                    <span className="text-sm text-muted-foreground ml-1">{metric.unit}</span>
                  </div>
                  <Progress 
                    value={(metric.value / metric.target) * 100} 
                    className="h-2"
                  />
                  <div className="text-xs text-muted-foreground">
                    Target: {metric.target}{metric.unit}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Active Workflows */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-performance" />
            Active Workflows
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {activeWorkflows.map((step) => (
              <div key={step.id} className="flex items-center gap-4 p-3 bg-card/30 rounded-lg">
                {getStepStatusIcon(step.status)}
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{step.name}</span>
                    <Badge variant="outline" className="text-xs">
                      {step.type}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground">{step.details}</div>
                </div>
                {step.duration > 0 && (
                  <div className="text-right">
                    <div className="text-sm font-medium">{step.duration}ms</div>
                    <div className="text-xs text-muted-foreground">Duration</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Automation Rules */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {automationRules.map((rule) => (
          <Card key={rule.id} className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getCategoryIcon(rule.category)}
                  <span className="text-lg">{rule.name}</span>
                </div>
                <Badge className={getStatusColor(rule.status)}>
                  {rule.status}
                </Badge>
              </CardTitle>
              <p className="text-sm text-muted-foreground">{rule.description}</p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="text-sm">
                    <span className="font-medium">Trigger:</span> {rule.trigger}
                  </div>
                  <div className="text-sm">
                    <span className="font-medium">Action:</span> {rule.action}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-lg font-bold">{rule.executions.toLocaleString()}</div>
                    <div className="text-xs text-muted-foreground">Executions</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold">{rule.successRate.toFixed(1)}%</div>
                    <div className="text-xs text-muted-foreground">Success Rate</div>
                  </div>
                </div>

                <div className="text-xs text-muted-foreground">
                  Last executed: {rule.lastExecution.toLocaleTimeString()}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Automation Summary */}
      <Alert className="border-blue-500/20 bg-blue-500/5">
        <Workflow className="h-4 w-4 text-blue-500" />
        <AlertDescription className="text-blue-700">
          <strong>Automation Engine Status:</strong> Operating at {overallEfficiency.toFixed(1)}% efficiency with {automationRules.filter(r => r.status === 'active').length} active rules. 
          Saving {automationMetrics.find(m => m.id === 'time-saved')?.value.toFixed(1)} hours daily through intelligent automation.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default EnterpriseAutomationEngine;