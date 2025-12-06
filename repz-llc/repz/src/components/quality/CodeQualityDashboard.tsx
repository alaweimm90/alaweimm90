import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { useToast } from '@/hooks/use-toast';
import { 
  Code, 
  Bug, 
  Shield, 
  Zap, 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  FileText,
  GitBranch,
  Users
} from 'lucide-react';

interface CodeMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  threshold: number;
  status: 'excellent' | 'good' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  description: string;
  category: 'complexity' | 'coverage' | 'quality' | 'security' | 'maintainability';
}

interface QualityGate {
  id: string;
  name: string;
  conditions: QualityCondition[];
  status: 'passed' | 'failed' | 'warning';
  lastUpdated: Date;
}

interface QualityCondition {
  metric: string;
  operator: 'greater_than' | 'less_than' | 'equals';
  threshold: number;
  actual: number;
  passed: boolean;
}

interface CodeIssue {
  id: string;
  type: 'bug' | 'vulnerability' | 'code_smell' | 'security_hotspot';
  severity: 'blocker' | 'critical' | 'major' | 'minor' | 'info';
  message: string;
  file: string;
  line: number;
  rule: string;
  effort: string; // Time to fix
  tags: string[];
}

export const CodeQualityDashboard = () => {
  const [metrics, setMetrics] = useState<CodeMetric[]>([]);
  const [qualityGates, setQualityGates] = useState<QualityGate[]>([]);
  const [issues, setIssues] = useState<CodeIssue[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { toast } = useToast();

  const mockMetrics: CodeMetric[] = [
    {
      id: 'coverage',
      name: 'Code Coverage',
      value: 87.5,
      unit: '%',
      threshold: 80,
      status: 'good',
      trend: 'up',
      description: 'Percentage of code covered by tests',
      category: 'coverage'
    },
    {
      id: 'complexity',
      name: 'Cyclomatic Complexity',
      value: 12.3,
      unit: '',
      threshold: 15,
      status: 'good',
      trend: 'stable',
      description: 'Average complexity of functions',
      category: 'complexity'
    },
    {
      id: 'duplicated-lines',
      name: 'Duplicated Lines',
      value: 2.1,
      unit: '%',
      threshold: 3,
      status: 'excellent',
      trend: 'down',
      description: 'Percentage of duplicated lines of code',
      category: 'quality'
    },
    {
      id: 'technical-debt',
      name: 'Technical Debt',
      value: 4.2,
      unit: 'days',
      threshold: 7,
      status: 'good',
      trend: 'stable',
      description: 'Estimated time to fix all issues',
      category: 'maintainability'
    },
    {
      id: 'security-rating',
      name: 'Security Rating',
      value: 1,
      unit: '',
      threshold: 1,
      status: 'excellent',
      trend: 'stable',
      description: 'Security rating (1=A, 5=E)',
      category: 'security'
    },
    {
      id: 'maintainability',
      name: 'Maintainability Rating',
      value: 1,
      unit: '',
      threshold: 1,
      status: 'excellent',
      trend: 'stable',
      description: 'Maintainability rating based on technical debt',
      category: 'maintainability'
    },
    {
      id: 'reliability',
      name: 'Reliability Rating',
      value: 1,
      unit: '',
      threshold: 1,
      status: 'excellent',
      trend: 'stable',
      description: 'Reliability rating based on bugs',
      category: 'quality'
    }
  ];

  const mockQualityGates: QualityGate[] = [
    {
      id: 'main-gate',
      name: 'Main Branch Quality Gate',
      lastUpdated: new Date(),
      status: 'passed',
      conditions: [
        { metric: 'Coverage', operator: 'greater_than', threshold: 80, actual: 87.5, passed: true },
        { metric: 'Duplicated Lines', operator: 'less_than', threshold: 3, actual: 2.1, passed: true },
        { metric: 'Security Rating', operator: 'equals', threshold: 1, actual: 1, passed: true },
        { metric: 'Maintainability Rating', operator: 'equals', threshold: 1, actual: 1, passed: true }
      ]
    },
    {
      id: 'pr-gate',
      name: 'Pull Request Quality Gate',
      lastUpdated: new Date(Date.now() - 1000 * 60 * 30),
      status: 'warning',
      conditions: [
        { metric: 'New Coverage', operator: 'greater_than', threshold: 80, actual: 75.2, passed: false },
        { metric: 'New Duplicated Lines', operator: 'less_than', threshold: 3, actual: 1.8, passed: true },
        { metric: 'New Bugs', operator: 'equals', threshold: 0, actual: 0, passed: true },
        { metric: 'New Vulnerabilities', operator: 'equals', threshold: 0, actual: 0, passed: true }
      ]
    }
  ];

  const mockIssues: CodeIssue[] = [
    {
      id: 'issue-1',
      type: 'bug',
      severity: 'major',
      message: 'Possible null pointer dereference',
      file: 'src/components/Dashboard.tsx',
      line: 45,
      rule: 'javascript:S2259',
      effort: '10min',
      tags: ['defensive-programming', 'bug']
    },
    {
      id: 'issue-2',
      type: 'code_smell',
      severity: 'minor',
      message: 'Function has too many parameters',
      file: 'src/utils/helpers.ts',
      line: 123,
      rule: 'javascript:S107',
      effort: '30min',
      tags: ['brain-overload', 'complexity']
    },
    {
      id: 'issue-3',
      type: 'vulnerability',
      severity: 'critical',
      message: 'SQL injection vulnerability',
      file: 'src/services/api.ts',
      line: 67,
      rule: 'javascript:S3649',
      effort: '1h',
      tags: ['sql-injection', 'security', 'cwe']
    },
    {
      id: 'issue-4',
      type: 'security_hotspot',
      severity: 'major',
      message: 'Make sure this regex is safe',
      file: 'src/utils/validation.ts',
      line: 12,
      rule: 'javascript:S5852',
      effort: '15min',
      tags: ['regex', 'security']
    }
  ];

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    
    toast({
      title: "Code Analysis Started",
      description: "Analyzing code quality metrics...",
    });

    // Simulate analysis time
    await new Promise(resolve => setTimeout(resolve, 3000));

    setMetrics(mockMetrics);
    setQualityGates(mockQualityGates);
    setIssues(mockIssues);
    setIsAnalyzing(false);

    toast({
      title: "Code Analysis Complete",
      description: "Quality metrics have been updated",
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-600 bg-green-50 border-green-200';
      case 'good': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'warning': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'passed': return 'text-green-600 bg-green-50 border-green-200';
      case 'failed': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent':
      case 'good':
      case 'passed': 
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'warning': 
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'critical':
      case 'failed': 
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default: 
        return <CheckCircle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'down': return <TrendingDown className="h-4 w-4 text-red-600" />;
      default: return <div className="h-4 w-4 rounded-full bg-gray-400" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'blocker': return 'text-red-800 bg-red-100';
      case 'critical': return 'text-red-700 bg-red-100';
      case 'major': return 'text-orange-700 bg-orange-100';
      case 'minor': return 'text-yellow-700 bg-yellow-100';
      case 'info': return 'text-blue-700 bg-blue-100';
      default: return 'text-gray-700 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'bug': return <Bug className="h-4 w-4 text-red-600" />;
      case 'vulnerability': return <Shield className="h-4 w-4 text-red-600" />;
      case 'code_smell': return <Code className="h-4 w-4 text-yellow-600" />;
      case 'security_hotspot': return <Shield className="h-4 w-4 text-orange-600" />;
      default: return <Code className="h-4 w-4 text-gray-600" />;
    }
  };

  const getCategoryIcon = (category: string): React.ComponentType<{ className?: string }> => {
    switch (category) {
      case 'complexity': return Zap;
      case 'coverage': return CheckCircle;
      case 'quality': return Code;
      case 'security': return Shield;
      case 'maintainability': return FileText;
      default: return Code;
    }
  };

  const getOverallScore = () => {
    if (metrics.length === 0) return 0;
    
    const scores = metrics.map(metric => {
      switch (metric.status) {
        case 'excellent': return 100;
        case 'good': return 80;
        case 'warning': return 60;
        case 'critical': return 30;
        default: return 70;
      }
    });

    return Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length);
  };

  const overallScore = getOverallScore();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Code className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">Code Quality Dashboard</h2>
        </div>
        <Button 
          onClick={runAnalysis}
          disabled={isAnalyzing}
          className="gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${isAnalyzing ? 'animate-spin' : ''}`} />
          {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
        </Button>
      </div>

      {/* Overall Score */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Overall Code Quality Score</span>
            <div className="text-3xl font-bold">
              {overallScore}%
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={overallScore} className="h-4" />
          <div className="mt-2 text-sm text-muted-foreground">
            Based on {metrics.length} quality metrics
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="metrics" className="w-full">
        <TabsList>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="gates">Quality Gates</TabsTrigger>
          <TabsTrigger value="issues">Issues</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {metrics.map((metric) => {
              const IconComponent = getCategoryIcon(metric.category);
              return (
                <Card key={metric.id}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <IconComponent className="h-4 w-4" />
                        {metric.name}
                      </div>
                      <div className="flex items-center gap-1">
                        {getStatusIcon(metric.status)}
                        {getTrendIcon(metric.trend)}
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="text-2xl font-bold">
                        {metric.value}{metric.unit}
                      </div>
                      <Progress 
                        value={metric.category === 'complexity' || metric.category === 'quality' 
                          ? ((metric.threshold - metric.value) / metric.threshold) * 100
                          : (metric.value / metric.threshold) * 100
                        } 
                        className="h-2"
                      />
                      <div className="text-xs text-muted-foreground">
                        Threshold: {metric.threshold}{metric.unit}
                      </div>
                      <Badge className={getStatusColor(metric.status)} variant="outline">
                        {metric.status.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="mt-3 text-sm text-muted-foreground">
                      {metric.description}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="gates">
          <div className="space-y-4">
            {qualityGates.map((gate) => (
              <Card key={gate.id}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <GitBranch className="h-5 w-5" />
                      {gate.name}
                    </div>
                    <Badge className={getStatusColor(gate.status)}>
                      {gate.status.toUpperCase()}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="text-sm text-muted-foreground">
                      Last updated: {gate.lastUpdated.toLocaleString()}
                    </div>
                    
                    <div className="space-y-2">
                      {gate.conditions.map((condition, index) => (
                        <div key={index} className={`p-3 border rounded ${condition.passed ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {condition.passed ? (
                                <CheckCircle className="h-4 w-4 text-green-600" />
                              ) : (
                                <AlertTriangle className="h-4 w-4 text-red-600" />
                              )}
                              <span className="font-medium">{condition.metric}</span>
                            </div>
                            <div className="text-sm">
                              {condition.actual} {condition.operator.replace('_', ' ')} {condition.threshold}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="issues">
          <div className="space-y-4">
            {issues.map((issue) => (
              <Card key={issue.id}>
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      {getTypeIcon(issue.type)}
                      <div>
                        <div className="font-medium">{issue.message}</div>
                        <div className="text-sm text-muted-foreground">
                          {issue.file}:{issue.line}
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Badge className={getSeverityColor(issue.severity)}>
                        {issue.severity.toUpperCase()}
                      </Badge>
                      <Badge variant="outline">{issue.type.replace('_', ' ')}</Badge>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="text-sm">
                      <span className="font-medium">Rule:</span> {issue.rule}
                    </div>
                    <div className="text-sm">
                      <span className="font-medium">Effort to fix:</span> {issue.effort}
                    </div>
                    <div className="flex gap-1">
                      {issue.tags.map((tag) => (
                        <Badge key={tag} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="trends">
          <Card>
            <CardHeader>
              <CardTitle>Quality Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="p-4 bg-green-50 rounded">
                    <div className="text-lg font-bold text-green-600">+5.2%</div>
                    <div className="text-sm text-green-700">Code Coverage</div>
                    <div className="text-xs text-green-600">vs last month</div>
                  </div>
                  <div className="p-4 bg-blue-50 rounded">
                    <div className="text-lg font-bold text-blue-600">-15%</div>
                    <div className="text-sm text-blue-700">Technical Debt</div>
                    <div className="text-xs text-blue-600">vs last month</div>
                  </div>
                  <div className="p-4 bg-red-50 rounded">
                    <div className="text-lg font-bold text-red-600">+3</div>
                    <div className="text-sm text-red-700">New Issues</div>
                    <div className="text-xs text-red-600">this week</div>
                  </div>
                </div>
                
                <div className="text-sm text-muted-foreground">
                  Quality trend analysis shows improvement in code coverage and reduction in technical debt.
                  Monitor new issues introduction rate to maintain quality standards.
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Code Quality Best Practices */}
      <Card>
        <CardHeader>
          <CardTitle>Code Quality Best Practices</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">📊 Quality Metrics</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Maintain code coverage above 80%</li>
              <li>• Keep cyclomatic complexity under 15</li>
              <li>• Minimize code duplication (under 3%)</li>
              <li>• Monitor technical debt and prioritize fixes</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">🚪 Quality Gates</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Set up quality gates for main branches</li>
              <li>• Enforce quality checks on pull requests</li>
              <li>• Block releases if quality gates fail</li>
              <li>• Regular review and adjust thresholds</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">🔧 Continuous Improvement</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Regular code reviews and pair programming</li>
              <li>• Automated code formatting and linting</li>
              <li>• Refactoring sessions to reduce technical debt</li>
              <li>• Team training on coding standards</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};