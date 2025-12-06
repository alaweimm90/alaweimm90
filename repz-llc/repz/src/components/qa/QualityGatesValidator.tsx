import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  Shield, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  Play,
  RefreshCw,
  Settings,
  Target,
  Clock,
  GitBranch,
  Lock,
  Zap
} from 'lucide-react';

interface QualityGate {
  id: string;
  name: string;
  description: string;
  category: 'testing' | 'performance' | 'security' | 'coverage' | 'deployment';
  status: 'passed' | 'failed' | 'warning' | 'pending';
  threshold: number;
  currentValue: number;
  unit: string;
  required: boolean;
  lastChecked: string;
  duration: number;
}

interface GateCategory {
  name: string;
  icon: React.ReactNode;
  gates: QualityGate[];
  status: 'passed' | 'failed' | 'warning';
}

export const QualityGatesValidator = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const qualityGates: QualityGate[] = useMemo(() => [
    {
      id: 'test-coverage',
      name: 'Code Coverage',
      description: 'Minimum code coverage percentage required',
      category: 'coverage',
      status: 'passed',
      threshold: 80,
      currentValue: 87,
      unit: '%',
      required: true,
      lastChecked: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
      duration: 45
    },
    {
      id: 'test-success-rate',
      name: 'Test Success Rate',
      description: 'Percentage of tests that must pass',
      category: 'testing',
      status: 'failed',
      threshold: 95,
      currentValue: 89,
      unit: '%',
      required: true,
      lastChecked: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
      duration: 120
    },
    {
      id: 'performance-budget',
      name: 'Performance Budget',
      description: 'Page load time must be under threshold',
      category: 'performance',
      status: 'warning',
      threshold: 2000,
      currentValue: 2150,
      unit: 'ms',
      required: false,
      lastChecked: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
      duration: 30
    },
    {
      id: 'security-vulnerabilities',
      name: 'Security Vulnerabilities',
      description: 'No critical or high severity vulnerabilities',
      category: 'security',
      status: 'passed',
      threshold: 0,
      currentValue: 0,
      unit: 'issues',
      required: true,
      lastChecked: new Date(Date.now() - 1 * 60 * 1000).toISOString(),
      duration: 180
    },
    {
      id: 'build-success',
      name: 'Build Success',
      description: 'Build must complete successfully',
      category: 'deployment',
      status: 'passed',
      threshold: 1,
      currentValue: 1,
      unit: 'status',
      required: true,
      lastChecked: new Date(Date.now() - 3 * 60 * 1000).toISOString(),
      duration: 60
    },
    {
      id: 'bundle-size',
      name: 'Bundle Size',
      description: 'Bundle size must be under limit',
      category: 'performance',
      status: 'passed',
      threshold: 500,
      currentValue: 423,
      unit: 'KB',
      required: false,
      lastChecked: new Date(Date.now() - 8 * 60 * 1000).toISOString(),
      duration: 15
    },
    {
      id: 'lint-violations',
      name: 'Lint Violations',
      description: 'No critical linting errors allowed',
      category: 'testing',
      status: 'warning',
      threshold: 5,
      currentValue: 7,
      unit: 'errors',
      required: false,
      lastChecked: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
      duration: 25
    },
    {
      id: 'dependency-check',
      name: 'Dependency Security',
      description: 'No vulnerable dependencies',
      category: 'security',
      status: 'passed',
      threshold: 0,
      currentValue: 0,
      unit: 'vulnerabilities',
      required: true,
      lastChecked: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
      duration: 90
    }
  ], []);

  const gateCategories: GateCategory[] = useMemo(() => {
    const categories = ['testing', 'performance', 'security', 'coverage', 'deployment'];
    
    return categories.map(category => {
      const categoryGates = qualityGates.filter(gate => gate.category === category);
      const failedGates = categoryGates.filter(gate => gate.status === 'failed');
      const warningGates = categoryGates.filter(gate => gate.status === 'warning');
      
      let status: 'passed' | 'failed' | 'warning' = 'passed';
      if (failedGates.length > 0) status = 'failed';
      else if (warningGates.length > 0) status = 'warning';

      const icons = {
        testing: <CheckCircle className="h-4 w-4" />,
        performance: <Zap className="h-4 w-4" />,
        security: <Shield className="h-4 w-4" />,
        coverage: <Target className="h-4 w-4" />,
        deployment: <GitBranch className="h-4 w-4" />
      };

      return {
        name: category.charAt(0).toUpperCase() + category.slice(1),
        icon: icons[category as keyof typeof icons],
        gates: categoryGates,
        status
      };
    });
  }, [qualityGates]);

  const overallStatus = useMemo(() => {
    const requiredGates = qualityGates.filter(gate => gate.required);
    const failedRequired = requiredGates.filter(gate => gate.status === 'failed');
    const allFailed = qualityGates.filter(gate => gate.status === 'failed');
    const allWarnings = qualityGates.filter(gate => gate.status === 'warning');

    if (failedRequired.length > 0) return 'blocked';
    if (allFailed.length > 0) return 'failed';
    if (allWarnings.length > 0) return 'warning';
    return 'passed';
  }, [qualityGates]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'pending': return <Clock className="h-4 w-4 text-gray-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed': return 'text-green-600';
      case 'failed': return 'text-red-600';
      case 'warning': return 'text-yellow-600';
      case 'pending': return 'text-gray-600';
      default: return 'text-gray-600';
    }
  };

  const runQualityGates = async () => {
    setIsRunning(true);
    // Simulate running quality gates
    await new Promise(resolve => setTimeout(resolve, 3000));
    setIsRunning(false);
  };

  const filteredGates = selectedCategory === 'all' 
    ? qualityGates 
    : qualityGates.filter(gate => gate.category === selectedCategory);

  const formatValue = (value: number, unit: string) => {
    if (unit === 'status') return value === 1 ? 'Success' : 'Failed';
    return `${value}${unit}`;
  };

  const calculateProgress = (gate: QualityGate) => {
    if (gate.unit === 'status') return gate.currentValue === 1 ? 100 : 0;
    if (gate.unit === 'issues' || gate.unit === 'errors' || gate.unit === 'vulnerabilities') {
      return Math.max(0, 100 - (gate.currentValue / Math.max(gate.threshold, 1)) * 100);
    }
    if (gate.unit === 'ms') {
      return Math.max(0, 100 - (gate.currentValue / gate.threshold) * 100);
    }
    return Math.min(100, (gate.currentValue / gate.threshold) * 100);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Quality Gates Validator</h2>
          <p className="text-muted-foreground">Validate quality gates before deployment</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => console.log("QualityGatesValidator button clicked")}>
            <Settings className="h-4 w-4 mr-2" />
            Configure
          </Button>
          <Button 
            onClick={runQualityGates} 
            disabled={isRunning}
            size="sm"
          >
            {isRunning ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            {isRunning ? 'Running...' : 'Run Gates'}
          </Button>
        </div>
      </div>

      {/* Overall Status */}
      <Alert className={
        overallStatus === 'passed' ? 'border-green-200 bg-green-50' :
        overallStatus === 'blocked' ? 'border-red-200 bg-red-50' :
        overallStatus === 'failed' ? 'border-red-200 bg-red-50' :
        'border-yellow-200 bg-yellow-50'
      }>
        <div className="flex items-center gap-2">
          {overallStatus === 'passed' && <CheckCircle className="h-4 w-4 text-green-600" />}
          {overallStatus === 'blocked' && <Lock className="h-4 w-4 text-red-600" />}
          {overallStatus === 'failed' && <XCircle className="h-4 w-4 text-red-600" />}
          {overallStatus === 'warning' && <AlertTriangle className="h-4 w-4 text-yellow-600" />}
          <strong>
            {overallStatus === 'passed' && 'Quality Gates Passed'}
            {overallStatus === 'blocked' && 'Deployment Blocked'}
            {overallStatus === 'failed' && 'Quality Gates Failed'}
            {overallStatus === 'warning' && 'Quality Gates Warning'}
          </strong>
        </div>
        <AlertDescription className="mt-2">
          {overallStatus === 'passed' && 'All required quality gates have passed. Deployment is approved.'}
          {overallStatus === 'blocked' && 'Required quality gates have failed. Deployment is blocked until issues are resolved.'}
          {overallStatus === 'failed' && 'Some quality gates have failed. Review and fix issues before deployment.'}
          {overallStatus === 'warning' && 'Some quality gates have warnings. Consider addressing before deployment.'}
        </AlertDescription>
      </Alert>

      {/* Category Overview */}
      <div className="grid gap-4 md:grid-cols-5">
        {gateCategories.map((category) => (
          <Card 
            key={category.name}
            className={`cursor-pointer transition-colors ${
              selectedCategory === category.name.toLowerCase() ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setSelectedCategory(category.name.toLowerCase())}
          >
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {category.icon}
                  <span className="font-medium">{category.name}</span>
                </div>
                {getStatusIcon(category.status)}
              </div>
              <div className="mt-2">
                <div className="text-sm text-muted-foreground">
                  {category.gates.length} gates
                </div>
                <Progress 
                  value={(category.gates.filter(g => g.status === 'passed').length / category.gates.length) * 100}
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Tabs defaultValue="gates" className="w-full">
        <TabsList>
          <TabsTrigger value="gates">Quality Gates</TabsTrigger>
          <TabsTrigger value="history">Execution History</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
        </TabsList>

        <TabsContent value="gates">
          <div className="flex justify-between items-center mb-4">
            <div className="flex gap-2">
              <Button
                variant={selectedCategory === 'all' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCategory('all')}
              >
                All Gates
              </Button>
              {gateCategories.map((category) => (
                <Button
                  key={category.name}
                  variant={selectedCategory === category.name.toLowerCase() ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedCategory(category.name.toLowerCase())}
                >
                  {category.name}
                </Button>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            {filteredGates.map((gate) => (
              <Card key={gate.id}>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(gate.status)}
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold">{gate.name}</h3>
                          {gate.required && (
                            <Badge variant="destructive" className="text-xs">Required</Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">{gate.description}</p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getStatusColor(gate.status)}`}>
                        {formatValue(gate.currentValue, gate.unit)}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Target: {formatValue(gate.threshold, gate.unit)}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <Progress value={calculateProgress(gate)} className="h-2" />
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {new Date(gate.lastChecked).toLocaleTimeString()} ({gate.duration}s)
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Execution History</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { timestamp: new Date(Date.now() - 5 * 60 * 1000), status: 'passed', duration: 245 },
                  { timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), status: 'failed', duration: 198 },
                  { timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000), status: 'warning', duration: 267 },
                  { timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000), status: 'passed', duration: 234 }
                ].map((execution, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(execution.status)}
                      <div>
                        <p className="font-medium">Quality Gate Execution</p>
                        <p className="text-sm text-muted-foreground">
                          {execution.timestamp.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Duration: {execution.duration}s
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="config">
          <Card>
            <CardHeader>
              <CardTitle>Quality Gate Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert>
                  <Settings className="h-4 w-4" />
                  <AlertDescription>
                    Quality gate thresholds can be configured per environment. Contact your administrator to modify these settings.
                  </AlertDescription>
                </Alert>
                
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <h4 className="font-semibold mb-2">Required Gates</h4>
                    <div className="space-y-2">
                      {qualityGates.filter(gate => gate.required).map((gate) => (
                        <div key={gate.id} className="flex items-center justify-between p-2 border rounded">
                          <span className="text-sm">{gate.name}</span>
                          <Badge variant="destructive" className="text-xs">Required</Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2">Optional Gates</h4>
                    <div className="space-y-2">
                      {qualityGates.filter(gate => !gate.required).map((gate) => (
                        <div key={gate.id} className="flex items-center justify-between p-2 border rounded">
                          <span className="text-sm">{gate.name}</span>
                          <Badge variant="outline" className="text-xs">Optional</Badge>
                        </div>
                      ))}
                    </div>
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