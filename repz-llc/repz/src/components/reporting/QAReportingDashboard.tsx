import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { Calendar } from '@/components/ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useToast } from '@/hooks/use-toast';
import { format } from 'date-fns';
import { cn } from '@/lib/utils';
import { 
  Calendar as CalendarIcon,
  Download,
  FileText,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  XCircle,
  AlertTriangle,
  BarChart3,
  PieChart,
  LineChart,
  Users
} from 'lucide-react';

interface QAMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  target: number;
  status: 'excellent' | 'good' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  change: number;
  category: 'testing' | 'security' | 'performance' | 'quality' | 'deployment';
}

interface QAReport {
  id: string;
  title: string;
  period: string;
  generatedAt: Date;
  status: 'generating' | 'completed' | 'failed';
  type: 'weekly' | 'monthly' | 'quarterly' | 'custom';
  metrics: QAMetric[];
  summary: {
    overallScore: number;
    totalTests: number;
    passRate: number;
    criticalIssues: number;
    securityScore: number;
    performanceScore: number;
  };
}

interface Recommendation {
  id: string;
  priority: 'high' | 'medium' | 'low';
  category: string;
  title: string;
  description: string;
  impact: string;
  effort: string;
  actionItems: string[];
}

export const QAReportingDashboard = () => {
  const [selectedPeriod, setSelectedPeriod] = useState<'week' | 'month' | 'quarter' | 'custom'>('month');
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({});
  const [reports, setReports] = useState<QAReport[]>([]);
  const [currentReport, setCurrentReport] = useState<QAReport | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();

  const mockMetrics: QAMetric[] = [
    {
      id: 'test-coverage',
      name: 'Test Coverage',
      value: 87.5,
      unit: '%',
      target: 85,
      status: 'good',
      trend: 'up',
      change: 2.3,
      category: 'testing'
    },
    {
      id: 'test-pass-rate',
      name: 'Test Pass Rate',
      value: 94.2,
      unit: '%',
      target: 95,
      status: 'warning',
      trend: 'down',
      change: -1.8,
      category: 'testing'
    },
    {
      id: 'security-score',
      name: 'Security Score',
      value: 92,
      unit: '/100',
      target: 90,
      status: 'excellent',
      trend: 'stable',
      change: 0,
      category: 'security'
    },
    {
      id: 'performance-score',
      name: 'Performance Score',
      value: 88,
      unit: '/100',
      target: 85,
      status: 'good',
      trend: 'up',
      change: 3.5,
      category: 'performance'
    },
    {
      id: 'code-quality',
      name: 'Code Quality Score',
      value: 91,
      unit: '/100',
      target: 85,
      status: 'excellent',
      trend: 'up',
      change: 1.2,
      category: 'quality'
    },
    {
      id: 'deployment-success',
      name: 'Deployment Success Rate',
      value: 97.8,
      unit: '%',
      target: 95,
      status: 'excellent',
      trend: 'stable',
      change: 0.2,
      category: 'deployment'
    }
  ];

  const mockRecommendations: Recommendation[] = [
    {
      id: 'rec-1',
      priority: 'high',
      category: 'Testing',
      title: 'Increase E2E Test Coverage',
      description: 'End-to-end test coverage is below target. Critical user journeys lack automated testing.',
      impact: 'Reduced risk of production bugs, improved user experience',
      effort: '2-3 weeks',
      actionItems: [
        'Identify critical user journeys without E2E tests',
        'Implement Playwright tests for checkout flow',
        'Add E2E tests for user registration and authentication',
        'Set up automated E2E test execution in CI/CD pipeline'
      ]
    },
    {
      id: 'rec-2',
      priority: 'medium',
      category: 'Performance',
      title: 'Optimize API Response Times',
      description: 'Several API endpoints exceed response time thresholds during peak hours.',
      impact: 'Better user experience, improved scalability',
      effort: '1-2 weeks',
      actionItems: [
        'Profile slow API endpoints',
        'Implement database query optimization',
        'Add response caching for frequently accessed data',
        'Consider implementing API rate limiting'
      ]
    },
    {
      id: 'rec-3',
      priority: 'low',
      category: 'Code Quality',
      title: 'Reduce Technical Debt',
      description: 'Accumulating technical debt in legacy components affecting maintainability.',
      impact: 'Improved code maintainability, faster development cycles',
      effort: '3-4 weeks',
      actionItems: [
        'Refactor outdated utility functions',
        'Update deprecated dependencies',
        'Improve code documentation',
        'Standardize coding patterns across codebase'
      ]
    }
  ];

  const generateReport = async (period: string) => {
    setIsGenerating(true);
    
    toast({
      title: "Generating QA Report",
      description: `Creating ${period} quality assurance report...`,
    });

    // Simulate report generation
    await new Promise(resolve => setTimeout(resolve, 3000));

    const newReport: QAReport = {
      id: `report-${Date.now()}`,
      title: `QA Report - ${period}`,
      period: period,
      generatedAt: new Date(),
      status: 'completed',
      type: period as QAReport['type'],
      metrics: mockMetrics,
      summary: {
        overallScore: 90,
        totalTests: 1247,
        passRate: 94.2,
        criticalIssues: 2,
        securityScore: 92,
        performanceScore: 88
      }
    };

    setCurrentReport(newReport);
    setReports(prev => [newReport, ...prev]);
    setIsGenerating(false);

    toast({
      title: "Report Generated Successfully",
      description: `${period} QA report is ready for review`,
    });
  };

  const exportReport = (reportId: string, format: 'pdf' | 'excel' | 'json') => {
    const report = reports.find(r => r.id === reportId) || currentReport;
    if (!report) return;

    // Simulate export
    toast({
      title: "Report Exported",
      description: `Report exported as ${format.toUpperCase()}`,
    });

    // In a real implementation, this would generate and download the file
    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `qa-report-${report.period}-${format}.${format === 'json' ? 'json' : format}`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-600 bg-green-50 border-green-200';
      case 'good': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'warning': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent':
      case 'good': 
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'warning': 
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'critical': 
        return <XCircle className="h-4 w-4 text-red-600" />;
      default: 
        return <CheckCircle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getTrendIcon = (trend: string, change: number) => {
    if (trend === 'up') {
      return <TrendingUp className={`h-4 w-4 ${change > 0 ? 'text-green-600' : 'text-red-600'}`} />;
    } else if (trend === 'down') {
      return <TrendingDown className={`h-4 w-4 ${change < 0 ? 'text-red-600' : 'text-green-600'}`} />;
    }
    return <div className="h-4 w-4 rounded-full bg-gray-400" />;
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-700 bg-red-100 border-red-200';
      case 'medium': return 'text-yellow-700 bg-yellow-100 border-yellow-200';
      case 'low': return 'text-green-700 bg-green-100 border-green-200';
      default: return 'text-gray-700 bg-gray-100 border-gray-200';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case 'testing': return <CheckCircle className="h-4 w-4" />;
      case 'performance': return <BarChart3 className="h-4 w-4" />;
      case 'code quality': return <FileText className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">QA Reporting Dashboard</h2>
        </div>
        <div className="flex gap-2">
          <Select value={selectedPeriod} onValueChange={(value) => setSelectedPeriod(value as typeof selectedPeriod)}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Select period" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="week">Weekly</SelectItem>
              <SelectItem value="month">Monthly</SelectItem>
              <SelectItem value="quarter">Quarterly</SelectItem>
              <SelectItem value="custom">Custom Range</SelectItem>
            </SelectContent>
          </Select>
          
          {selectedPeriod === 'custom' && (
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="gap-2" onClick={() => console.log("QAReportingDashboard button clicked")}>
                  <CalendarIcon className="h-4 w-4" />
                  {dateRange.from ? (
                    dateRange.to ? (
                      `${format(dateRange.from, "MMM dd")} - ${format(dateRange.to, "MMM dd")}`
                    ) : (
                      format(dateRange.from, "MMM dd, yyyy")
                    )
                  ) : (
                    "Pick date range"
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="range"
                  selected={dateRange.from && dateRange.to ? { from: dateRange.from, to: dateRange.to } : undefined}
                  onSelect={(range) => setDateRange(range || {})}
                  numberOfMonths={2}
                  className={cn("p-3 pointer-events-auto")}
                />
              </PopoverContent>
            </Popover>
          )}
          
          <Button 
            onClick={() => generateReport(selectedPeriod)}
            disabled={isGenerating}
            className="gap-2"
          >
            <FileText className="h-4 w-4" />
            {isGenerating ? 'Generating...' : 'Generate Report'}
          </Button>
        </div>
      </div>

      {/* Current Report Summary */}
      {currentReport && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{currentReport.title}</span>
              <div className="flex gap-2">
                <Button 
                  onClick={() => exportReport(currentReport.id, 'pdf')}
                  size="sm" 
                  variant="outline"
                >
                  <Download className="h-4 w-4 mr-1" />
                  PDF
                </Button>
                <Button 
                  onClick={() => exportReport(currentReport.id, 'excel')}
                  size="sm" 
                  variant="outline"
                >
                  <Download className="h-4 w-4 mr-1" />
                  Excel
                </Button>
                <Button 
                  onClick={() => exportReport(currentReport.id, 'json')}
                  size="sm" 
                  variant="outline"
                >
                  <Download className="h-4 w-4 mr-1" />
                  JSON
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-6 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {currentReport.summary.overallScore}
                </div>
                <div className="text-sm text-muted-foreground">Overall Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {currentReport.summary.totalTests}
                </div>
                <div className="text-sm text-muted-foreground">Total Tests</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {currentReport.summary.passRate}%
                </div>
                <div className="text-sm text-muted-foreground">Pass Rate</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {currentReport.summary.criticalIssues}
                </div>
                <div className="text-sm text-muted-foreground">Critical Issues</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {currentReport.summary.securityScore}
                </div>
                <div className="text-sm text-muted-foreground">Security Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {currentReport.summary.performanceScore}
                </div>
                <div className="text-sm text-muted-foreground">Performance</div>
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              Generated on {currentReport.generatedAt.toLocaleString()}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Metrics Breakdown */}
      {currentReport && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {currentReport.metrics.map((metric) => (
            <Card key={metric.id}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between text-sm">
                  <span>{metric.name}</span>
                  <div className="flex items-center gap-1">
                    {getStatusIcon(metric.status)}
                    {getTrendIcon(metric.trend, metric.change)}
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">
                    {metric.value}{metric.unit}
                  </div>
                  <Progress value={(metric.value / metric.target) * 100} className="h-2" />
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">
                      Target: {metric.target}{metric.unit}
                    </span>
                    <span className={metric.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {metric.change >= 0 ? '+' : ''}{metric.change}%
                    </span>
                  </div>
                  <Badge className={getStatusColor(metric.status)} variant="outline">
                    {metric.status.toUpperCase()}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>QA Improvement Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {mockRecommendations.map((rec) => (
              <div key={rec.id} className="p-4 border rounded">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    {getCategoryIcon(rec.category)}
                    <div>
                      <div className="font-medium">{rec.title}</div>
                      <div className="text-sm text-muted-foreground">{rec.category}</div>
                    </div>
                  </div>
                  <Badge className={getPriorityColor(rec.priority)}>
                    {rec.priority.toUpperCase()} PRIORITY
                  </Badge>
                </div>
                
                <div className="space-y-3">
                  <p className="text-sm text-muted-foreground">{rec.description}</p>
                  
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <div className="text-sm font-medium mb-1">Expected Impact</div>
                      <div className="text-sm text-muted-foreground">{rec.impact}</div>
                    </div>
                    <div>
                      <div className="text-sm font-medium mb-1">Estimated Effort</div>
                      <div className="text-sm text-muted-foreground">{rec.effort}</div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-sm font-medium mb-2">Action Items</div>
                    <ul className="space-y-1">
                      {rec.actionItems.map((item, index) => (
                        <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                          <span className="text-blue-600 mt-1">•</span>
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Reports */}
      {reports.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Reports</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {reports.slice(0, 5).map((report) => (
                <div key={report.id} className="flex items-center justify-between p-3 border rounded">
                  <div className="flex items-center gap-3">
                    <FileText className="h-4 w-4 text-blue-600" />
                    <div>
                      <div className="font-medium">{report.title}</div>
                      <div className="text-sm text-muted-foreground">
                        {report.generatedAt.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{report.type}</Badge>
                    <Button
                      onClick={() => setCurrentReport(report)}
                      size="sm"
                      variant="outline"
                    >
                      View
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Report Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Automated Reporting</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">📊 Report Types</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• <strong>Weekly Reports:</strong> Quick overview of key metrics and recent issues</li>
              <li>• <strong>Monthly Reports:</strong> Comprehensive analysis with trends and recommendations</li>
              <li>• <strong>Quarterly Reports:</strong> Strategic insights and long-term improvement plans</li>
              <li>• <strong>Custom Reports:</strong> Tailored analysis for specific date ranges or metrics</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">🔧 Configuration Options</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Set up automated report generation schedules</li>
              <li>• Configure stakeholder distribution lists</li>
              <li>• Customize metrics inclusion and thresholds</li>
              <li>• Define quality gates and success criteria</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">📈 Actionable Insights</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• AI-powered recommendations for quality improvements</li>
              <li>• Trend analysis and predictive quality forecasting</li>
              <li>• Risk assessment and mitigation strategies</li>
              <li>• ROI analysis for quality improvement initiatives</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};