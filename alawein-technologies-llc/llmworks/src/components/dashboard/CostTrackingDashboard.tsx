import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Calendar } from '@/components/ui/calendar';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Calendar as CalendarIcon,
  Download,
  AlertTriangle,
  Settings,
  Target,
  CreditCard,
  BarChart3,
  PieChart,
  Zap,
  Clock,
  Brain,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  RefreshCw,
} from 'lucide-react';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { LineChart, Line, XAxis, YAxis, BarChart, Bar, ResponsiveContainer, PieChart as RechartsPieChart, Pie, Cell } from 'recharts';
import { trackEvent } from '@/lib/analytics';
import { format } from 'date-fns';

interface CostData {
  date: string;
  totalCost: number;
  apiCosts: {
    openai: number;
    anthropic: number;
    google: number;
    meta: number;
  };
  usage: {
    totalTokens: number;
    evaluations: number;
    avgCostPerEvaluation: number;
  };
}

interface ModelUsage {
  modelId: string;
  modelName: string;
  provider: string;
  usage: {
    totalTokens: number;
    inputTokens: number;
    outputTokens: number;
    requests: number;
  };
  cost: {
    total: number;
    inputCost: number;
    outputCost: number;
    perRequest: number;
  };
  trend: {
    change: number;
    direction: 'up' | 'down' | 'stable';
  };
}

interface Budget {
  id: string;
  name: string;
  limit: number;
  spent: number;
  period: 'daily' | 'weekly' | 'monthly';
  alerts: {
    percentage: number;
    enabled: boolean;
  }[];
}

const mockCostData: CostData[] = [
  {
    date: '2024-01-01',
    totalCost: 45.67,
    apiCosts: { openai: 25.30, anthropic: 12.45, google: 5.67, meta: 2.25 },
    usage: { totalTokens: 125000, evaluations: 15, avgCostPerEvaluation: 3.04 },
  },
  {
    date: '2024-01-02',
    totalCost: 52.34,
    apiCosts: { openai: 28.90, anthropic: 15.22, google: 6.12, meta: 2.10 },
    usage: { totalTokens: 142000, evaluations: 18, avgCostPerEvaluation: 2.91 },
  },
  {
    date: '2024-01-03',
    totalCost: 38.91,
    apiCosts: { openai: 21.45, anthropic: 10.33, google: 4.88, meta: 2.25 },
    usage: { totalTokens: 98000, evaluations: 12, avgCostPerEvaluation: 3.24 },
  },
  {
    date: '2024-01-04',
    totalCost: 67.23,
    apiCosts: { openai: 35.67, anthropic: 18.90, google: 8.45, meta: 4.21 },
    usage: { totalTokens: 178000, evaluations: 22, avgCostPerEvaluation: 3.05 },
  },
  {
    date: '2024-01-05',
    totalCost: 78.45,
    apiCosts: { openai: 42.30, anthropic: 21.15, google: 10.25, meta: 4.75 },
    usage: { totalTokens: 195000, evaluations: 25, avgCostPerEvaluation: 3.14 },
  },
];

const mockModelUsage: ModelUsage[] = [
  {
    modelId: 'gpt-4',
    modelName: 'GPT-4',
    provider: 'OpenAI',
    usage: {
      totalTokens: 125000,
      inputTokens: 75000,
      outputTokens: 50000,
      requests: 245,
    },
    cost: {
      total: 156.75,
      inputCost: 56.25,
      outputCost: 100.50,
      perRequest: 0.64,
    },
    trend: {
      change: 12.5,
      direction: 'up',
    },
  },
  {
    modelId: 'claude-3-sonnet',
    modelName: 'Claude 3.5 Sonnet',
    provider: 'Anthropic',
    usage: {
      totalTokens: 98000,
      inputTokens: 58000,
      outputTokens: 40000,
      requests: 189,
    },
    cost: {
      total: 89.45,
      inputCost: 34.80,
      outputCost: 54.65,
      perRequest: 0.47,
    },
    trend: {
      change: 8.3,
      direction: 'up',
    },
  },
  {
    modelId: 'gemini-pro',
    modelName: 'Gemini 1.5 Pro',
    provider: 'Google',
    usage: {
      totalTokens: 67000,
      inputTokens: 42000,
      outputTokens: 25000,
      requests: 134,
    },
    cost: {
      total: 23.45,
      inputCost: 12.60,
      outputCost: 10.85,
      perRequest: 0.18,
    },
    trend: {
      change: -5.2,
      direction: 'down',
    },
  },
  {
    modelId: 'llama-70b',
    modelName: 'Llama 3.1 70B',
    provider: 'Meta',
    usage: {
      totalTokens: 45000,
      inputTokens: 30000,
      outputTokens: 15000,
      requests: 78,
    },
    cost: {
      total: 8.90,
      inputCost: 6.00,
      outputCost: 2.90,
      perRequest: 0.11,
    },
    trend: {
      change: -12.8,
      direction: 'down',
    },
  },
];

const mockBudgets: Budget[] = [
  {
    id: 'monthly-main',
    name: 'Monthly Evaluation Budget',
    limit: 1000,
    spent: 678.45,
    period: 'monthly',
    alerts: [
      { percentage: 50, enabled: true },
      { percentage: 75, enabled: true },
      { percentage: 90, enabled: true },
    ],
  },
  {
    id: 'weekly-testing',
    name: 'Weekly Testing Budget',
    limit: 250,
    spent: 189.23,
    period: 'weekly',
    alerts: [
      { percentage: 80, enabled: true },
      { percentage: 95, enabled: true },
    ],
  },
];

const chartConfig = {
  totalCost: { label: 'Total Cost', color: 'hsl(var(--primary))' },
  openai: { label: 'OpenAI', color: '#10a37f' },
  anthropic: { label: 'Anthropic', color: '#d97706' },
  google: { label: 'Google', color: '#2563eb' },
  meta: { label: 'Meta', color: '#1e40af' },
  evaluations: { label: 'Evaluations', color: 'hsl(var(--accent))' },
  tokens: { label: 'Tokens', color: 'hsl(var(--secondary))' },
};

export const CostTrackingDashboard = () => {
  const [costData, setCostData] = useState<CostData[]>(mockCostData);
  const [modelUsage, setModelUsage] = useState<ModelUsage[]>(mockModelUsage);
  const [budgets, setBudgets] = useState<Budget[]>(mockBudgets);
  const [selectedPeriod, setSelectedPeriod] = useState<'7d' | '30d' | '90d'>('30d');
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({});

  const totalSpent = costData.reduce((sum, day) => sum + day.totalCost, 0);
  const avgDailySpend = totalSpent / costData.length;
  const totalTokens = costData.reduce((sum, day) => sum + day.usage.totalTokens, 0);
  const totalEvaluations = costData.reduce((sum, day) => sum + day.usage.evaluations, 0);

  // Calculate provider breakdown for pie chart
  const providerBreakdown = [
    { name: 'OpenAI', value: costData.reduce((sum, day) => sum + day.apiCosts.openai, 0), color: '#10a37f' },
    { name: 'Anthropic', value: costData.reduce((sum, day) => sum + day.apiCosts.anthropic, 0), color: '#d97706' },
    { name: 'Google', value: costData.reduce((sum, day) => sum + day.apiCosts.google, 0), color: '#2563eb' },
    { name: 'Meta', value: costData.reduce((sum, day) => sum + day.apiCosts.meta, 0), color: '#1e40af' },
  ];

  const handleExportData = () => {
    trackEvent('cost_data_export', { period: selectedPeriod });
    // Export logic would go here
  };

  const getBudgetStatus = (budget: Budget) => {
    const percentage = (budget.spent / budget.limit) * 100;
    if (percentage >= 90) return { color: 'red', status: 'danger' };
    if (percentage >= 75) return { color: 'yellow', status: 'warning' };
    return { color: 'green', status: 'good' };
  };

  const getTrendIcon = (direction: ModelUsage['trend']['direction']) => {
    switch (direction) {
      case 'up':
        return <ArrowUpRight className="h-4 w-4 text-red-500" />;
      case 'down':
        return <ArrowDownRight className="h-4 w-4 text-green-500" />;
      case 'stable':
        return <div className="h-4 w-4" />;
    }
  };

  const getTrendColor = (direction: ModelUsage['trend']['direction']) => {
    switch (direction) {
      case 'up':
        return 'text-red-600';
      case 'down':
        return 'text-green-600';
      case 'stable':
        return 'text-muted-foreground';
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Spent</p>
              <p className="text-2xl font-bold text-foreground">${totalSpent.toFixed(2)}</p>
              <div className="flex items-center gap-1 mt-1">
                <TrendingUp className="h-3 w-3 text-red-500" />
                <span className="text-xs text-red-600">+15.2% this period</span>
              </div>
            </div>
            <DollarSign className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Daily Average</p>
              <p className="text-2xl font-bold text-foreground">${avgDailySpend.toFixed(2)}</p>
              <div className="flex items-center gap-1 mt-1">
                <TrendingDown className="h-3 w-3 text-green-500" />
                <span className="text-xs text-green-600">-3.8% vs last period</span>
              </div>
            </div>
            <BarChart3 className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Tokens</p>
              <p className="text-2xl font-bold text-foreground">{(totalTokens / 1000).toFixed(0)}K</p>
              <div className="flex items-center gap-1 mt-1">
                <Zap className="h-3 w-3 text-primary" />
                <span className="text-xs text-muted-foreground">{totalEvaluations} evaluations</span>
              </div>
            </div>
            <Brain className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Cost per Evaluation</p>
              <p className="text-2xl font-bold text-foreground">
                ${(totalSpent / totalEvaluations).toFixed(2)}
              </p>
              <div className="flex items-center gap-1 mt-1">
                <Target className="h-3 w-3 text-green-500" />
                <span className="text-xs text-green-600">-8.1% efficiency gain</span>
              </div>
            </div>
            <CreditCard className="h-8 w-8 text-primary" />
          </div>
        </Card>
      </div>

      {/* Budget Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {budgets.map((budget) => {
          const status = getBudgetStatus(budget);
          const percentage = (budget.spent / budget.limit) * 100;
          
          return (
            <Card key={budget.id} className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="font-medium text-foreground">{budget.name}</h3>
                  <p className="text-sm text-muted-foreground capitalize">
                    {budget.period} • ${budget.spent.toFixed(2)} of ${budget.limit.toFixed(2)}
                  </p>
                </div>
                {status.status === 'danger' && (
                  <AlertTriangle className="h-5 w-5 text-red-500" />
                )}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Usage</span>
                  <span className="font-medium">{percentage.toFixed(1)}%</span>
                </div>
                <Progress 
                  value={percentage} 
                  className="h-2"
                  // Add color based on status
                />
              </div>
            </Card>
          );
        })}
      </div>

      {/* Controls */}
      <Card className="p-4">
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
          <div className="flex gap-4">
            <Select value={selectedPeriod} onValueChange={(value: any) => setSelectedPeriod(value)}>
              <SelectTrigger className="w-40">
                <CalendarIcon className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="7d">Last 7 days</SelectItem>
                <SelectItem value="30d">Last 30 days</SelectItem>
                <SelectItem value="90d">Last 90 days</SelectItem>
              </SelectContent>
            </Select>

            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline">
                  <Filter className="h-4 w-4 mr-2" />
                  Custom Range
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  initialFocus
                  mode="range"
                  selected={dateRange as any}
                  onSelect={setDateRange}
                  numberOfMonths={2}
                />
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleExportData}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Budget Settings
            </Button>
            <Button variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>
      </Card>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost Trend */}
        <Card className="p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Cost Trend</h3>
          <ChartContainer config={chartConfig} className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={costData}>
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => format(new Date(value), 'MMM dd')}
                />
                <YAxis tickFormatter={(value) => `$${value}`} />
                <ChartTooltip 
                  content={<ChartTooltipContent />}
                  labelFormatter={(value) => format(new Date(value), 'MMM dd, yyyy')}
                />
                <Line 
                  type="monotone" 
                  dataKey="totalCost" 
                  stroke="var(--color-totalCost)" 
                  strokeWidth={2}
                  dot={{ fill: 'var(--color-totalCost)' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>

        {/* Provider Breakdown */}
        <Card className="p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Cost by Provider</h3>
          <div className="flex items-center justify-center">
            <ChartContainer config={chartConfig} className="h-[300px] w-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPieChart>
                  <Pie
                    data={providerBreakdown}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {providerBreakdown.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <ChartTooltip 
                    content={<ChartTooltipContent />}
                    formatter={(value) => [`$${Number(value).toFixed(2)}`, 'Cost']}
                  />
                </RechartsPieChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </Card>
      </div>

      {/* Model Usage Table */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-foreground">Model Usage & Costs</h3>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Details
          </Button>
        </div>
        
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Tokens</TableHead>
                <TableHead>Requests</TableHead>
                <TableHead>Total Cost</TableHead>
                <TableHead>Cost/Request</TableHead>
                <TableHead>Trend</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {modelUsage.map((model) => (
                <TableRow key={model.modelId}>
                  <TableCell>
                    <div>
                      <div className="font-medium">{model.modelName}</div>
                      <div className="text-sm text-muted-foreground">{model.provider}</div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div>
                      <div className="font-medium">{model.usage.totalTokens.toLocaleString()}</div>
                      <div className="text-xs text-muted-foreground">
                        In: {model.usage.inputTokens.toLocaleString()} • 
                        Out: {model.usage.outputTokens.toLocaleString()}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">
                    {model.usage.requests.toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <div>
                      <div className="font-medium">${model.cost.total.toFixed(2)}</div>
                      <div className="text-xs text-muted-foreground">
                        In: ${model.cost.inputCost.toFixed(2)} • 
                        Out: ${model.cost.outputCost.toFixed(2)}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">
                    ${model.cost.perRequest.toFixed(3)}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      {getTrendIcon(model.trend.direction)}
                      <span className={`text-sm font-medium ${getTrendColor(model.trend.direction)}`}>
                        {Math.abs(model.trend.change).toFixed(1)}%
                      </span>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Cost Optimization Recommendations */}
      <Card className="p-6">
        <h3 className="text-lg font-bold text-foreground mb-4">Cost Optimization Recommendations</h3>
        <div className="space-y-4">
          <div className="flex items-start gap-3 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-foreground">High OpenAI Usage</h4>
              <p className="text-sm text-muted-foreground mt-1">
                OpenAI accounts for 65% of your costs. Consider using Claude 3.5 Sonnet for similar performance at lower cost.
              </p>
            </div>
          </div>
          
          <div className="flex items-start gap-3 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <Target className="h-5 w-5 text-green-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-foreground">Optimization Opportunity</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Your cost per evaluation decreased 8.1% this period. Consider batching similar evaluations to maintain this efficiency.
              </p>
            </div>
          </div>
          
          <div className="flex items-start gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <Zap className="h-5 w-5 text-blue-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-foreground">Token Optimization</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Reduce prompt length by 15-20% to save on input token costs without affecting evaluation quality.
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};