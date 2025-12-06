import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Brain,
  Settings,
  Play,
  Pause,
  Trash2,
  Edit,
  MoreVertical,
  Plus,
  Search,
  Filter,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Zap,
  Activity,
  Cpu,
  Database,
} from 'lucide-react';
import { trackEvent } from '@/lib/analytics';

interface Model {
  id: string;
  name: string;
  provider: 'openai' | 'anthropic' | 'google' | 'meta' | 'custom';
  version: string;
  status: 'active' | 'paused' | 'error' | 'offline';
  performance: {
    accuracy: number;
    latency: number;
    throughput: number;
    cost: number;
  };
  usage: {
    totalRequests: number;
    successRate: number;
    avgTokensPerRequest: number;
    monthlySpend: number;
  };
  limits: {
    rateLimit: number;
    dailyQuota: number;
    usedQuota: number;
  };
  lastActive: string;
  configuration: {
    temperature: number;
    maxTokens: number;
    topP: number;
  };
}

const mockModels: Model[] = [
  {
    id: 'gpt-4',
    name: 'GPT-4',
    provider: 'openai',
    version: 'gpt-4-0125-preview',
    status: 'active',
    performance: {
      accuracy: 92.3,
      latency: 2.1,
      throughput: 45,
      cost: 0.03,
    },
    usage: {
      totalRequests: 1247,
      successRate: 98.2,
      avgTokensPerRequest: 523,
      monthlySpend: 234.56,
    },
    limits: {
      rateLimit: 10000,
      dailyQuota: 50000,
      usedQuota: 12340,
    },
    lastActive: '2 minutes ago',
    configuration: {
      temperature: 0.7,
      maxTokens: 4096,
      topP: 0.9,
    },
  },
  {
    id: 'claude-3-sonnet',
    name: 'Claude 3.5 Sonnet',
    provider: 'anthropic',
    version: 'claude-3-5-sonnet-20241022',
    status: 'active',
    performance: {
      accuracy: 89.7,
      latency: 1.8,
      throughput: 52,
      cost: 0.015,
    },
    usage: {
      totalRequests: 892,
      successRate: 99.1,
      avgTokensPerRequest: 612,
      monthlySpend: 187.23,
    },
    limits: {
      rateLimit: 8000,
      dailyQuota: 40000,
      usedQuota: 8921,
    },
    lastActive: '5 minutes ago',
    configuration: {
      temperature: 0.7,
      maxTokens: 8192,
      topP: 0.9,
    },
  },
  {
    id: 'gemini-pro',
    name: 'Gemini 1.5 Pro',
    provider: 'google',
    version: 'gemini-1.5-pro',
    status: 'paused',
    performance: {
      accuracy: 87.4,
      latency: 3.2,
      throughput: 38,
      cost: 0.005,
    },
    usage: {
      totalRequests: 643,
      successRate: 96.8,
      avgTokensPerRequest: 445,
      monthlySpend: 78.45,
    },
    limits: {
      rateLimit: 5000,
      dailyQuota: 30000,
      usedQuota: 6432,
    },
    lastActive: '2 hours ago',
    configuration: {
      temperature: 0.5,
      maxTokens: 2048,
      topP: 0.8,
    },
  },
  {
    id: 'llama-70b',
    name: 'Llama 3.1 70B',
    provider: 'meta',
    version: 'meta-llama/Llama-3.1-70B-Instruct',
    status: 'error',
    performance: {
      accuracy: 84.2,
      latency: 4.1,
      throughput: 28,
      cost: 0.001,
    },
    usage: {
      totalRequests: 234,
      successRate: 87.2,
      avgTokensPerRequest: 389,
      monthlySpend: 12.67,
    },
    limits: {
      rateLimit: 2000,
      dailyQuota: 15000,
      usedQuota: 2341,
    },
    lastActive: '1 day ago',
    configuration: {
      temperature: 0.6,
      maxTokens: 1024,
      topP: 0.7,
    },
  },
];

export const ModelManagementDashboard = () => {
  const [models, setModels] = useState<Model[]>(mockModels);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [providerFilter, setProviderFilter] = useState<string>('all');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);

  const filteredModels = models.filter((model) => {
    const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || model.status === statusFilter;
    const matchesProvider = providerFilter === 'all' || model.provider === providerFilter;
    return matchesSearch && matchesStatus && matchesProvider;
  });

  const getStatusIcon = (status: Model['status']) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'paused':
        return <Pause className="h-4 w-4 text-yellow-500" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'offline':
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: Model['status']) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'error':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
      case 'offline':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400';
    }
  };

  const getProviderLogo = (provider: Model['provider']) => {
    const iconClasses = 'h-6 w-6';
    switch (provider) {
      case 'openai':
        return <Brain className={`${iconClasses} text-green-600`} />;
      case 'anthropic':
        return <Brain className={`${iconClasses} text-orange-600`} />;
      case 'google':
        return <Brain className={`${iconClasses} text-blue-600`} />;
      case 'meta':
        return <Brain className={`${iconClasses} text-blue-800`} />;
      case 'custom':
        return <Brain className={`${iconClasses} text-purple-600`} />;
    }
  };

  const handleModelAction = (action: string, model: Model) => {
    trackEvent('model_action', { action, modelId: model.id });
    // Implementation would update model status, etc.
    console.log(`Action: ${action} on model: ${model.name}`);
  };

  const totalActiveModels = models.filter(m => m.status === 'active').length;
  const totalMonthlySpend = models.reduce((sum, model) => sum + model.usage.monthlySpend, 0);
  const avgSuccessRate = models.reduce((sum, model) => sum + model.usage.successRate, 0) / models.length;
  const avgAccuracy = models.reduce((sum, model) => sum + model.performance.accuracy, 0) / models.length;

  return (
    <div className="space-y-6">
      {/* Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Active Models</p>
              <p className="text-2xl font-bold text-foreground">{totalActiveModels}</p>
              <p className="text-xs text-green-600">
                <TrendingUp className="inline h-3 w-3" /> +2 this week
              </p>
            </div>
            <Activity className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Monthly Spend</p>
              <p className="text-2xl font-bold text-foreground">${totalMonthlySpend.toFixed(2)}</p>
              <p className="text-xs text-red-600">
                <TrendingUp className="inline h-3 w-3" /> +$23.45 vs last month
              </p>
            </div>
            <DollarSign className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Avg Success Rate</p>
              <p className="text-2xl font-bold text-foreground">{avgSuccessRate.toFixed(1)}%</p>
              <p className="text-xs text-green-600">
                <TrendingUp className="inline h-3 w-3" /> +1.2% this week
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Avg Accuracy</p>
              <p className="text-2xl font-bold text-foreground">{avgAccuracy.toFixed(1)}%</p>
              <p className="text-xs text-green-600">
                <TrendingUp className="inline h-3 w-3" /> +0.8% this week
              </p>
            </div>
            <Zap className="h-8 w-8 text-primary" />
          </div>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card className="p-4">
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search models..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-64"
              />
            </div>

            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-36">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="paused">Paused</SelectItem>
                <SelectItem value="error">Error</SelectItem>
                <SelectItem value="offline">Offline</SelectItem>
              </SelectContent>
            </Select>

            <Select value={providerFilter} onValueChange={setProviderFilter}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Provider" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Providers</SelectItem>
                <SelectItem value="openai">OpenAI</SelectItem>
                <SelectItem value="anthropic">Anthropic</SelectItem>
                <SelectItem value="google">Google</SelectItem>
                <SelectItem value="meta">Meta</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Add Model
          </Button>
        </div>
      </Card>

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {filteredModels.map((model) => (
          <Card key={model.id} className="p-6">
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  {getProviderLogo(model.provider)}
                  <div>
                    <h3 className="font-semibold text-foreground">{model.name}</h3>
                    <p className="text-sm text-muted-foreground">{model.version}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusColor(model.status)}>
                    {getStatusIcon(model.status)}
                    {model.status}
                  </Badge>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => handleModelAction('configure', model)}>
                        <Settings className="h-4 w-4 mr-2" />
                        Configure
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleModelAction('test', model)}>
                        <Play className="h-4 w-4 mr-2" />
                        Test Connection
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleModelAction('pause', model)}>
                        <Pause className="h-4 w-4 mr-2" />
                        Pause
                      </DropdownMenuItem>
                      <DropdownMenuItem 
                        onClick={() => handleModelAction('delete', model)}
                        className="text-red-600"
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Accuracy</span>
                    <span className="font-medium">{model.performance.accuracy}%</span>
                  </div>
                  <Progress value={model.performance.accuracy} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Success Rate</span>
                    <span className="font-medium">{model.usage.successRate}%</span>
                  </div>
                  <Progress value={model.usage.successRate} className="h-2" />
                </div>
              </div>

              {/* Usage Stats */}
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Requests</p>
                  <p className="font-medium">{model.usage.totalRequests.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Latency</p>
                  <p className="font-medium">{model.performance.latency}s</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Cost/req</p>
                  <p className="font-medium">${model.performance.cost}</p>
                </div>
              </div>

              {/* Quota Usage */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Daily Quota Usage</span>
                  <span className="font-medium">
                    {model.limits.usedQuota.toLocaleString()} / {model.limits.dailyQuota.toLocaleString()}
                  </span>
                </div>
                <Progress 
                  value={(model.limits.usedQuota / model.limits.dailyQuota) * 100} 
                  className="h-2" 
                />
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <span className="text-xs text-muted-foreground">
                  Last active: {model.lastActive}
                </span>
                <span className="text-xs font-medium text-foreground">
                  ${model.usage.monthlySpend.toFixed(2)}/month
                </span>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {filteredModels.length === 0 && (
        <div className="text-center py-12">
          <Brain className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">No models found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery || statusFilter !== 'all' || providerFilter !== 'all'
              ? 'Try adjusting your filters or search query.'
              : 'Get started by adding your first model.'}
          </p>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Add Model
          </Button>
        </div>
      )}
    </div>
  );
};