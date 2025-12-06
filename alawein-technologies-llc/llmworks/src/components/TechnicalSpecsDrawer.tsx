import { memo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  FileText, 
  Cpu, 
  Database, 
  Network, 
  Shield, 
  Zap, 
  Code, 
  Layers,
  HardDrive, 
  Monitor,
  Server,
  Cloud,
  Lock,
  Settings,
  Activity,
  BarChart3,
  CheckCircle,
  AlertTriangle,
  Clock,
  Download,
  Copy,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Info,
  Maximize2,
  Minimize2
} from "lucide-react";

interface TechnicalSpec {
  category: string;
  specs: {
    label: string;
    value: string;
    description?: string;
    status?: 'optimal' | 'warning' | 'critical';
  }[];
}

interface APIEndpoint {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  parameters?: string[];
  response: string;
}

interface Architecture {
  layer: string;
  components: {
    name: string;
    technology: string;
    status: 'active' | 'inactive' | 'maintenance';
    load: number;
    description: string;
  }[];
}

const TechnicalSpecsDrawerComponent = () => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeSpec, setActiveSpec] = useState('system');
  const [expandedSections, setExpandedSections] = useState<string[]>(['architecture']);

  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  };

  const technicalSpecs: TechnicalSpec[] = [
    {
      category: 'System Architecture',
      specs: [
        { label: 'Platform', value: 'Kubernetes + Docker', description: 'Container orchestration platform', status: 'optimal' },
        { label: 'Load Balancer', value: 'NGINX Ingress', description: 'High-performance reverse proxy', status: 'optimal' },
        { label: 'Service Mesh', value: 'Istio 1.19', description: 'Microservices communication layer', status: 'optimal' },
        { label: 'Message Queue', value: 'Apache Kafka', description: 'Distributed streaming platform', status: 'optimal' },
        { label: 'Cache Layer', value: 'Redis Cluster', description: 'In-memory data structure store', status: 'optimal' },
        { label: 'CDN', value: 'CloudFlare Enterprise', description: 'Global content delivery network', status: 'optimal' },
      ]
    },
    {
      category: 'Compute Resources',
      specs: [
        { label: 'CPU Architecture', value: 'Intel Xeon Platinum 8375C', description: '32 cores, 2.9GHz base frequency' },
        { label: 'Total vCPUs', value: '1,024 cores', description: 'Distributed across 32 nodes' },
        { label: 'RAM', value: '2.5 TB DDR4', description: '64GB per node, ECC memory' },
        { label: 'GPU Acceleration', value: 'NVIDIA A100 80GB', description: '8x A100s for model inference' },
        { label: 'Storage', value: '50 TB NVMe SSD', description: 'High-performance persistent storage' },
        { label: 'Network', value: '100 Gbps Ethernet', description: 'Dedicated fiber optic backbone' },
      ]
    },
    {
      category: 'Database Systems',
      specs: [
        { label: 'Primary Database', value: 'PostgreSQL 15.4', description: 'ACID-compliant relational database' },
        { label: 'Time Series DB', value: 'InfluxDB 2.7', description: 'High-performance metrics storage' },
        { label: 'Document Store', value: 'MongoDB 7.0', description: 'Flexible schema for model metadata' },
        { label: 'Search Engine', value: 'Elasticsearch 8.10', description: 'Full-text search and analytics' },
        { label: 'Graph Database', value: 'Neo4j 5.12', description: 'Relationship mapping and analysis' },
        { label: 'Vector Database', value: 'Pinecone', description: 'Semantic similarity search' },
      ]
    }
  ];

  const apiEndpoints: APIEndpoint[] = [
    {
      method: 'POST',
      path: '/api/v2/evaluations',
      description: 'Create new strategic evaluation',
      parameters: ['model_a', 'model_b', 'task_config', 'priority'],
      response: 'evaluation_id, status, estimated_completion'
    },
    {
      method: 'GET',
      path: '/api/v2/evaluations/{id}',
      description: 'Get evaluation status and results',
      parameters: ['include_details', 'format'],
      response: 'evaluation_data, progress, results'
    },
    {
      method: 'GET',
      path: '/api/v2/models',
      description: 'List available models and capabilities',
      parameters: ['category', 'status', 'sort'],
      response: 'models[], metadata, pagination'
    },
    {
      method: 'POST',
      path: '/api/v2/models/register',
      description: 'Register new model for evaluation',
      parameters: ['model_config', 'access_credentials', 'capabilities'],
      response: 'model_id, validation_status, deployment_url'
    },
    {
      method: 'GET',
      path: '/api/v2/metrics/system',
      description: 'Get real-time system metrics',
      parameters: ['time_range', 'granularity', 'metrics'],
      response: 'system_health, performance_data, alerts'
    }
  ];

  const architectureLayers: Architecture[] = [
    {
      layer: 'Presentation Layer',
      components: [
        { name: 'React Frontend', technology: 'Vite + TypeScript', status: 'active', load: 23, description: 'Strategic Command Center interface' },
        { name: 'API Gateway', technology: 'Kong Gateway', status: 'active', load: 45, description: 'Rate limiting and authentication' },
        { name: 'WebSocket Service', technology: 'Socket.io', status: 'active', load: 12, description: 'Real-time updates and notifications' }
      ]
    },
    {
      layer: 'Application Layer',
      components: [
        { name: 'Evaluation Engine', technology: 'Node.js + Python', status: 'active', load: 67, description: 'Core strategic evaluation processing' },
        { name: 'Model Registry', technology: 'FastAPI', status: 'active', load: 34, description: 'Model metadata and capability management' },
        { name: 'Result Analyzer', technology: 'Python + NumPy', status: 'maintenance', load: 0, description: 'Statistical analysis and ranking' }
      ]
    },
    {
      layer: 'Data Layer',
      components: [
        { name: 'Primary Database', technology: 'PostgreSQL', status: 'active', load: 56, description: 'Evaluation results and user data' },
        { name: 'Cache Cluster', technology: 'Redis', status: 'active', load: 23, description: 'Session and frequently accessed data' },
        { name: 'Object Storage', technology: 'MinIO S3', status: 'active', load: 15, description: 'Model artifacts and large files' }
      ]
    },
    {
      layer: 'Infrastructure Layer',
      components: [
        { name: 'Container Orchestration', technology: 'Kubernetes', status: 'active', load: 78, description: 'Service deployment and scaling' },
        { name: 'Service Mesh', technology: 'Istio', status: 'active', load: 34, description: 'Inter-service communication' },
        { name: 'Monitoring Stack', technology: 'Prometheus + Grafana', status: 'active', load: 45, description: 'Metrics collection and visualization' }
      ]
    }
  ];

  const securitySpecs = [
    { feature: 'Multi-Factor Authentication', status: 'Enabled', description: 'TOTP + Hardware keys supported' },
    { feature: 'Zero Trust Network', status: 'Active', description: 'Every request verified and encrypted' },
    { feature: 'Data Encryption', status: 'AES-256', description: 'At rest and in transit encryption' },
    { feature: 'API Rate Limiting', status: 'Configured', description: '1000 req/min per API key' },
    { feature: 'Audit Logging', status: 'Comprehensive', description: 'All actions logged and immutable' },
    { feature: 'Vulnerability Scanning', status: 'Automated', description: 'Daily security assessments' },
  ];

  const performanceMetrics = [
    { metric: 'Average Response Time', value: '89ms', target: '<100ms', status: 'optimal' as const },
    { metric: 'Throughput', value: '1,247 req/min', target: '>1000 req/min', status: 'optimal' as const },
    { metric: 'Uptime SLA', value: '99.97%', target: '99.9%', status: 'optimal' as const },
    { metric: 'Error Rate', value: '0.03%', target: '<0.1%', status: 'optimal' as const },
    { metric: 'P95 Latency', value: '156ms', target: '<200ms', status: 'optimal' as const },
    { metric: 'Concurrent Users', value: '2,847', target: '>2000', status: 'optimal' as const },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'optimal': return 'text-green-500 bg-green-500/10 border-green-500/30';
      case 'inactive': return 'text-gray-500 bg-gray-500/10 border-gray-500/30';
      case 'maintenance': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30';
      case 'warning': return 'text-orange-500 bg-orange-500/10 border-orange-500/30';
      case 'critical': return 'text-red-500 bg-red-500/10 border-red-500/30';
      default: return 'text-muted-foreground bg-muted/10 border-muted/30';
    }
  };

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET': return 'performance-elite strategic-rank';
      case 'POST': return 'performance-superior strategic-rank';
      case 'PUT': return 'rank-gold strategic-rank';
      case 'DELETE': return 'rank-bronze strategic-rank';
      default: return 'performance-standard strategic-rank';
    }
  };

  return (
    <div className={`glass-panel transition-all duration-500 ${isExpanded ? 'fixed inset-4 z-50' : 'w-full max-w-6xl mx-auto'}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="glass-subtle p-2 rounded-xl">
              <FileText className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="heading-refined text-lg">Technical Specifications</CardTitle>
              <p className="text-xs text-muted-foreground">
                Comprehensive platform architecture and capabilities reference
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="glass-minimal">
              <Download className="h-4 w-4 mr-2" />
              <span className="text-xs">Export</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="glass-minimal p-2"
            >
              {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs value={activeSpec} onValueChange={setActiveSpec} className="space-y-6">
          <TabsList className="glass-subtle p-1 rounded-xl grid grid-cols-5 w-full max-w-2xl">
            <TabsTrigger value="system" className="text-xs">System</TabsTrigger>
            <TabsTrigger value="api" className="text-xs">API</TabsTrigger>
            <TabsTrigger value="architecture" className="text-xs">Architecture</TabsTrigger>
            <TabsTrigger value="security" className="text-xs">Security</TabsTrigger>
            <TabsTrigger value="performance" className="text-xs">Performance</TabsTrigger>
          </TabsList>

          <TabsContent value="system" className="space-y-6">
            {technicalSpecs.map((category, categoryIndex) => (
              <div key={category.category}>
                <Button
                  variant="ghost"
                  onClick={() => toggleSection(category.category)}
                  className="w-full justify-between p-4 h-auto glass-subtle hover:shadow-md transition-all duration-300"
                >
                  <div className="flex items-center gap-3">
                    {categoryIndex === 0 && <Layers className="h-4 w-4 text-primary" />}
                    {categoryIndex === 1 && <Cpu className="h-4 w-4 text-secondary" />}
                    {categoryIndex === 2 && <Database className="h-4 w-4 text-accent" />}
                    <span className="heading-refined text-sm">{category.category}</span>
                  </div>
                  {expandedSections.includes(category.category) ? 
                    <ChevronDown className="h-4 w-4" /> : 
                    <ChevronRight className="h-4 w-4" />
                  }
                </Button>

                {expandedSections.includes(category.category) && (
                  <div className="mt-3 space-y-3 pl-4">
                    {category.specs.map((spec, specIndex) => (
                      <Card key={specIndex} className="glass-minimal border-border/20">
                        <CardContent className="p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-3">
                              <span className="text-sm font-medium">{spec.label}</span>
                              {spec.status && (
                                <Badge className={`text-xs ${getStatusColor(spec.status)}`}>
                                  {spec.status.toUpperCase()}
                                </Badge>
                              )}
                            </div>
                            <span className="text-sm font-mono text-primary">{spec.value}</span>
                          </div>
                          {spec.description && (
                            <p className="text-xs text-muted-foreground">{spec.description}</p>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </TabsContent>

          <TabsContent value="api" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="heading-refined text-sm">REST API Endpoints</h3>
              <Badge className="performance-elite strategic-rank">v2.1.4</Badge>
            </div>

            <div className="space-y-3">
              {apiEndpoints.map((endpoint, index) => (
                <Card key={index} className="glass-subtle border-border/20 hover:shadow-md transition-all duration-300">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <Badge className={getMethodColor(endpoint.method)}>
                          {endpoint.method}
                        </Badge>
                        <div>
                          <div className="text-sm font-mono font-medium text-primary">{endpoint.path}</div>
                          <div className="text-xs text-muted-foreground">{endpoint.description}</div>
                        </div>
                      </div>
                      <Button variant="ghost" size="sm" className="glass-minimal p-1">
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>

                    {endpoint.parameters && (
                      <div className="mb-2">
                        <div className="text-xs text-muted-foreground mb-1">Parameters:</div>
                        <div className="flex flex-wrap gap-2">
                          {endpoint.parameters.map((param, paramIndex) => (
                            <Badge key={paramIndex} variant="outline" className="text-xs font-mono">
                              {param}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Response:</div>
                      <div className="text-xs font-mono text-green-400 bg-black/20 p-2 rounded">
                        {endpoint.response}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="architecture" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="heading-refined text-sm">System Architecture Layers</h3>
              <Button variant="outline" size="sm" className="glass-minimal">
                <ExternalLink className="h-4 w-4 mr-2" />
                View Diagram
              </Button>
            </div>

            <div className="space-y-4">
              {architectureLayers.map((layer, layerIndex) => (
                <Card key={layer.layer} className="glass-subtle border-border/20">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      {layerIndex === 0 && <Monitor className="h-4 w-4 text-primary" />}
                      {layerIndex === 1 && <Server className="h-4 w-4 text-secondary" />}
                      {layerIndex === 2 && <Database className="h-4 w-4 text-accent" />}
                      {layerIndex === 3 && <Cloud className="h-4 w-4 text-green-500" />}
                      {layer.layer}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {layer.components.map((component, componentIndex) => (
                      <div key={componentIndex} className="glass-minimal p-3 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <Badge className={getStatusColor(component.status)}>
                              {component.status.toUpperCase()}
                            </Badge>
                            <div>
                              <div className="text-sm font-medium">{component.name}</div>
                              <div className="text-xs text-muted-foreground">{component.technology}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-muted-foreground">Load</div>
                            <div className="text-sm font-bold text-primary">{component.load}%</div>
                          </div>
                        </div>
                        <Progress value={component.load} className="h-1 mb-2" />
                        <div className="text-xs text-muted-foreground">{component.description}</div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="security" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="heading-refined text-sm">Security Framework</h3>
              <Badge className="performance-elite strategic-rank">SOC 2 Type II</Badge>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {securitySpecs.map((spec, index) => (
                <Card key={index} className="glass-subtle border-border/20">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Lock className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-medium">{spec.feature}</span>
                      </div>
                      <Badge className="performance-superior strategic-rank">
                        {spec.status}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground">{spec.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="heading-refined text-sm">Performance Metrics</h3>
              <Badge className="performance-elite strategic-rank">SLA: 99.9%</Badge>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {performanceMetrics.map((metric, index) => (
                <Card key={index} className="glass-subtle border-border/20">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">{metric.metric}</span>
                      <Badge className={getStatusColor(metric.status)}>
                        {metric.status.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="text-lg font-bold text-primary">{metric.value}</div>
                      <div className="text-xs text-muted-foreground">Target: {metric.target}</div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Card className="glass-subtle border-border/20">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className="h-4 w-4 text-primary" />
                  System Health Score
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-2xl font-bold text-green-500">98.7%</span>
                  <Badge className="performance-elite strategic-rank">EXCELLENT</Badge>
                </div>
                <Progress value={98.7} className="h-3 mb-2" />
                <div className="text-xs text-muted-foreground">
                  Based on availability, performance, security, and user satisfaction metrics
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </div>
  );
};

export const TechnicalSpecsDrawer = memo(TechnicalSpecsDrawerComponent);