import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/ui/molecules/Card"
import { Badge } from "@/ui/atoms/Badge"
import { Button } from "@/ui/atoms/Button"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/ui/molecules/Alert"
import { 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Activity, 
  Database, 
  Wifi, 
  Shield,
  RefreshCw
} from "lucide-react"

interface QualityMetric {
  id: string
  name: string
  status: 'healthy' | 'warning' | 'critical'
  value: number
  threshold: number
  description: string
  lastChecked: string
}

const qualityMetrics: QualityMetric[] = [
  {
    id: 'performance',
    name: 'App Performance',
    status: 'healthy',
    value: 95,
    threshold: 80,
    description: 'Page load times and responsiveness',
    lastChecked: '2 minutes ago'
  },
  {
    id: 'uptime',
    name: 'System Uptime',
    status: 'healthy',
    value: 99.8,
    threshold: 99,
    description: 'Service availability and reliability',
    lastChecked: '1 minute ago'
  },
  {
    id: 'database',
    name: 'Database Health',
    status: 'warning',
    value: 85,
    threshold: 90,
    description: 'Query performance and connection health',
    lastChecked: '30 seconds ago'
  },
  {
    id: 'security',
    name: 'Security Score',
    status: 'healthy',
    value: 92,
    threshold: 85,
    description: 'Security compliance and vulnerability status',
    lastChecked: '5 minutes ago'
  }
]

export function QualityAssuranceMonitor() {
  const [metrics, setMetrics] = useState(qualityMetrics)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600'
      case 'warning': return 'text-yellow-600'
      case 'critical': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4" />
      case 'warning': return <AlertTriangle className="h-4 w-4" />
      case 'critical': return <XCircle className="h-4 w-4" />
      default: return <Activity className="h-4 w-4" />
    }
  }

  const getMetricIcon = (id: string) => {
    switch (id) {
      case 'performance': return <Activity className="h-5 w-5" />
      case 'uptime': return <Wifi className="h-5 w-5" />
      case 'database': return <Database className="h-5 w-5" />
      case 'security': return <Shield className="h-5 w-5" />
      default: return <Activity className="h-5 w-5" />
    }
  }

  const handleRefresh = async () => {
    setIsRefreshing(true)
    // Simulate refresh delay
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // Simulate metric updates
    setMetrics(current => current.map(metric => ({
      ...metric,
      lastChecked: 'just now',
      value: Math.max(70, Math.min(100, metric.value + (Math.random() - 0.5) * 10))
    })))
    
    setIsRefreshing(false)
  }

  const overallStatus = metrics.every(m => m.status === 'healthy') ? 'healthy' : 
                       metrics.some(m => m.status === 'critical') ? 'critical' : 'warning'

  const criticalIssues = metrics.filter(m => m.status === 'critical').length
  const warningIssues = metrics.filter(m => m.status === 'warning').length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Quality Assurance Monitor</h2>
          <p className="text-muted-foreground">Real-time monitoring of system health and performance</p>
        </div>
        <Button onClick={handleRefresh} disabled={isRefreshing} size="sm">
          <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Overall Status Alert */}
      <Alert className={
        overallStatus === 'healthy' ? 'border-green-200 bg-green-50' :
        overallStatus === 'critical' ? 'border-red-200 bg-red-50' :
        'border-yellow-200 bg-yellow-50'
      }>
        <div className={`flex items-center gap-2 ${getStatusColor(overallStatus)}`}>
          {getStatusIcon(overallStatus)}
          <AlertDescription>
            {overallStatus === 'healthy' ? 'All systems operational' :
             overallStatus === 'critical' ? `${criticalIssues} critical issue${criticalIssues > 1 ? 's' : ''} detected` :
             `${warningIssues} warning${warningIssues > 1 ? 's' : ''} detected`}
          </AlertDescription>
        </div>
      </Alert>

      {/* Metrics Grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {metrics.map((metric) => (
          <Card key={metric.id}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                {getMetricIcon(metric.id)}
                {metric.name}
              </CardTitle>
              <Badge variant="outline" className={`${getStatusColor(metric.status)} border-current`}>
                {getStatusIcon(metric.status)}
                <span className="ml-1 capitalize">{metric.status}</span>
              </Badge>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metric.value}%</div>
              <p className="text-xs text-muted-foreground mb-3">
                {metric.description}
              </p>
              
              <div className="space-y-2">
                <Progress 
                  value={metric.value} 
                  className={`h-2 ${
                    metric.status === 'healthy' ? '[&>div]:bg-green-500' :
                    metric.status === 'warning' ? '[&>div]:bg-yellow-500' :
                    '[&>div]:bg-red-500'
                  }`}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Threshold: {metric.threshold}%</span>
                  <span>Last checked: {metric.lastChecked}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Test Results Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Latest Test Results</CardTitle>
          <CardDescription>Automated testing pipeline status</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-green-600">247</div>
              <div className="text-sm text-muted-foreground">Tests Passed</div>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-yellow-600">3</div>
              <div className="text-sm text-muted-foreground">Tests Skipped</div>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-red-600">0</div>
              <div className="text-sm text-muted-foreground">Tests Failed</div>
            </div>
          </div>
          
          <div className="mt-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span>Coverage</span>
              <span>87%</span>
            </div>
            <Progress value={87} className="h-2" />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}