import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { 
  Shield, 
  ShieldAlert, 
  ShieldCheck,
  Lock,
  Unlock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Scan,
  Eye,
  Database,
  Globe
} from 'lucide-react';

interface SecurityTest {
  id: string;
  name: string;
  category: 'authentication' | 'authorization' | 'injection' | 'encryption' | 'network' | 'data';
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'pending' | 'running' | 'passed' | 'failed' | 'warning';
  description: string;
  lastRun?: string;
  duration?: number;
  issues?: number;
}

interface SecurityVulnerability {
  id: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: string;
  description: string;
  impact: string;
  recommendation: string;
  cve?: string;
}

export const SecurityTestSuite = () => {
  const [isScanning, setIsScanning] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  
  const [securityTests, setSecurityTests] = useState<SecurityTest[]>([
    {
      id: 'auth-bruteforce',
      name: 'Brute Force Protection',
      category: 'authentication',
      severity: 'high',
      status: 'passed',
      description: 'Tests protection against brute force attacks on login endpoints',
      lastRun: '2024-01-15T10:30:00Z',
      duration: 45000,
      issues: 0
    },
    {
      id: 'sql-injection',
      name: 'SQL Injection Tests',
      category: 'injection',
      severity: 'critical',
      status: 'failed',
      description: 'Scans for SQL injection vulnerabilities in database queries',
      lastRun: '2024-01-15T10:25:00Z',
      duration: 32000,
      issues: 2
    },
    {
      id: 'xss-protection',
      name: 'Cross-Site Scripting (XSS)',
      category: 'injection',
      severity: 'high',
      status: 'warning',
      description: 'Tests for XSS vulnerabilities in user inputs',
      lastRun: '2024-01-15T10:20:00Z',
      duration: 28000,
      issues: 1
    },
    {
      id: 'jwt-security',
      name: 'JWT Token Security',
      category: 'authentication',
      severity: 'high',
      status: 'passed',
      description: 'Validates JWT token implementation and security',
      lastRun: '2024-01-15T10:15:00Z',
      duration: 15000,
      issues: 0
    },
    {
      id: 'data-encryption',
      name: 'Data Encryption at Rest',
      category: 'encryption',
      severity: 'critical',
      status: 'passed',
      description: 'Verifies sensitive data is properly encrypted',
      lastRun: '2024-01-15T10:10:00Z',
      duration: 22000,
      issues: 0
    },
    {
      id: 'rbac-tests',
      name: 'Role-Based Access Control',
      category: 'authorization',
      severity: 'high',
      status: 'passed',
      description: 'Tests proper implementation of RBAC policies',
      lastRun: '2024-01-15T10:05:00Z',
      duration: 38000,
      issues: 0
    }
  ]);

  const [vulnerabilities, setVulnerabilities] = useState<SecurityVulnerability[]>([
    {
      id: 'vuln-1',
      title: 'Potential SQL Injection in User Search',
      severity: 'critical',
      category: 'injection',
      description: 'User search functionality may be vulnerable to SQL injection attacks',
      impact: 'Unauthorized database access, data theft, potential system compromise',
      recommendation: 'Implement parameterized queries and input validation',
      cve: 'CVE-2024-0001'
    },
    {
      id: 'vuln-2',
      title: 'XSS Vulnerability in Comment System',
      severity: 'high',
      category: 'injection',
      description: 'User comments are not properly sanitized before display',
      impact: 'Session hijacking, defacement, malicious script execution',
      recommendation: 'Implement proper HTML sanitization and CSP headers'
    }
  ]);

  const runSecurityScan = async () => {
    setIsScanning(true);
    
    // Reset all tests to pending
    setSecurityTests(prev => prev.map(test => ({ ...test, status: 'running' as const })));
    
    // Simulate security scan
    for (let i = 0; i < securityTests.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setSecurityTests(prev => prev.map((test, index) => {
        if (index === i) {
          const outcomes = ['passed', 'failed', 'warning'];
          const randomOutcome = outcomes[Math.floor(Math.random() * outcomes.length)] as SecurityTest['status'];
          return {
            ...test,
            status: randomOutcome,
            lastRun: new Date().toISOString(),
            issues: randomOutcome === 'failed' ? Math.floor(Math.random() * 3) + 1 : 
                   randomOutcome === 'warning' ? 1 : 0
          };
        }
        return test;
      }));
    }
    
    setIsScanning(false);
  };

  const getStatusIcon = (status: SecurityTest['status']) => {
    switch (status) {
      case 'running': return <Scan className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'passed': return <ShieldCheck className="h-4 w-4 text-green-500" />;
      case 'failed': return <ShieldAlert className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default: return <Shield className="h-4 w-4 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: SecurityTest['severity'] | SecurityVulnerability['severity']) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      case 'low': return 'outline';
    }
  };

  const getCategoryIcon = (category: SecurityTest['category']) => {
    switch (category) {
      case 'authentication': return <Lock className="h-4 w-4" />;
      case 'authorization': return <Unlock className="h-4 w-4" />;
      case 'injection': return <AlertTriangle className="h-4 w-4" />;
      case 'encryption': return <Shield className="h-4 w-4" />;
      case 'network': return <Globe className="h-4 w-4" />;
      case 'data': return <Database className="h-4 w-4" />;
    }
  };

  const filteredTests = selectedCategory === 'all' ? 
    securityTests : 
    securityTests.filter(test => test.category === selectedCategory);

  const testStats = {
    total: securityTests.length,
    passed: securityTests.filter(t => t.status === 'passed').length,
    failed: securityTests.filter(t => t.status === 'failed').length,
    warnings: securityTests.filter(t => t.status === 'warning').length,
    critical: vulnerabilities.filter(v => v.severity === 'critical').length,
    high: vulnerabilities.filter(v => v.severity === 'high').length
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Security Test Suite</h2>
          <p className="text-muted-foreground">Comprehensive security vulnerability scanning</p>
        </div>
        <Button 
          onClick={runSecurityScan} 
          disabled={isScanning}
          className="gap-2"
        >
          <Scan className={`h-4 w-4 ${isScanning ? 'animate-spin' : ''}`} />
          {isScanning ? 'Scanning...' : 'Run Security Scan'}
        </Button>
      </div>

      {/* Security Overview */}
      <div className="grid gap-4 md:grid-cols-6">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{testStats.total}</div>
            <p className="text-xs text-muted-foreground">Total Tests</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-green-600">{testStats.passed}</div>
            <p className="text-xs text-muted-foreground">Passed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-red-600">{testStats.failed}</div>
            <p className="text-xs text-muted-foreground">Failed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-yellow-600">{testStats.warnings}</div>
            <p className="text-xs text-muted-foreground">Warnings</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-red-700">{testStats.critical}</div>
            <p className="text-xs text-muted-foreground">Critical Issues</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-orange-600">{testStats.high}</div>
            <p className="text-xs text-muted-foreground">High Risk</p>
          </CardContent>
        </Card>
      </div>

      {/* Critical Alerts */}
      {testStats.critical > 0 && (
        <Alert>
          <ShieldAlert className="h-4 w-4" />
          <AlertDescription>
            <strong>Critical Security Issues Detected!</strong> {testStats.critical} critical vulnerabilities require immediate attention.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="tests" className="w-full">
        <TabsList>
          <TabsTrigger value="tests">Security Tests</TabsTrigger>
          <TabsTrigger value="vulnerabilities">Vulnerabilities</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
        </TabsList>

        <TabsContent value="tests">
          <div className="space-y-4">
            {/* Category Filter */}
            <div className="flex gap-2 flex-wrap">
              <Button
                variant={selectedCategory === 'all' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCategory('all')}
              >
                All Categories
              </Button>
              {['authentication', 'authorization', 'injection', 'encryption', 'network', 'data'].map((category) => (
                <Button
                  key={category}
                  variant={selectedCategory === category ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedCategory(category)}
                  className="capitalize"
                >
                  {category}
                </Button>
              ))}
            </div>

            {/* Test Results */}
            <div className="grid gap-4">
              {filteredTests.map((test) => (
                <Card key={test.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(test.status)}
                        <CardTitle className="text-lg">{test.name}</CardTitle>
                        <Badge variant={getSeverityColor(test.severity)}>
                          {test.severity}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        {getCategoryIcon(test.category)}
                        <Badge variant="outline" className="capitalize">
                          {test.category}
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-3">{test.description}</p>
                    
                    <div className="grid gap-2 md:grid-cols-4">
                      <div>
                        <p className="text-sm font-medium">Status</p>
                        <p className="capitalize">{test.status}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Issues Found</p>
                        <p className={test.issues && test.issues > 0 ? 'text-red-600 font-bold' : ''}>
                          {test.issues || 0}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Duration</p>
                        <p>{test.duration ? `${(test.duration / 1000).toFixed(1)}s` : 'N/A'}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Last Run</p>
                        <p className="text-sm">
                          {test.lastRun ? new Date(test.lastRun).toLocaleString() : 'Never'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="vulnerabilities">
          <div className="space-y-4">
            {vulnerabilities.map((vuln) => (
              <Card key={vuln.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{vuln.title}</CardTitle>
                    <div className="flex gap-2">
                      <Badge variant={getSeverityColor(vuln.severity)}>
                        {vuln.severity}
                      </Badge>
                      {vuln.cve && (
                        <Badge variant="outline">{vuln.cve}</Badge>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-1">Description</h4>
                      <p className="text-muted-foreground">{vuln.description}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-1">Impact</h4>
                      <p className="text-muted-foreground">{vuln.impact}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-1">Recommendation</h4>
                      <p className="text-muted-foreground">{vuln.recommendation}</p>
                    </div>
                    
                    <div className="flex gap-2">
                      <Button size="sm" onClick={() => console.log("SecurityTestSuite button clicked")}>Fix Issue</Button>
                      <Button size="sm" variant="outline" onClick={() => console.log("SecurityTestSuite button clicked")}>Mark as False Positive</Button>
                      <Button size="sm" variant="outline" onClick={() => console.log("SecurityTestSuite button clicked")}>View Details</Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="compliance">
          <Card>
            <CardHeader>
              <CardTitle>Security Compliance Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {[
                  { name: 'OWASP Top 10', score: 85, status: 'Good' },
                  { name: 'GDPR Compliance', score: 92, status: 'Excellent' },
                  { name: 'SOC 2 Type II', score: 78, status: 'Needs Improvement' },
                  { name: 'ISO 27001', score: 88, status: 'Good' }
                ].map((compliance) => (
                  <div key={compliance.name}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">{compliance.name}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm">{compliance.score}%</span>
                        <Badge variant={
                          compliance.score >= 90 ? 'default' : 
                          compliance.score >= 80 ? 'secondary' : 'destructive'
                        }>
                          {compliance.status}
                        </Badge>
                      </div>
                    </div>
                    <Progress value={compliance.score} />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};