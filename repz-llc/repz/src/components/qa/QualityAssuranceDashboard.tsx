import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { ComprehensiveTestRunner } from '../testing/ComprehensiveTestRunner';
import { SecurityTestSuite } from '../security/SecurityTestSuite';
import { ErrorTracker } from '../monitoring/ErrorTracker';
import { LoadTestingSuite } from '../production/LoadTestingSuite';
import { DeploymentMonitor } from '../production/DeploymentMonitor';
import { ProductionReadinessChecker } from '../production/ProductionReadinessChecker';
import { AdvancedPerformanceMonitor } from '../monitoring/AdvancedPerformanceMonitor';
import { CICDMonitor } from '../production/CICDMonitor';
import { TestAutomationScheduler } from '../automation/TestAutomationScheduler';
import { CodeQualityDashboard } from '../quality/CodeQualityDashboard';
import { APIMonitoringDashboard } from '../monitoring/APIMonitoringDashboard';
import { QAReportingDashboard } from '../reporting/QAReportingDashboard';
import { 
  TestTube, 
  Shield, 
  Bug, 
  BarChart3, 
  Rocket, 
  FileCheck, 
  Activity,
  GitBranch,
  Calendar,
  Code,
  Globe,
  FileText
} from 'lucide-react';

export const QualityAssuranceDashboard = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <TestTube className="h-6 w-6 text-blue-600" />
        <h1 className="text-3xl font-bold">Quality Assurance Dashboard</h1>
      </div>

      <p className="text-muted-foreground">
        Comprehensive testing, monitoring, and production readiness tools for maintaining high-quality software delivery.
      </p>

      <Tabs defaultValue="testing" className="w-full">
        <TabsList className="grid w-full grid-cols-10">
          <TabsTrigger value="testing" className="gap-2">
            <TestTube className="h-4 w-4" />
            Testing
          </TabsTrigger>
          <TabsTrigger value="security" className="gap-2">
            <Shield className="h-4 w-4" />
            Security
          </TabsTrigger>
          <TabsTrigger value="errors" className="gap-2">
            <Bug className="h-4 w-4" />
            Errors
          </TabsTrigger>
          <TabsTrigger value="performance" className="gap-2">
            <Activity className="h-4 w-4" />
            Performance
          </TabsTrigger>
          <TabsTrigger value="load" className="gap-2">
            <BarChart3 className="h-4 w-4" />
            Load Testing
          </TabsTrigger>
          <TabsTrigger value="deployment" className="gap-2">
            <Rocket className="h-4 w-4" />
            Deployment
          </TabsTrigger>
          <TabsTrigger value="readiness" className="gap-2">
            <FileCheck className="h-4 w-4" />
            Readiness
          </TabsTrigger>
          <TabsTrigger value="cicd" className="gap-2">
            <GitBranch className="h-4 w-4" />
            CI/CD
          </TabsTrigger>
          <TabsTrigger value="automation" className="gap-2">
            <Calendar className="h-4 w-4" />
            Automation
          </TabsTrigger>
          <TabsTrigger value="quality" className="gap-2">
            <Code className="h-4 w-4" />
            Code Quality
          </TabsTrigger>
          <TabsTrigger value="api" className="gap-2">
            <Globe className="h-4 w-4" />
            API Monitor
          </TabsTrigger>
          <TabsTrigger value="reporting" className="gap-2">
            <FileText className="h-4 w-4" />
            Reports
          </TabsTrigger>
        </TabsList>

        <TabsContent value="testing">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TestTube className="h-5 w-5" />
                Comprehensive Test Runner
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ComprehensiveTestRunner />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Security Testing Suite
              </CardTitle>
            </CardHeader>
            <CardContent>
              <SecurityTestSuite />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="errors">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bug className="h-5 w-5" />
                Error Tracking & Monitoring
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ErrorTracker />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Advanced Performance Monitor
              </CardTitle>
            </CardHeader>
            <CardContent>
              <AdvancedPerformanceMonitor />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="load">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Load Testing Suite
              </CardTitle>
            </CardHeader>
            <CardContent>
              <LoadTestingSuite />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="deployment">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Rocket className="h-5 w-5" />
                Deployment Monitor
              </CardTitle>
            </CardHeader>
            <CardContent>
              <DeploymentMonitor />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="readiness">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileCheck className="h-5 w-5" />
                Production Readiness Checker
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ProductionReadinessChecker />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cicd">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="h-5 w-5" />
                CI/CD Pipeline Monitor
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CICDMonitor />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="automation">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                Test Automation Scheduler
              </CardTitle>
            </CardHeader>
            <CardContent>
              <TestAutomationScheduler />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quality">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="h-5 w-5" />
                Code Quality Dashboard
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CodeQualityDashboard />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                API Monitoring Dashboard
              </CardTitle>
            </CardHeader>
            <CardContent>
              <APIMonitoringDashboard />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reporting">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                QA Reporting Dashboard
              </CardTitle>
            </CardHeader>
            <CardContent>
              <QAReportingDashboard />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Quick Stats Overview */}
      <div className="grid gap-4 md:grid-cols-6">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-green-600">87%</div>
            <p className="text-xs text-muted-foreground">Test Success Rate</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-blue-600">2</div>
            <p className="text-xs text-muted-foreground">Security Issues</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-yellow-600">5</div>
            <p className="text-xs text-muted-foreground">Active Errors</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-purple-600">92%</div>
            <p className="text-xs text-muted-foreground">Production Ready</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-orange-600">98%</div>
            <p className="text-xs text-muted-foreground">API Uptime</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-indigo-600">91</div>
            <p className="text-xs text-muted-foreground">Quality Score</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};