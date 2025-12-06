import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Shield, FileText, Globe, Users, Lock, AlertTriangle, CheckCircle, Clock, Download } from 'lucide-react';

interface ComplianceStandard {
  name: string;
  status: 'compliant' | 'partial' | 'non-compliant';
  coverage: number;
  lastAudit: string;
  nextReview: string;
  requirements: number;
  completed: number;
}

interface DataProcessingActivity {
  id: string;
  purpose: string;
  dataTypes: string[];
  legalBasis: string;
  retention: string;
  recipients: string[];
  transfers: boolean;
}

export const ComplianceManager: React.FC = () => {
  const [complianceStandards, setComplianceStandards] = useState<ComplianceStandard[]>([
    {
      name: 'GDPR (EU)',
      status: 'compliant',
      coverage: 100,
      lastAudit: '2024-07-15',
      nextReview: '2024-10-15',
      requirements: 25,
      completed: 25,
    },
    {
      name: 'CCPA (California)',
      status: 'compliant',
      coverage: 95,
      lastAudit: '2024-07-20',
      nextReview: '2024-10-20',
      requirements: 18,
      completed: 17,
    },
    {
      name: 'HIPAA (US Healthcare)',
      status: 'compliant',
      coverage: 100,
      lastAudit: '2024-07-10',
      nextReview: '2024-10-10',
      requirements: 32,
      completed: 32,
    },
    {
      name: 'SOC 2 Type II',
      status: 'compliant',
      coverage: 98,
      lastAudit: '2024-06-30',
      nextReview: '2024-12-30',
      requirements: 40,
      completed: 39,
    },
    {
      name: 'ISO 27001',
      status: 'partial',
      coverage: 85,
      lastAudit: '2024-07-05',
      nextReview: '2024-10-05',
      requirements: 114,
      completed: 97,
    },
    {
      name: 'PCI DSS',
      status: 'compliant',
      coverage: 100,
      lastAudit: '2024-07-25',
      nextReview: '2024-10-25',
      requirements: 12,
      completed: 12,
    },
  ]);

  const [dataActivities] = useState<DataProcessingActivity[]>([
    {
      id: 'user-registration',
      purpose: 'User account creation and management',
      dataTypes: ['Name', 'Email', 'Phone', 'Address'],
      legalBasis: 'Contract performance',
      retention: '7 years after account closure',
      recipients: ['Internal teams', 'Cloud service providers'],
      transfers: false,
    },
    {
      id: 'fitness-tracking',
      purpose: 'Fitness progress monitoring and coaching',
      dataTypes: ['Health metrics', 'Workout data', 'Progress photos'],
      legalBasis: 'Consent',
      retention: '5 years or until consent withdrawal',
      recipients: ['Assigned coaches', 'Analytics team'],
      transfers: false,
    },
    {
      id: 'payment-processing',
      purpose: 'Subscription and payment processing',
      dataTypes: ['Payment information', 'Billing address', 'Transaction history'],
      legalBasis: 'Contract performance',
      retention: '10 years for financial records',
      recipients: ['Payment processors', 'Financial institutions'],
      transfers: true,
    },
  ]);

  const [auditLog, setAuditLog] = useState([
    {
      timestamp: '2024-08-01T10:30:00Z',
      event: 'Data export request',
      user: 'user@example.com',
      action: 'GDPR Article 20 - Data portability',
      status: 'completed',
    },
    {
      timestamp: '2024-08-01T09:15:00Z',
      event: 'Data deletion request',
      user: 'delete@example.com',
      action: 'GDPR Article 17 - Right to erasure',
      status: 'completed',
    },
    {
      timestamp: '2024-08-01T08:45:00Z',
      event: 'Consent withdrawal',
      user: 'withdraw@example.com',
      action: 'Marketing consent revoked',
      status: 'processed',
    },
  ]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant':
        return 'default';
      case 'partial':
        return 'secondary';
      case 'non-compliant':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'partial':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'non-compliant':
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  };

  const exportDataReport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      standards: complianceStandards,
      dataActivities,
      auditLog,
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `compliance-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Compliance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Standards</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{complianceStandards.length}</div>
            <p className="text-xs text-muted-foreground">
              Compliance frameworks
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fully Compliant</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {complianceStandards.filter(s => s.status === 'compliant').length}
            </div>
            <p className="text-xs text-muted-foreground">
              100% compliance achieved
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Activities</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dataActivities.length}</div>
            <p className="text-xs text-muted-foreground">
              Processing activities mapped
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Audit</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">15</div>
            <p className="text-xs text-muted-foreground">
              Days ago
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="standards" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="standards">
            <Shield className="h-4 w-4 mr-2" />
            Standards
          </TabsTrigger>
          <TabsTrigger value="data-processing">
            <FileText className="h-4 w-4 mr-2" />
            Data Processing
          </TabsTrigger>
          <TabsTrigger value="audit-log">
            <Users className="h-4 w-4 mr-2" />
            Audit Log
          </TabsTrigger>
          <TabsTrigger value="privacy-rights">
            <Lock className="h-4 w-4 mr-2" />
            Privacy Rights
          </TabsTrigger>
        </TabsList>

        <TabsContent value="standards" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Compliance Standards Overview
                <Button onClick={exportDataReport} variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export Report
                </Button>
              </CardTitle>
              <CardDescription>
                Current status across all regulatory frameworks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {complianceStandards.map((standard) => (
                  <div key={standard.name} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(standard.status)}
                        <div>
                          <h4 className="font-medium">{standard.name}</h4>
                          <p className="text-sm text-muted-foreground">
                            {standard.completed}/{standard.requirements} requirements completed
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className="text-sm font-medium">{standard.coverage}%</div>
                          <div className="text-xs text-muted-foreground">Coverage</div>
                        </div>
                        <Badge variant={getStatusColor(standard.status)}>
                          {standard.status}
                        </Badge>
                      </div>
                    </div>
                    <Progress value={standard.coverage} className="h-2" />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Last audit: {standard.lastAudit}</span>
                      <span>Next review: {standard.nextReview}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data-processing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Data Processing Activities</CardTitle>
              <CardDescription>
                Record of processing activities (ROPA) as required by GDPR Article 30
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {dataActivities.map((activity) => (
                  <div key={activity.id} className="border rounded-lg p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">{activity.purpose}</h4>
                      <Badge variant={activity.transfers ? 'secondary' : 'outline'}>
                        {activity.transfers ? 'International transfers' : 'Domestic only'}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Data types:</span>
                        <div className="mt-1">
                          {activity.dataTypes.map((type) => (
                            <Badge key={type} variant="outline" className="mr-1 mb-1">
                              {type}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <span className="font-medium">Legal basis:</span>
                        <p className="text-muted-foreground">{activity.legalBasis}</p>
                      </div>
                      <div>
                        <span className="font-medium">Retention period:</span>
                        <p className="text-muted-foreground">{activity.retention}</p>
                      </div>
                      <div>
                        <span className="font-medium">Recipients:</span>
                        <p className="text-muted-foreground">{activity.recipients.join(', ')}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audit-log" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Audit Log</CardTitle>
              <CardDescription>
                Track all compliance-related activities and data subject requests
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {auditLog.map((entry, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="space-y-1">
                      <div className="font-medium">{entry.event}</div>
                      <div className="text-sm text-muted-foreground">
                        {entry.user} - {entry.action}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(entry.timestamp).toLocaleString()}
                      </div>
                    </div>
                    <Badge variant={entry.status === 'completed' ? 'default' : 'secondary'}>
                      {entry.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="privacy-rights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Privacy Rights Management</CardTitle>
              <CardDescription>
                Tools for handling data subject rights under GDPR and other privacy laws
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-4">
                  <h4 className="font-medium">Available Rights</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2 border rounded">
                      <span className="text-sm">Right to Access (Art. 15)</span>
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="flex items-center justify-between p-2 border rounded">
                      <span className="text-sm">Right to Rectification (Art. 16)</span>
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="flex items-center justify-between p-2 border rounded">
                      <span className="text-sm">Right to Erasure (Art. 17)</span>
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="flex items-center justify-between p-2 border rounded">
                      <span className="text-sm">Right to Portability (Art. 20)</span>
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h4 className="font-medium">Response Times</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Average response time</span>
                      <span className="font-medium">2.3 days</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Legal requirement</span>
                      <span className="font-medium">30 days</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Requests this month</span>
                      <span className="font-medium">12</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Completion rate</span>
                      <span className="font-medium text-green-600">100%</span>
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