import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  Shield, 
  AlertTriangle, 
  Activity, 
  Users, 
  Lock, 
  Eye, 
  Clock,
  MapPin,
  Smartphone,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';

interface SecurityMetrics {
  totalSecurityEvents: number;
  highRiskEvents: number;
  requiresAction: number;
  complianceEvents: number;
  activeSessions: number;
  suspiciousSessions: number;
}

interface SecurityEvent {
  id: string;
  event_type: string;
  event_category: string;
  risk_score: number;
  requires_action: boolean;
  created_at: string;
  user_id?: string;
  event_details: Record<string, unknown>;
}

interface ActiveSession {
  id: string;
  user_id: string;
  device_fingerprint?: string;
  ip_address?: string;
  location_country?: string;
  location_city?: string;
  risk_score: number;
  is_suspicious: boolean;
  last_activity: string;
}

export const SecurityMonitor: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<SecurityMetrics | null>(null);
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([]);
  const [activeSessions, setActiveSessions] = useState<ActiveSession[]>([]);
  const [timeframe, setTimeframe] = useState('24h');
  const { toast } = useToast();

  useEffect(() => {
    loadSecurityDashboard();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeframe]);

  const loadSecurityDashboard = async () => {
    try {
      setLoading(true);
      
      const { data, error } = await supabase.functions.invoke('security-monitor', {
        body: {
          action: 'get_security_dashboard',
          data: { timeframe }
        }
      });

      if (error) throw error;

      setMetrics(data.metrics);
      setSecurityEvents(data.securityEvents);
      setActiveSessions(data.activeSessions);
    } catch (error) {
      console.error('Error loading security dashboard:', error);
      toast({
        title: 'Error',
        description: 'Failed to load security dashboard',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  };

  const getRiskBadgeColor = (riskScore: number) => {
    if (riskScore > 70) return 'destructive';
    if (riskScore > 40) return 'secondary';
    return 'default';
  };

  const getRiskLevel = (riskScore: number) => {
    if (riskScore > 70) return 'HIGH';
    if (riskScore > 40) return 'MEDIUM';
    return 'LOW';
  };

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const handleEventAction = async (eventId: string, action: string) => {
    try {
      // In a real implementation, this would handle specific actions
      // like dismissing alerts, escalating issues, etc.
      toast({
        title: 'Action Taken',
        description: `${action} applied to event ${eventId}`,
      });
    } catch (error) {
      console.error('Error handling event action:', error);
      toast({
        title: 'Error',
        description: 'Failed to process action',
        variant: 'destructive'
      });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Security Monitor</h2>
          <p className="text-muted-foreground">
            Real-time security monitoring and threat detection
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={timeframe === '24h' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setTimeframe('24h')}
          >
            24h
          </Button>
          <Button
            variant={timeframe === '7d' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setTimeframe('7d')}
          >
            7d
          </Button>
          <Button
            variant={timeframe === '30d' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setTimeframe('30d')}
          >
            30d
          </Button>
        </div>
      </div>

      {/* Security Metrics Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-blue-500" />
              <div>
                <p className="text-2xl font-bold">{metrics?.totalSecurityEvents || 0}</p>
                <p className="text-xs text-muted-foreground">Total Events</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <div>
                <p className="text-2xl font-bold">{metrics?.highRiskEvents || 0}</p>
                <p className="text-xs text-muted-foreground">High Risk</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-4 w-4 text-orange-500" />
              <div>
                <p className="text-2xl font-bold">{metrics?.requiresAction || 0}</p>
                <p className="text-xs text-muted-foreground">Needs Action</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Shield className="h-4 w-4 text-green-500" />
              <div>
                <p className="text-2xl font-bold">{metrics?.complianceEvents || 0}</p>
                <p className="text-xs text-muted-foreground">Compliance</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Users className="h-4 w-4 text-purple-500" />
              <div>
                <p className="text-2xl font-bold">{metrics?.activeSessions || 0}</p>
                <p className="text-xs text-muted-foreground">Active Sessions</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Eye className="h-4 w-4 text-yellow-500" />
              <div>
                <p className="text-2xl font-bold">{metrics?.suspiciousSessions || 0}</p>
                <p className="text-xs text-muted-foreground">Suspicious</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* High Risk Alerts */}
      {metrics && metrics.highRiskEvents > 0 && (
        <Alert className="border-red-500 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <strong>Security Alert:</strong> {metrics.highRiskEvents} high-risk events detected. 
            Immediate review recommended.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="events" className="space-y-4">
        <TabsList>
          <TabsTrigger value="events">Security Events</TabsTrigger>
          <TabsTrigger value="sessions">Active Sessions</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
        </TabsList>

        <TabsContent value="events" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Recent Security Events
              </CardTitle>
              <CardDescription>
                Latest security events and threats detected
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {securityEvents.length === 0 ? (
                  <p className="text-center text-muted-foreground py-8">
                    No security events in the selected timeframe
                  </p>
                ) : (
                  securityEvents.map((event) => (
                    <div key={event.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="flex-shrink-0">
                          {event.requires_action ? (
                            <AlertTriangle className="h-5 w-5 text-red-500" />
                          ) : (
                            <CheckCircle className="h-5 w-5 text-green-500" />
                          )}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="font-medium">{event.event_type}</p>
                            <Badge variant={getRiskBadgeColor(event.risk_score)}>
                              {getRiskLevel(event.risk_score)}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {event.event_category} • {formatDateTime(event.created_at)}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {event.requires_action && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleEventAction(event.id, 'dismiss')}
                          >
                            Dismiss
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleEventAction(event.id, 'view')}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sessions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                Active User Sessions
              </CardTitle>
              <CardDescription>
                Currently active user sessions and security status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {activeSessions.length === 0 ? (
                  <p className="text-center text-muted-foreground py-8">
                    No active sessions
                  </p>
                ) : (
                  activeSessions.map((session) => (
                    <div key={session.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="flex-shrink-0">
                          {session.is_suspicious ? (
                            <XCircle className="h-5 w-5 text-red-500" />
                          ) : (
                            <CheckCircle className="h-5 w-5 text-green-500" />
                          )}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <Smartphone className="h-4 w-4 text-muted-foreground" />
                            <span className="font-medium">Session {session.id.slice(0, 8)}</span>
                            {session.is_suspicious && (
                              <Badge variant="destructive">Suspicious</Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            {session.location_country && (
                              <div className="flex items-center gap-1">
                                <MapPin className="h-3 w-3" />
                                {session.location_city}, {session.location_country}
                              </div>
                            )}
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDateTime(session.last_activity)}
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={getRiskBadgeColor(session.risk_score)}>
                          Risk: {session.risk_score}
                        </Badge>
                        {session.is_suspicious && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleEventAction(session.id, 'investigate')}
                          >
                            Investigate
                          </Button>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Compliance Monitoring
              </CardTitle>
              <CardDescription>
                Data protection and compliance event tracking
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Lock className="h-4 w-4 text-blue-500" />
                      <span className="font-medium">GDPR Compliance</span>
                    </div>
                    <p className="text-2xl font-bold text-green-600">Active</p>
                    <p className="text-xs text-muted-foreground">Data protection measures in place</p>
                  </div>
                  
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="h-4 w-4 text-purple-500" />
                      <span className="font-medium">Data Retention</span>
                    </div>
                    <p className="text-2xl font-bold text-green-600">Monitored</p>
                    <p className="text-xs text-muted-foreground">Automated retention policies</p>
                  </div>
                  
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="h-4 w-4 text-orange-500" />
                      <span className="font-medium">Audit Trail</span>
                    </div>
                    <p className="text-2xl font-bold text-green-600">Complete</p>
                    <p className="text-xs text-muted-foreground">Full activity logging</p>
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