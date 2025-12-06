import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/ui/molecules/Dialog';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { TierGate } from '@/components/auth/TierGate';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle,
  Calendar,
  Activity,
  Target,
  Heart,
  Zap,
  Syringe,
  UserCheck,
  FileText,
  TrendingUp,
  Clock
} from 'lucide-react';

interface PEDProtocol {
  id: string;
  protocol_name: string;
  protocol_type: string;
  cycle_phase: string;
  start_date: string;
  planned_end_date: string;
  cycle_week: number;
  total_planned_weeks: number;
  compounds: Array<{
    name: string;
    dosage: string;
    frequency: string;
  }>;
  status: string;
  medical_approval: boolean;
  client_consent: boolean;
  coach_approval: boolean;
  medical_notes?: string;
}

interface DailyTracking {
  tracking_date: string;
  energy_level: number;
  libido: number;
  mood_stability: number;
  strength_feeling: number;
  recovery_quality: number;
  side_effects_experienced: string[];
  side_effect_severity: string;
  morning_weight_kg: number;
}

export const PEDsProtocolDashboard: React.FC = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [protocols, setProtocols] = useState<PEDProtocol[]>([]);
  const [activeProtocol, setActiveProtocol] = useState<PEDProtocol | null>(null);
  const [dailyTracking, setDailyTracking] = useState<DailyTracking[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchProtocols();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const fetchProtocols = async () => {
    try {
      const { data: protocolsData, error } = await supabase
        .from('peds_protocols')
        .select('*')
        .eq('client_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;

      setProtocols((protocolsData || []) as PEDProtocol[]);
      
      const active = protocolsData?.find(p => p.status === 'active');
      if (active) {
        setActiveProtocol(active as PEDProtocol);
        await fetchDailyTracking(active.id);
      }
    } catch (error) {
      console.error('Error fetching PED protocols:', error);
      toast({
        title: "Error",
        description: "Failed to load PED protocols",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchDailyTracking = async (protocolId: string) => {
    try {
      const { data, error } = await supabase
        .from('peds_daily_tracking')
        .select('*')
        .eq('protocol_id', protocolId)
        .order('tracking_date', { ascending: true });

      if (error) throw error;
      setDailyTracking((data || []) as DailyTracking[]);
    } catch (error) {
      console.error('Error fetching daily tracking:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'completed': return 'bg-blue-100 text-blue-800';
      case 'discontinued': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'active': return 'text-green-600';
      case 'pct': return 'text-orange-600';
      case 'off': return 'text-gray-600';
      case 'cruise': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  const calculateProgress = (protocol: PEDProtocol) => {
    if (!protocol.start_date || !protocol.total_planned_weeks) return 0;
    
    const startDate = new Date(protocol.start_date);
    const now = new Date();
    const weeksElapsed = Math.floor((now.getTime() - startDate.getTime()) / (7 * 24 * 60 * 60 * 1000));
    
    return Math.min(Math.max((weeksElapsed / protocol.total_planned_weeks) * 100, 0), 100);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Activity className="h-8 w-8 animate-pulse text-primary" />
      </div>
    );
  }

  return (
    <TierGate requiredTier="performance" feature="peds_protocols">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
              <Shield className="h-6 w-6 text-red-600" />
              PEDs Protocol Management
            </h2>
            <p className="text-gray-600 mt-1">Performance Enhancement Protocols - Medical Supervision Required</p>
          </div>
          
          <Button onClick={() => console.log("New protocol")}>
            <Syringe className="h-4 w-4 mr-2" />
            Request New Protocol
          </Button>
        </div>

        {/* Medical Approval Alert */}
        <Alert className="border-red-500/30 bg-red-500/10">
          <AlertTriangle className="h-4 w-4 text-red-500" />
          <AlertDescription>
            <strong className="text-red-600">Medical Supervision Required:</strong> All PED protocols require medical approval and regular monitoring. 
            Consult with a qualified healthcare provider before starting any performance enhancement protocol.
          </AlertDescription>
        </Alert>

        {/* Active Protocol Overview */}
        {activeProtocol && (
          <Card className="border-l-4 border-l-green-500">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-green-600" />
                    {activeProtocol.protocol_name}
                  </CardTitle>
                  <CardDescription>
                    Week {activeProtocol.cycle_week} of {activeProtocol.total_planned_weeks} • 
                    <span className={`ml-1 font-medium ${getPhaseColor(activeProtocol.cycle_phase)}`}>
                      {activeProtocol.cycle_phase.toUpperCase()} PHASE
                    </span>
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusColor(activeProtocol.status)}>
                    {activeProtocol.status.toUpperCase()}
                  </Badge>
                  {activeProtocol.medical_approval && (
                    <Badge className="bg-green-100 text-green-800">
                      <UserCheck className="h-3 w-3 mr-1" />
                      MEDICALLY APPROVED
                    </Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Progress Bar */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Cycle Progress</span>
                    <span>{Math.round(calculateProgress(activeProtocol))}%</span>
                  </div>
                  <Progress value={calculateProgress(activeProtocol)} className="h-2" />
                </div>

                {/* Compounds */}
                <div>
                  <h4 className="font-medium mb-2">Current Compounds:</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {activeProtocol.compounds.map((compound, index: number) => (
                      <div key={index} className="bg-gray-50 rounded p-2 text-sm">
                        <div className="font-medium">{compound.name}</div>
                        <div className="text-gray-600">{compound.dosage}mg {compound.frequency}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Quick Actions */}
                <div className="flex gap-2 pt-2">
                  <Button size="sm" variant="outline">
                    <Calendar className="h-4 w-4 mr-2" />
                    Log Today
                  </Button>
                  <Button size="sm" variant="outline">
                    <Heart className="h-4 w-4 mr-2" />
                    View Monitoring
                  </Button>
                  <Button size="sm" variant="outline">
                    <FileText className="h-4 w-4 mr-2" />
                    Medical Notes
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Detailed Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="tracking">Daily Tracking</TabsTrigger>
            <TabsTrigger value="protocols">All Protocols</TabsTrigger>
            <TabsTrigger value="medical">Medical</TabsTrigger>
            <TabsTrigger value="education">Education</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {activeProtocol && dailyTracking.length > 0 ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Subjective Effects Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle>Subjective Effects Trends</CardTitle>
                    <CardDescription>Energy, mood, and performance metrics over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={dailyTracking}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="tracking_date" 
                            tickFormatter={(value) => new Date(value).toLocaleDateString()}
                          />
                          <YAxis domain={[1, 10]} />
                          <Tooltip 
                            labelFormatter={(value) => new Date(value).toLocaleDateString()}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="energy_level" 
                            stroke="#3b82f6" 
                            strokeWidth={2}
                            name="Energy"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="mood_stability" 
                            stroke="#10b981" 
                            strokeWidth={2}
                            name="Mood"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="strength_feeling" 
                            stroke="#f59e0b" 
                            strokeWidth={2}
                            name="Strength"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                {/* Side Effects Monitoring */}
                <Card>
                  <CardHeader>
                    <CardTitle>Side Effects Monitor</CardTitle>
                    <CardDescription>Safety tracking and adverse event monitoring</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {dailyTracking.slice(-7).map((day, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                          <div className="text-sm">
                            {new Date(day.tracking_date).toLocaleDateString()}
                          </div>
                          <div className="flex items-center gap-2">
                            {day.side_effects_experienced.length > 0 ? (
                              <Badge variant="outline" className="text-yellow-600">
                                {day.side_effects_experienced.length} effects
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="text-green-600">
                                No issues
                              </Badge>
                            )}
                            <span className={`text-sm ${
                              day.side_effect_severity === 'none' ? 'text-green-600' :
                              day.side_effect_severity === 'mild' ? 'text-yellow-600' :
                              day.side_effect_severity === 'moderate' ? 'text-orange-600' :
                              'text-red-600'
                            }`}>
                              {day.side_effect_severity || 'none'}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <Syringe className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No Active Protocol</h3>
                  <p className="text-muted-foreground mb-4">Start tracking your performance enhancement protocol</p>
                  <Button onClick={() => console.log("Request protocol")}>
                    Request Medical Consultation
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="tracking" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Daily Protocol Tracking</CardTitle>
                <CardDescription>Log your daily administration, effects, and observations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <Clock className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">Daily tracking interface will be implemented here</p>
                  <Button className="mt-4" onClick={() => console.log("Add tracking")}>
                    Add Today's Entry
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="protocols" className="space-y-4">
            <div className="grid gap-4">
              {protocols.map((protocol) => (
                <Card key={protocol.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-lg">{protocol.protocol_name}</CardTitle>
                        <CardDescription>
                          {protocol.protocol_type} • Started {new Date(protocol.start_date).toLocaleDateString()}
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={getStatusColor(protocol.status)}>
                          {protocol.status.toUpperCase()}
                        </Badge>
                        {protocol.medical_approval && (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Duration</p>
                        <p className="font-medium">{protocol.total_planned_weeks} weeks</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Phase</p>
                        <p className={`font-medium capitalize ${getPhaseColor(protocol.cycle_phase)}`}>
                          {protocol.cycle_phase}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Medical Approval</p>
                        <p className={`font-medium ${protocol.medical_approval ? 'text-green-600' : 'text-red-600'}`}>
                          {protocol.medical_approval ? 'Approved' : 'Pending'}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Compounds</p>
                        <p className="font-medium">{protocol.compounds.length} items</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="medical" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <UserCheck className="h-5 w-5 text-blue-600" />
                  Medical Oversight & Safety
                </CardTitle>
                <CardDescription>
                  Medical consultations, clearances, and safety monitoring
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Alert>
                    <Shield className="h-4 w-4" />
                    <AlertDescription>
                      All PED protocols require medical supervision. Regular blood work and health monitoring are mandatory.
                    </AlertDescription>
                  </Alert>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Button variant="outline" className="h-20 flex-col">
                      <UserCheck className="h-6 w-6 mb-2" />
                      Schedule Medical Consultation
                    </Button>
                    <Button variant="outline" className="h-20 flex-col">
                      <FileText className="h-6 w-6 mb-2" />
                      Upload Blood Work
                    </Button>
                    <Button variant="outline" className="h-20 flex-col">
                      <Heart className="h-6 w-6 mb-2" />
                      Health Monitoring
                    </Button>
                    <Button variant="outline" className="h-20 flex-col">
                      <Shield className="h-6 w-6 mb-2" />
                      Safety Resources
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="education" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-purple-600" />
                  Education & Safety Resources
                </CardTitle>
                <CardDescription>
                  Learn about safe practices, side effects, and harm reduction
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Compound Safety Profiles</h4>
                    <p className="text-sm text-muted-foreground">Detailed information about each compound's effects and safety</p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Side Effect Management</h4>
                    <p className="text-sm text-muted-foreground">How to identify and manage common side effects</p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Post Cycle Therapy</h4>
                    <p className="text-sm text-muted-foreground">Essential information about PCT protocols</p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Blood Work Interpretation</h4>
                    <p className="text-sm text-muted-foreground">Understanding your lab results and markers</p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Emergency Protocols</h4>
                    <p className="text-sm text-muted-foreground">What to do in case of adverse reactions</p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Legal Considerations</h4>
                    <p className="text-sm text-muted-foreground">Understanding legal status and regulations</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </TierGate>
  );
};