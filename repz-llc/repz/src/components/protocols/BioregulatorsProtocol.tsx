import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/ui/molecules/Dialog';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { TierGate } from '@/components/auth/TierGate';
import { 
  Dna, 
  Brain, 
  Heart,
  Shield,
  Zap,
  Calendar,
  Clock,
  Activity,
  AlertTriangle,
  CheckCircle,
  Star,
  FileText,
  Beaker,
  TrendingUp,
  Target,
  Plus
} from 'lucide-react';

interface BioregulatorProtocol {
  id: string;
  protocol_name: string;
  bioregulator_type: string;
  peptide_category: string;
  regulatory_status: string;
  dosage_amount: number;
  dosage_unit: string;
  administration_route: string;
  frequency_per_day: number;
  cycle_length_weeks: number;
  start_date: string;
  planned_end_date: string;
  primary_targets: string[];
  expected_benefits: string[];
  mechanism_of_action: string;
  evidence_level: string;
  status: string;
  medical_approval: boolean;
}

interface DailyTracking {
  tracking_date: string;
  dose_taken: boolean;
  energy_vitality: number;
  cognitive_clarity: number;
  recovery_quality: number;
  immune_feeling: number;
  overall_wellbeing: number;
  sleep_quality: number;
  skin_appearance: number;
  joint_comfort: number;
  mental_sharpness: number;
  daily_observations: string;
}

const BIOREGULATOR_TYPES = [
  {
    name: 'Epitalon',
    category: 'geroprotective',
    description: 'Telomerase activator for cellular aging',
    targets: ['Telomere length', 'Cellular repair', 'Sleep quality'],
    evidence: 'moderate'
  },
  {
    name: 'Thymalin',
    category: 'immune_modulating',
    description: 'Thymus peptide for immune function',
    targets: ['Immune system', 'T-cell function', 'Inflammation'],
    evidence: 'limited'
  },
  {
    name: 'Cerebramine',
    category: 'organ_specific',
    description: 'Brain tissue extract for cognitive function',
    targets: ['Cognitive function', 'Neurotransmitter balance', 'Memory'],
    evidence: 'preliminary'
  },
  {
    name: 'Hepatamine',
    category: 'organ_specific',
    description: 'Liver tissue extract for detoxification',
    targets: ['Liver function', 'Detoxification', 'Metabolism'],
    evidence: 'limited'
  }
];

export const BioregulatorsProtocol: React.FC = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [protocols, setProtocols] = useState<BioregulatorProtocol[]>([]);
  const [activeProtocol, setActiveProtocol] = useState<BioregulatorProtocol | null>(null);
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
        .from('bioregulators_protocols')
        .select('*')
        .eq('client_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;

      setProtocols(protocolsData || []);
      
      const active = protocolsData?.find(p => p.status === 'active');
      if (active) {
        setActiveProtocol(active);
        await fetchDailyTracking(active.id);
      }
    } catch (error) {
      console.error('Error fetching bioregulator protocols:', error);
      toast({
        title: "Error",
        description: "Failed to load bioregulator protocols",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchDailyTracking = async (protocolId: string) => {
    try {
      const { data, error } = await supabase
        .from('bioregulators_daily_tracking')
        .select('*')
        .eq('protocol_id', protocolId)
        .order('tracking_date', { ascending: true });

      if (error) throw error;
      setDailyTracking(data || []);
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

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'geroprotective': return <Dna className="h-4 w-4" />;
      case 'organ_specific': return <Heart className="h-4 w-4" />;
      case 'immune_modulating': return <Shield className="h-4 w-4" />;
      case 'metabolic': return <Zap className="h-4 w-4" />;
      default: return <Beaker className="h-4 w-4" />;
    }
  };

  const getEvidenceColor = (level: string) => {
    switch (level) {
      case 'strong': return 'text-green-600';
      case 'moderate': return 'text-blue-600';
      case 'limited': return 'text-yellow-600';
      case 'preliminary': return 'text-orange-600';
      default: return 'text-gray-600';
    }
  };

  const calculateProgress = (protocol: BioregulatorProtocol) => {
    if (!protocol.start_date || !protocol.cycle_length_weeks) return 0;
    
    const startDate = new Date(protocol.start_date);
    const now = new Date();
    const weeksElapsed = Math.floor((now.getTime() - startDate.getTime()) / (7 * 24 * 60 * 60 * 1000));
    
    return Math.min(Math.max((weeksElapsed / protocol.cycle_length_weeks) * 100, 0), 100);
  };

  // Prepare radar chart data for wellbeing metrics
  const getRadarData = () => {
    if (dailyTracking.length === 0) return [];
    
    const latest = dailyTracking[dailyTracking.length - 1];
    return [
      { metric: 'Energy', value: latest.energy_vitality || 0, fullMark: 10 },
      { metric: 'Cognitive', value: latest.cognitive_clarity || 0, fullMark: 10 },
      { metric: 'Recovery', value: latest.recovery_quality || 0, fullMark: 10 },
      { metric: 'Immune', value: latest.immune_feeling || 0, fullMark: 10 },
      { metric: 'Wellbeing', value: latest.overall_wellbeing || 0, fullMark: 10 },
      { metric: 'Sleep', value: latest.sleep_quality || 0, fullMark: 10 }
    ];
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Dna className="h-8 w-8 animate-pulse text-primary" />
      </div>
    );
  }

  return (
    <TierGate requiredTier="longevity" feature="bioregulators_protocols">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
              <Dna className="h-6 w-6 text-purple-600" />
              Bioregulators Protocol
            </h2>
            <p className="text-gray-600 mt-1">Advanced peptide bioregulators for longevity and optimization</p>
          </div>
          
          <Button onClick={() => console.log("New protocol")}>
            <Plus className="h-4 w-4 mr-2" />
            Request Protocol
          </Button>
        </div>

        {/* Research Alert */}
        <Alert className="border-purple-500/30 bg-purple-500/10">
          <Brain className="h-4 w-4 text-purple-500" />
          <AlertDescription>
            <strong className="text-purple-600">Research-Based Protocols:</strong> Bioregulators are cutting-edge peptides with varying levels of scientific evidence. 
            Always consult with a qualified healthcare provider experienced in peptide therapy.
          </AlertDescription>
        </Alert>

        {/* Active Protocol Overview */}
        {activeProtocol && (
          <Card className="border-l-4 border-l-purple-500">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    {getCategoryIcon(activeProtocol.peptide_category)}
                    {activeProtocol.protocol_name}
                  </CardTitle>
                  <CardDescription>
                    {activeProtocol.bioregulator_type} • 
                    <span className={`ml-1 font-medium ${getEvidenceColor(activeProtocol.evidence_level)}`}>
                      {activeProtocol.evidence_level} evidence
                    </span>
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusColor(activeProtocol.status)}>
                    {activeProtocol.status.toUpperCase()}
                  </Badge>
                  {activeProtocol.medical_approval && (
                    <Badge className="bg-green-100 text-green-800">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      APPROVED
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

                {/* Protocol Details */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Dosage</p>
                    <p className="font-medium">{activeProtocol.dosage_amount}{activeProtocol.dosage_unit}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Route</p>
                    <p className="font-medium capitalize">{activeProtocol.administration_route}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Frequency</p>
                    <p className="font-medium">{activeProtocol.frequency_per_day}x daily</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Cycle Length</p>
                    <p className="font-medium">{activeProtocol.cycle_length_weeks} weeks</p>
                  </div>
                </div>

                {/* Primary Targets */}
                <div>
                  <h4 className="font-medium mb-2">Primary Targets:</h4>
                  <div className="flex flex-wrap gap-2">
                    {activeProtocol.primary_targets.map((target, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        <Target className="h-3 w-3 mr-1" />
                        {target}
                      </Badge>
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
                    <TrendingUp className="h-4 w-4 mr-2" />
                    View Progress
                  </Button>
                  <Button size="sm" variant="outline">
                    <FileText className="h-4 w-4 mr-2" />
                    Research Notes
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
            <TabsTrigger value="research">Research Library</TabsTrigger>
            <TabsTrigger value="bioregulators">Bioregulators Guide</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {activeProtocol && dailyTracking.length > 0 ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Wellbeing Radar Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle>Current Wellbeing Profile</CardTitle>
                    <CardDescription>Multi-dimensional assessment of bioregulator effects</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart data={getRadarData()}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                          <PolarRadiusAxis 
                            angle={90} 
                            domain={[0, 10]} 
                            tick={{ fontSize: 10 }}
                          />
                          <Radar
                            name="Current"
                            dataKey="value"
                            stroke="#8b5cf6"
                            fill="#8b5cf6"
                            fillOpacity={0.1}
                            strokeWidth={2}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                {/* Trends Over Time */}
                <Card>
                  <CardHeader>
                    <CardTitle>Progress Trends</CardTitle>
                    <CardDescription>Key metrics over time showing bioregulator effects</CardDescription>
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
                            dataKey="energy_vitality" 
                            stroke="#3b82f6" 
                            strokeWidth={2}
                            name="Energy"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="cognitive_clarity" 
                            stroke="#10b981" 
                            strokeWidth={2}
                            name="Cognitive"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="overall_wellbeing" 
                            stroke="#f59e0b" 
                            strokeWidth={2}
                            name="Wellbeing"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                {/* Mechanism of Action */}
                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Beaker className="h-5 w-5 text-purple-600" />
                      Mechanism of Action
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <p className="text-sm leading-relaxed">{activeProtocol.mechanism_of_action}</p>
                      
                      <div>
                        <h4 className="font-medium mb-2">Expected Benefits:</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {activeProtocol.expected_benefits.map((benefit, index) => (
                            <div key={index} className="flex items-center gap-2 text-sm">
                              <CheckCircle className="h-4 w-4 text-green-500" />
                              {benefit}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <Dna className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No Active Protocol</h3>
                  <p className="text-muted-foreground mb-4">Request a bioregulator protocol to begin optimization</p>
                  <Button onClick={() => console.log("Request protocol")}>
                    Request Consultation
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="tracking" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Daily Protocol Tracking</CardTitle>
                <CardDescription>Monitor your bioregulator effects and observations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <Clock className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">Daily tracking interface for bioregulator protocols</p>
                  <Button className="mt-4" onClick={() => console.log("Add tracking")}>
                    Log Today's Effects
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
                        <CardTitle className="text-lg flex items-center gap-2">
                          {getCategoryIcon(protocol.peptide_category)}
                          {protocol.protocol_name}
                        </CardTitle>
                        <CardDescription>
                          {protocol.bioregulator_type} • {protocol.peptide_category} • 
                          <span className={`ml-1 ${getEvidenceColor(protocol.evidence_level)}`}>
                            {protocol.evidence_level} evidence
                          </span>
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
                        <p className="text-muted-foreground">Dosage</p>
                        <p className="font-medium">{protocol.dosage_amount}{protocol.dosage_unit}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Cycle Length</p>
                        <p className="font-medium">{protocol.cycle_length_weeks} weeks</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Status</p>
                        <p className="font-medium capitalize">{protocol.regulatory_status}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Targets</p>
                        <p className="font-medium">{protocol.primary_targets.length} areas</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="research" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-blue-600" />
                  Research Library & Citations
                </CardTitle>
                <CardDescription>
                  Scientific literature and research supporting bioregulator protocols
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">Research library and citation database will be available here</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="bioregulators" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {BIOREGULATOR_TYPES.map((bioregulator, index) => (
                <Card key={index}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      {getCategoryIcon(bioregulator.category)}
                      {bioregulator.name}
                    </CardTitle>
                    <CardDescription>{bioregulator.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <Badge variant="outline" className="mb-2">
                          {bioregulator.category.replace('_', ' ')}
                        </Badge>
                        <Badge variant="outline" className={`ml-2 ${getEvidenceColor(bioregulator.evidence)}`}>
                          {bioregulator.evidence} evidence
                        </Badge>
                      </div>
                      
                      <div>
                        <h4 className="font-medium mb-2">Primary Targets:</h4>
                        <div className="space-y-1">
                          {bioregulator.targets.map((target, targetIndex) => (
                            <div key={targetIndex} className="flex items-center gap-2 text-sm">
                              <Target className="h-3 w-3 text-purple-500" />
                              {target}
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <Button variant="outline" size="sm" className="w-full">
                        <FileText className="h-3 w-3 mr-2" />
                        Learn More
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </TierGate>
  );
};