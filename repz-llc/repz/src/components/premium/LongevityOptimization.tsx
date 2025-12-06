import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { useTierAccess } from '@/hooks/useTierAccess';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar } from 'recharts';
import { 
  Activity, 
  Heart, 
  Brain,
  Shield,
  Zap,
  Target,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Pill,
  TestTube,
  Dna,
  Crown,
  Calendar,
  BookOpen
} from 'lucide-react';

interface BiomarkerData {
  date: string;
  testosterone: number;
  growth_hormone: number;
  cortisol: number;
  vitamin_d: number;
  b12: number;
  omega3: number;
  crp: number;
  glucose: number;
  cholesterol: number;
  triglycerides: number;
}

interface LongevityScore {
  overall: number;
  cardiovascular: number;
  metabolic: number;
  hormonal: number;
  inflammatory: number;
  cognitive: number;
  cellular: number;
}

interface Intervention {
  id: string;
  type: 'supplement' | 'lifestyle' | 'therapy' | 'monitoring';
  name: string;
  description: string;
  dosage?: string;
  frequency: string;
  expected_outcome: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'recommended' | 'active' | 'completed' | 'discontinued';
  start_date?: string;
  next_review?: string;
}

export const LongevityOptimization: React.FC = () => {
  const [biomarkers, setBiomarkers] = useState<BiomarkerData[]>([]);
  const [longevityScore, setLongevityScore] = useState<LongevityScore | null>(null);
  const [interventions, setInterventions] = useState<Intervention[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();
  const { hasFeature } = useTierAccess();

  useEffect(() => {
    fetchData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchData = async () => {
    try {
      // Mock biomarker data
      const mockBiomarkers: BiomarkerData[] = [
        {
          date: '2024-01-01',
          testosterone: 650,
          growth_hormone: 2.1,
          cortisol: 15,
          vitamin_d: 32,
          b12: 450,
          omega3: 5.2,
          crp: 2.1,
          glucose: 95,
          cholesterol: 180,
          triglycerides: 120
        },
        {
          date: '2024-02-01',
          testosterone: 675,
          growth_hormone: 2.3,
          cortisol: 13,
          vitamin_d: 42,
          b12: 520,
          omega3: 6.1,
          crp: 1.8,
          glucose: 88,
          cholesterol: 175,
          triglycerides: 110
        },
        {
          date: '2024-03-01',
          testosterone: 720,
          growth_hormone: 2.5,
          cortisol: 12,
          vitamin_d: 48,
          b12: 580,
          omega3: 6.8,
          crp: 1.5,
          glucose: 85,
          cholesterol: 170,
          triglycerides: 95
        }
      ];

      const mockLongevityScore: LongevityScore = {
        overall: 87,
        cardiovascular: 92,
        metabolic: 89,
        hormonal: 85,
        inflammatory: 88,
        cognitive: 84,
        cellular: 86
      };

      const mockInterventions: Intervention[] = [
        {
          id: '1',
          type: 'supplement',
          name: 'NAD+ Precursor',
          description: 'Nicotinamide riboside for cellular energy and mitochondrial function',
          dosage: '300mg',
          frequency: 'Daily with breakfast',
          expected_outcome: 'Improved cellular energy, enhanced recovery',
          priority: 'high',
          status: 'active',
          start_date: '2024-01-15',
          next_review: '2024-04-15'
        },
        {
          id: '2',
          type: 'therapy',
          name: 'Cold Exposure Therapy',
          description: 'Structured cold plunge protocol for hormetic stress response',
          frequency: '3x per week, 3-4 minutes',
          expected_outcome: 'Improved stress resilience, enhanced recovery',
          priority: 'medium',
          status: 'active',
          start_date: '2024-02-01',
          next_review: '2024-03-01'
        },
        {
          id: '3',
          type: 'monitoring',
          name: 'HRV Tracking',
          description: 'Daily heart rate variability monitoring for autonomic optimization',
          frequency: 'Daily upon waking',
          expected_outcome: 'Optimized training and recovery protocols',
          priority: 'high',
          status: 'active',
          start_date: '2024-01-01'
        },
        {
          id: '4',
          type: 'supplement',
          name: 'Metformin',
          description: 'Glucose regulation and longevity benefits',
          dosage: '500mg',
          frequency: 'Daily with dinner',
          expected_outcome: 'Improved glucose sensitivity, longevity benefits',
          priority: 'critical',
          status: 'recommended'
        }
      ];

      setBiomarkers(mockBiomarkers);
      setLongevityScore(mockLongevityScore);
      setInterventions(mockInterventions);
    } catch (error) {
      console.error('Error fetching data:', error);
      toast({
        title: "Error",
        description: "Failed to load longevity optimization data",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 80) return 'text-blue-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreGradient = (score: number) => {
    if (score >= 90) return 'from-green-500 to-green-600';
    if (score >= 80) return 'from-blue-500 to-blue-600';
    if (score >= 70) return 'from-yellow-500 to-yellow-600';
    return 'from-red-500 to-red-600';
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'high': return <TrendingUp className="h-4 w-4 text-orange-500" />;
      case 'medium': return <Target className="h-4 w-4 text-yellow-500" />;
      case 'low': return <CheckCircle className="h-4 w-4 text-green-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'supplement': return <Pill className="h-4 w-4" />;
      case 'therapy': return <Heart className="h-4 w-4" />;
      case 'monitoring': return <Activity className="h-4 w-4" />;
      case 'lifestyle': return <Brain className="h-4 w-4" />;
      default: return <TestTube className="h-4 w-4" />;
    }
  };

  const radarData = longevityScore ? [
    { subject: 'Cardiovascular', A: longevityScore.cardiovascular, fullMark: 100 },
    { subject: 'Metabolic', A: longevityScore.metabolic, fullMark: 100 },
    { subject: 'Hormonal', A: longevityScore.hormonal, fullMark: 100 },
    { subject: 'Inflammatory', A: longevityScore.inflammatory, fullMark: 100 },
    { subject: 'Cognitive', A: longevityScore.cognitive, fullMark: 100 },
    { subject: 'Cellular', A: longevityScore.cellular, fullMark: 100 }
  ] : [];

  if (!hasFeature('biomarker_integration')) {
    return (
      <Card className="border-gold">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-gold" />
            <CardTitle>Longevity Optimization</CardTitle>
          </div>
          <CardDescription>
            Advanced longevity protocols available with Longevity tier
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center py-8">
          <p className="text-muted-foreground mb-4">
            Upgrade to access comprehensive longevity optimization with biomarker tracking and personalized interventions
          </p>
          <Button variant="outline" onClick={() => window.location.href = '/pricing'}>
            Upgrade to Longevity
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Loading Longevity Data...</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="border-gold">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-gold" />
            <CardTitle>Longevity Optimization Protocol</CardTitle>
            <Badge variant="secondary" className="ml-auto">
              Precision Medicine
            </Badge>
          </div>
          <CardDescription>
            Evidence-based longevity interventions tailored to your biomarkers
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Longevity Score Overview */}
      {longevityScore && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="col-span-2 md:col-span-1">
            <CardContent className="pt-4">
              <div className="text-center">
                <div className={`text-4xl font-bold ${getScoreColor(longevityScore.overall)} mb-2`}>
                  {longevityScore.overall}
                </div>
                <div className="text-sm text-muted-foreground">Overall Longevity Score</div>
                <div className={`h-2 bg-gradient-to-r ${getScoreGradient(longevityScore.overall)} rounded-full mt-2`} />
              </div>
            </CardContent>
          </Card>
          
          {[
            { name: 'Cardiovascular', score: longevityScore.cardiovascular, icon: Heart },
            { name: 'Metabolic', score: longevityScore.metabolic, icon: Zap },
            { name: 'Cognitive', score: longevityScore.cognitive, icon: Brain }
          ].map((category) => {
            const IconComponent = category.icon;
            return (
              <Card key={category.name}>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between mb-2">
                    <IconComponent className="h-4 w-4 text-muted-foreground" />
                    <div className={`text-xl font-bold ${getScoreColor(category.score)}`}>
                      {category.score}
                    </div>
                  </div>
                  <div className="text-sm font-medium">{category.name}</div>
                  <Progress value={category.score} className="h-1 mt-2" />
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="biomarkers">Biomarkers</TabsTrigger>
          <TabsTrigger value="interventions">Interventions</TabsTrigger>
          <TabsTrigger value="protocols">Protocols</TabsTrigger>
          <TabsTrigger value="research">Research</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Longevity Profile</CardTitle>
                <CardDescription>Multi-dimensional health assessment</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar name="Score" dataKey="A" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Priority Interventions</CardTitle>
                <CardDescription>Recommended actions for optimization</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {interventions
                  .filter(i => i.priority === 'critical' || i.priority === 'high')
                  .slice(0, 4)
                  .map((intervention) => (
                    <div key={intervention.id} className="flex items-center gap-3 p-3 border rounded-lg">
                      <div className="flex items-center gap-2">
                        {getPriorityIcon(intervention.priority)}
                        {getTypeIcon(intervention.type)}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-sm">{intervention.name}</div>
                        <div className="text-xs text-muted-foreground">{intervention.frequency}</div>
                      </div>
                      <Badge variant={intervention.status === 'active' ? 'default' : 'secondary'} className="text-xs">
                        {intervention.status}
                      </Badge>
                    </div>
                  ))}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Key Health Trends</CardTitle>
              <CardDescription>Critical biomarker trajectories</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={biomarkers}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="testosterone" stroke="#3b82f6" name="Testosterone (ng/dL)" />
                  <Line type="monotone" dataKey="vitamin_d" stroke="#22c55e" name="Vitamin D (ng/mL)" />
                  <Line type="monotone" dataKey="crp" stroke="#ef4444" name="CRP (mg/L)" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="biomarkers" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Comprehensive Biomarker Panel</CardTitle>
              <CardDescription>Latest lab results and optimal ranges</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {biomarkers.length > 0 && [
                  { name: 'Testosterone', value: biomarkers[biomarkers.length - 1].testosterone, unit: 'ng/dL', optimal: [400, 1000], critical: false },
                  { name: 'Growth Hormone', value: biomarkers[biomarkers.length - 1].growth_hormone, unit: 'ng/mL', optimal: [0.5, 3.0], critical: false },
                  { name: 'Cortisol', value: biomarkers[biomarkers.length - 1].cortisol, unit: 'μg/dL', optimal: [6, 20], critical: false },
                  { name: 'Vitamin D', value: biomarkers[biomarkers.length - 1].vitamin_d, unit: 'ng/mL', optimal: [30, 60], critical: false },
                  { name: 'CRP', value: biomarkers[biomarkers.length - 1].crp, unit: 'mg/L', optimal: [0, 1.0], critical: biomarkers[biomarkers.length - 1].crp > 3 },
                  { name: 'Glucose', value: biomarkers[biomarkers.length - 1].glucose, unit: 'mg/dL', optimal: [70, 100], critical: biomarkers[biomarkers.length - 1].glucose > 125 }
                ].map((marker) => {
                  const isInRange = marker.value >= marker.optimal[0] && marker.value <= marker.optimal[1];
                  return (
                    <div key={marker.name} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <TestTube className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <div className="font-medium">{marker.name}</div>
                          <div className="text-sm text-muted-foreground">
                            Optimal: {marker.optimal[0]} - {marker.optimal[1]} {marker.unit}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-bold ${marker.critical ? 'text-red-600' : isInRange ? 'text-green-600' : 'text-yellow-600'}`}>
                          {marker.value} {marker.unit}
                        </div>
                        <Badge variant={marker.critical ? 'destructive' : isInRange ? 'default' : 'secondary'} className="text-xs">
                          {marker.critical ? 'Critical' : isInRange ? 'Optimal' : 'Sub-optimal'}
                        </Badge>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Biomarker Trends</CardTitle>
              <CardDescription>Track changes over time</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={biomarkers}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="testosterone" stroke="#3b82f6" name="Testosterone" />
                  <Line type="monotone" dataKey="growth_hormone" stroke="#8b5cf6" name="Growth Hormone" />
                  <Line type="monotone" dataKey="vitamin_d" stroke="#22c55e" name="Vitamin D" />
                  <Line type="monotone" dataKey="omega3" stroke="#f59e0b" name="Omega-3 Index" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="interventions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Interventions</CardTitle>
              <CardDescription>Current longevity optimization protocols</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {interventions.map((intervention) => (
                <Card key={intervention.id} className="border-l-4 border-l-primary">
                  <CardContent className="pt-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        {getTypeIcon(intervention.type)}
                        <h4 className="font-semibold">{intervention.name}</h4>
                        {getPriorityIcon(intervention.priority)}
                      </div>
                      <Badge variant={intervention.status === 'active' ? 'default' : intervention.status === 'recommended' ? 'secondary' : 'outline'}>
                        {intervention.status}
                      </Badge>
                    </div>
                    
                    <p className="text-sm text-muted-foreground mb-3">{intervention.description}</p>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      {intervention.dosage && (
                        <div>
                          <span className="font-medium">Dosage:</span>
                          <p className="text-muted-foreground">{intervention.dosage}</p>
                        </div>
                      )}
                      <div>
                        <span className="font-medium">Frequency:</span>
                        <p className="text-muted-foreground">{intervention.frequency}</p>
                      </div>
                      <div>
                        <span className="font-medium">Expected Outcome:</span>
                        <p className="text-muted-foreground">{intervention.expected_outcome}</p>
                      </div>
                      {intervention.next_review && (
                        <div>
                          <span className="font-medium">Next Review:</span>
                          <p className="text-muted-foreground">{new Date(intervention.next_review).toLocaleDateString()}</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="protocols" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Longevity Protocols</CardTitle>
              <CardDescription>Evidence-based interventions for healthspan extension</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Hormetic Stress</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                      <span className="font-medium">Cold Exposure</span>
                      <Badge>Active</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-orange-50 rounded-lg">
                      <span className="font-medium">Heat Therapy</span>
                      <Badge variant="secondary">Planned</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                      <span className="font-medium">Fasting Protocols</span>
                      <Badge>Active</Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Cellular Optimization</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                      <span className="font-medium">NAD+ Enhancement</span>
                      <Badge>Active</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-pink-50 rounded-lg">
                      <span className="font-medium">Mitochondrial Support</span>
                      <Badge>Active</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                      <span className="font-medium">Autophagy Induction</span>
                      <Badge variant="secondary">Monitoring</Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="research" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Latest Longevity Research</CardTitle>
              <CardDescription>Cutting-edge studies relevant to your profile</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                {[
                  {
                    title: "NAD+ Precursors and Cellular Aging",
                    journal: "Nature Aging",
                    relevance: "High - Active NAD+ supplementation",
                    summary: "Recent study shows significant improvements in cellular energy metabolism with NR supplementation."
                  },
                  {
                    title: "Cold Exposure and Longevity Markers",
                    journal: "Cell Metabolism",
                    relevance: "High - Current cold therapy protocol",
                    summary: "Cold exposure activates longevity pathways including SIRT1 and AMPK signaling."
                  },
                  {
                    title: "Testosterone Optimization in Aging",
                    journal: "Journal of Clinical Endocrinology",
                    relevance: "Medium - Hormone optimization focus",
                    summary: "Optimal testosterone levels associated with improved healthspan and cognitive function."
                  }
                ].map((study, index) => (
                  <Card key={index} className="border-l-4 border-l-blue-500">
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold">{study.title}</h4>
                        <BookOpen className="h-4 w-4 text-muted-foreground" />
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">{study.journal}</p>
                      <p className="text-sm mb-2">{study.summary}</p>
                      <Badge variant="outline" className="text-xs">
                        {study.relevance}
                      </Badge>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};