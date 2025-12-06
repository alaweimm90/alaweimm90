import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { useTierAccess } from '@/hooks/useTierAccess';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, RadialBarChart, RadialBar } from 'recharts';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown,
  Target,
  Calendar,
  Camera,
  Zap,
  Heart,
  Dumbbell,
  Scale,
  Crown,
  BarChart3,
  Upload,
  Eye,
  Maximize
} from 'lucide-react';

interface BodyCompositionData {
  date: string;
  weight: number;
  body_fat: number;
  muscle_mass: number;
  visceral_fat: number;
  bone_density: number;
  water_percentage: number;
  metabolic_rate: number;
}

interface PhotoProgress {
  id: string;
  date: string;
  front_photo: string;
  side_photo: string;
  back_photo: string;
  notes?: string;
  measurements?: {
    chest: number;
    waist: number;
    hips: number;
    bicep: number;
    thigh: number;
  };
}

export const AdvancedBodyComposition: React.FC = () => {
  const [compositionData, setCompositionData] = useState<BodyCompositionData[]>([]);
  const [photoProgress, setPhotoProgress] = useState<PhotoProgress[]>([]);
  const [selectedPhoto, setSelectedPhoto] = useState<PhotoProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();
  const { hasFeature } = useTierAccess();

  useEffect(() => {
    fetchData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchData = async () => {
    try {
      // Mock data for demonstration
      const mockCompositionData: BodyCompositionData[] = [
        {
          date: '2024-01-01',
          weight: 180,
          body_fat: 18.5,
          muscle_mass: 135,
          visceral_fat: 8,
          bone_density: 3.2,
          water_percentage: 58,
          metabolic_rate: 2250
        },
        {
          date: '2024-01-15',
          weight: 178,
          body_fat: 17.8,
          muscle_mass: 136,
          visceral_fat: 7,
          bone_density: 3.3,
          water_percentage: 59,
          metabolic_rate: 2280
        },
        {
          date: '2024-02-01',
          weight: 176,
          body_fat: 16.9,
          muscle_mass: 138,
          visceral_fat: 6,
          bone_density: 3.4,
          water_percentage: 60,
          metabolic_rate: 2320
        },
        {
          date: '2024-02-15',
          weight: 175,
          body_fat: 16.2,
          muscle_mass: 140,
          visceral_fat: 5,
          bone_density: 3.5,
          water_percentage: 61,
          metabolic_rate: 2350
        }
      ];

      const mockPhotoProgress: PhotoProgress[] = [
        {
          id: '1',
          date: '2024-01-01',
          front_photo: '/lovable-uploads/user-profile.png',
          side_photo: '/lovable-uploads/user-profile.png',
          back_photo: '/lovable-uploads/user-profile.png',
          notes: 'Starting point - motivated and ready!',
          measurements: {
            chest: 42,
            waist: 34,
            hips: 40,
            bicep: 15,
            thigh: 24
          }
        },
        {
          id: '2',
          date: '2024-02-01',
          front_photo: '/lovable-uploads/user-profile.png',
          side_photo: '/lovable-uploads/user-profile.png',
          back_photo: '/lovable-uploads/user-profile.png',
          notes: 'Month 1 - visible definition improvements',
          measurements: {
            chest: 43,
            waist: 32.5,
            hips: 39,
            bicep: 15.5,
            thigh: 24.5
          }
        }
      ];

      setCompositionData(mockCompositionData);
      setPhotoProgress(mockPhotoProgress);
    } catch (error) {
      console.error('Error fetching data:', error);
      toast({
        title: "Error",
        description: "Failed to load body composition data",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const calculateProgress = (current: number, initial: number, isPositiveGood: boolean = true) => {
    const change = current - initial;
    const percentage = (Math.abs(change) / initial) * 100;
    const isImprovement = isPositiveGood ? change > 0 : change < 0;
    return { change, percentage, isImprovement };
  };

  const getLatestData = () => {
    if (compositionData.length === 0) return null;
    return compositionData[compositionData.length - 1];
  };

  const getInitialData = () => {
    if (compositionData.length === 0) return null;
    return compositionData[0];
  };

  const radialData = getLatestData() ? [
    { name: 'Body Fat', value: getLatestData()!.body_fat, fill: '#ef4444' },
    { name: 'Muscle', value: (getLatestData()!.muscle_mass / getLatestData()!.weight) * 100, fill: '#22c55e' },
    { name: 'Water', value: getLatestData()!.water_percentage, fill: '#3b82f6' }
  ] : [];

  if (!hasFeature('biomarker_integration')) {
    return (
      <Card className="border-gold">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-gold" />
            <CardTitle>Advanced Body Composition</CardTitle>
          </div>
          <CardDescription>
            Detailed body composition analysis available with Adaptive tier and above
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center py-8">
          <p className="text-muted-foreground mb-4">
            Upgrade to access comprehensive body composition tracking and AI-powered photo analysis
          </p>
          <Button variant="outline" onClick={() => window.location.href = '/pricing'}>
            Upgrade Plan
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Loading Body Composition Data...</CardTitle>
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

  const latestData = getLatestData();
  const initialData = getInitialData();

  return (
    <div className="space-y-6">
      <Card className="border-gold">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-gold" />
            <CardTitle>Advanced Body Composition Analysis</CardTitle>
            <Badge variant="secondary" className="ml-auto">
              AI-Powered
            </Badge>
          </div>
          <CardDescription>
            Comprehensive body composition tracking with AI photo analysis
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Key Metrics Overview */}
      {latestData && initialData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { 
              name: 'Weight', 
              current: latestData.weight, 
              initial: initialData.weight, 
              unit: 'lbs', 
              icon: Scale,
              isPositiveGood: false
            },
            { 
              name: 'Body Fat', 
              current: latestData.body_fat, 
              initial: initialData.body_fat, 
              unit: '%', 
              icon: Activity,
              isPositiveGood: false
            },
            { 
              name: 'Muscle Mass', 
              current: latestData.muscle_mass, 
              initial: initialData.muscle_mass, 
              unit: 'lbs', 
              icon: Dumbbell,
              isPositiveGood: true
            },
            { 
              name: 'Metabolic Rate', 
              current: latestData.metabolic_rate, 
              initial: initialData.metabolic_rate, 
              unit: 'cal', 
              icon: Zap,
              isPositiveGood: true
            }
          ].map((metric) => {
            const IconComponent = metric.icon;
            const progress = calculateProgress(metric.current, metric.initial, metric.isPositiveGood);
            return (
              <Card key={metric.name}>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between mb-2">
                    <IconComponent className="h-4 w-4 text-muted-foreground" />
                    {progress.isImprovement ? (
                      <TrendingUp className="h-4 w-4 text-green-500" />
                    ) : (
                      <TrendingDown className="h-4 w-4 text-red-500" />
                    )}
                  </div>
                  <div className="text-2xl font-bold">{metric.current}{metric.unit}</div>
                  <p className="text-xs text-muted-foreground">
                    {progress.change > 0 ? '+' : ''}{progress.change.toFixed(1)}{metric.unit} ({progress.percentage.toFixed(1)}%)
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <Tabs defaultValue="trends" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="composition">Composition</TabsTrigger>
          <TabsTrigger value="photos">Photo Progress</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Body Composition Trends</CardTitle>
              <CardDescription>Track your progress over time</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={compositionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="weight" stroke="#3b82f6" name="Weight (lbs)" />
                  <Line type="monotone" dataKey="body_fat" stroke="#ef4444" name="Body Fat %" />
                  <Line type="monotone" dataKey="muscle_mass" stroke="#22c55e" name="Muscle Mass (lbs)" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Metabolic Health</CardTitle>
              <CardDescription>Visceral fat and metabolic rate trends</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={compositionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="visceral_fat" stackId="1" stroke="#f59e0b" fill="#f59e0b" name="Visceral Fat" />
                  <Area type="monotone" dataKey="metabolic_rate" stackId="2" stroke="#8b5cf6" fill="#8b5cf6" name="BMR (cal)" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="composition" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Current Body Composition</CardTitle>
              <CardDescription>Latest measurement breakdown</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadialBarChart cx="50%" cy="50%" innerRadius="20%" outerRadius="90%" data={radialData}>
                      <RadialBar label={{ position: 'insideStart', fill: '#fff' }} background dataKey="value" />
                      <Legend iconSize={18} layout="horizontal" verticalAlign="bottom" align="center" />
                    </RadialBarChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-4">
                  {latestData && [
                    { label: 'Body Fat', value: latestData.body_fat, unit: '%', target: 15, color: 'bg-red-500' },
                    { label: 'Muscle Mass', value: (latestData.muscle_mass / latestData.weight) * 100, unit: '%', target: 80, color: 'bg-green-500' },
                    { label: 'Water', value: latestData.water_percentage, unit: '%', target: 60, color: 'bg-blue-500' },
                    { label: 'Bone Density', value: latestData.bone_density, unit: 'g/cm²', target: 3.5, color: 'bg-gray-500' }
                  ].map((metric) => (
                    <div key={metric.label} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>{metric.label}</span>
                        <span>{metric.value.toFixed(1)}{metric.unit} / {metric.target}{metric.unit}</span>
                      </div>
                      <Progress value={(metric.value / metric.target) * 100} className="h-2" />
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="photos" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Photo Progress</CardTitle>
                  <CardDescription>Visual transformation tracking</CardDescription>
                </div>
                <Button onClick={() => console.log("AdvancedBodyComposition button clicked")}>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Photos
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6">
                {photoProgress.map((progress) => (
                  <Card key={progress.id} className="border-l-4 border-l-primary">
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="font-semibold">{new Date(progress.date).toLocaleDateString()}</h4>
                          {progress.notes && (
                            <p className="text-sm text-muted-foreground">{progress.notes}</p>
                          )}
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setSelectedPhoto(progress)}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          View Details
                        </Button>
                      </div>

                      <div className="grid grid-cols-3 gap-4 mb-4">
                        {[
                          { label: 'Front', photo: progress.front_photo },
                          { label: 'Side', photo: progress.side_photo },
                          { label: 'Back', photo: progress.back_photo }
                        ].map((view) => (
                          <div key={view.label} className="space-y-2">
                            <p className="text-sm font-medium text-center">{view.label}</p>
                            <div className="aspect-[3/4] bg-muted rounded-lg overflow-hidden">
                              <img
                                src={view.photo}
                                alt={`${view.label} progress photo`}
                                className="w-full h-full object-cover"
                              />
                            </div>
                          </div>
                        ))}
                      </div>

                      {progress.measurements && (
                        <div className="grid grid-cols-5 gap-4 text-sm">
                          {Object.entries(progress.measurements).map(([part, measurement]) => (
                            <div key={part} className="text-center">
                              <p className="font-medium capitalize">{part}</p>
                              <p className="text-muted-foreground">{measurement}"</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Analysis & Insights</CardTitle>
              <CardDescription>Personalized recommendations based on your data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <h4 className="font-semibold text-green-800 mb-2">✅ Excellent Progress</h4>
                <p className="text-green-700 text-sm">
                  Your body composition is improving consistently. Muscle mass increased by 3.7% while body fat decreased by 12.4%. 
                  Continue current training and nutrition protocol.
                </p>
              </div>

              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-2">💡 Optimization Suggestion</h4>
                <p className="text-blue-700 text-sm">
                  Consider increasing protein intake by 15g daily to support your current muscle building rate. 
                  Your metabolic rate has increased significantly, indicating effective lean mass gains.
                </p>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                <h4 className="font-semibold text-yellow-800 mb-2">⚠️ Monitor Closely</h4>
                <p className="text-yellow-700 text-sm">
                  Hydration levels show slight downward trend. Ensure adequate water intake, especially around workouts. 
                  Target 1 gallon daily given your training intensity.
                </p>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Predicted Trajectory</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-600">14.5%</p>
                      <p className="text-sm text-muted-foreground">Body Fat in 3 months</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-600">145 lbs</p>
                      <p className="text-sm text-muted-foreground">Muscle Mass in 3 months</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-600">2,450</p>
                      <p className="text-sm text-muted-foreground">BMR in 3 months</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Photo Detail Modal would go here */}
      {selectedPhoto && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="max-w-4xl w-full max-h-[90vh] overflow-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Progress Photos - {new Date(selectedPhoto.date).toLocaleDateString()}</CardTitle>
                <Button variant="outline" onClick={() => setSelectedPhoto(null)}>
                  <Maximize className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: 'Front', photo: selectedPhoto.front_photo },
                  { label: 'Side', photo: selectedPhoto.side_photo },
                  { label: 'Back', photo: selectedPhoto.back_photo }
                ].map((view) => (
                  <div key={view.label} className="space-y-2">
                    <p className="font-medium text-center">{view.label}</p>
                    <img
                      src={view.photo}
                      alt={`${view.label} progress photo`}
                      className="w-full rounded-lg"
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};