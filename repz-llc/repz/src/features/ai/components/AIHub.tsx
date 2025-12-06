import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  TrendingUp, 
  Target, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Zap,
  Activity,
  Utensils,
  Camera,
  RefreshCw
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';

interface AIFeature {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'loading' | 'error' | 'ready';
  lastUpdated: Date | null;
  confidence: number;
  icon: React.ReactNode;
}

const AIHub: React.FC = () => {
  const { toast } = useToast();
  const [features, setFeatures] = useState<AIFeature[]>([
    {
      id: 'workout-recommendations',
      name: 'Smart Workout Recommendations',
      description: 'AI-powered personalized workout suggestions based on your goals and progress',
      status: 'ready',
      lastUpdated: null,
      confidence: 0,
      icon: <Target className="h-5 w-5" />
    },
    {
      id: 'predictive-analytics',
      name: 'Predictive Health Analytics',
      description: 'Advanced analytics to predict your fitness journey and potential challenges',
      status: 'ready',
      lastUpdated: null,
      confidence: 0,
      icon: <TrendingUp className="h-5 w-5" />
    },
    {
      id: 'nutrition-ai',
      name: 'Nutrition AI Assistant',
      description: 'Personalized meal planning and nutrition optimization',
      status: 'ready',
      lastUpdated: null,
      confidence: 0,
      icon: <Utensils className="h-5 w-5" />
    },
    {
      id: 'form-analysis',
      name: 'Exercise Form Analysis',
      description: 'Computer vision-powered form checking and technique improvement',
      status: 'ready',
      lastUpdated: null,
      confidence: 0,
      icon: <Camera className="h-5 w-5" />
    }
  ]);

  const [recommendations, setRecommendations] = useState<Record<string, unknown> | null>(null);
  const [analytics, setAnalytics] = useState<Record<string, unknown> | null>(null);
  const [nutritionPlan, setNutritionPlan] = useState<Record<string, unknown> | null>(null);
  const [formAnalysis, setFormAnalysis] = useState<Record<string, unknown> | null>(null);

  const updateFeatureStatus = (featureId: string, status: AIFeature['status'], confidence = 0) => {
    setFeatures(prev => prev.map(feature => 
      feature.id === featureId 
        ? { ...feature, status, lastUpdated: new Date(), confidence }
        : feature
    ));
  };

  const generateWorkoutRecommendations = async () => {
    updateFeatureStatus('workout-recommendations', 'loading');
    
    try {
      const { data, error } = await supabase.functions.invoke('ai-workout-recommendations', {
        body: {
          userProfile: {
            age: 28,
            fitnessLevel: 'intermediate',
            goals: ['strength', 'muscle_gain'],
            injuries: [],
            preferences: ['compound_movements', 'free_weights'],
            availableTime: 60,
            equipment: ['barbell', 'dumbbells', 'bench']
          },
          recentWorkouts: [],
          progressData: [],
          preferences: {
            intensity: 'medium',
            duration: 60,
            workoutTypes: ['strength_training']
          }
        }
      });

      if (error) throw error;

      setRecommendations(data);
      updateFeatureStatus('workout-recommendations', 'active', data.confidence || 0.95);
      
      toast({
        title: "Workout Recommendations Generated",
        description: `AI created a personalized ${data.workoutName} workout for you.`,
      });

    } catch (error) {
      console.error('Error generating workout recommendations:', error);
      updateFeatureStatus('workout-recommendations', 'error');
      toast({
        title: "Error",
        description: "Failed to generate workout recommendations",
        variant: "destructive",
      });
    }
  };

  const generatePredictiveAnalytics = async () => {
    updateFeatureStatus('predictive-analytics', 'loading');
    
    try {
      const { data, error } = await supabase.functions.invoke('predictive-health-analytics', {
        body: {
          userId: 'demo-user',
          historicalData: {
            workouts: [],
            progressMeasurements: [],
            healthMetrics: [],
            sleepData: [],
            nutritionData: []
          },
          currentGoals: ['weight_loss', 'strength_gain'],
          timeframe: '3months'
        }
      });

      if (error) throw error;

      setAnalytics(data);
      updateFeatureStatus('predictive-analytics', 'active', data.confidence?.overall || 0.89);
      
      toast({
        title: "Predictive Analysis Complete",
        description: "AI has analyzed your data and generated insights for the next 3 months.",
      });

    } catch (error) {
      console.error('Error generating predictive analytics:', error);
      updateFeatureStatus('predictive-analytics', 'error');
      toast({
        title: "Error",
        description: "Failed to generate predictive analytics",
        variant: "destructive",
      });
    }
  };

  const generateNutritionPlan = async () => {
    updateFeatureStatus('nutrition-ai', 'loading');
    
    try {
      const { data, error } = await supabase.functions.invoke('nutrition-ai-assistant', {
        body: {
          userProfile: {
            age: 28,
            weight: 75,
            height: 175,
            activityLevel: 'active',
            goals: ['muscle_gain', 'energy_optimization'],
            dietaryRestrictions: [],
            allergies: [],
            preferences: ['high_protein', 'balanced']
          },
          currentMeals: [],
          fitnessData: {
            workoutIntensity: 'high',
            weeklyWorkouts: 4,
            currentProgram: 'strength_training'
          },
          healthMetrics: {
            energyLevels: 7,
            sleepQuality: 8,
            stressLevel: 4,
            digestiveHealth: 8
          },
          nutritionGoals: {
            calories: 2500,
            protein: 150,
            carbs: 300,
            fats: 80,
            fiber: 35,
            water: 3
          },
          requestType: 'meal_plan'
        }
      });

      if (error) throw error;

      setNutritionPlan(data);
      updateFeatureStatus('nutrition-ai', 'active', 0.92);
      
      toast({
        title: "Nutrition Plan Created",
        description: "AI has created a personalized meal plan optimized for your goals.",
      });

    } catch (error) {
      console.error('Error generating nutrition plan:', error);
      updateFeatureStatus('nutrition-ai', 'error');
      toast({
        title: "Error",
        description: "Failed to generate nutrition plan",
        variant: "destructive",
      });
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-500';
      case 'loading': return 'text-blue-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'loading': return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'error': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* AI Hub Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-purple-500" />
            AI & Machine Learning Hub
          </CardTitle>
          <p className="text-muted-foreground">
            Advanced AI-powered features to optimize your fitness journey
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {features.map((feature) => (
              <Card key={feature.id} className="relative">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    {feature.icon}
                    {getStatusIcon(feature.status)}
                  </div>
                  <h3 className="font-medium text-sm mb-1">{feature.name}</h3>
                  <p className="text-xs text-muted-foreground mb-3">
                    {feature.description}
                  </p>
                  {feature.confidence > 0 && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span>Confidence</span>
                        <span>{Math.round(feature.confidence * 100)}%</span>
                      </div>
                      <Progress value={feature.confidence * 100} className="h-1" />
                    </div>
                  )}
                  <Badge 
                    variant={feature.status === 'active' ? 'default' : 'secondary'}
                    className="mt-2 text-xs"
                  >
                    {feature.status}
                  </Badge>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* AI Features Tabs */}
      <Tabs defaultValue="recommendations" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="recommendations">Workouts</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="nutrition">Nutrition</TabsTrigger>
          <TabsTrigger value="form">Form Check</TabsTrigger>
        </TabsList>

        <TabsContent value="recommendations">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                AI Workout Recommendations
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={generateWorkoutRecommendations}
                disabled={features.find(f => f.id === 'workout-recommendations')?.status === 'loading'}
                className="flex items-center gap-2"
              >
                <Zap className="h-4 w-4" />
                Generate Smart Workout
              </Button>
              
              {recommendations && (
                <div className="space-y-4">
                  <div className="p-4 border rounded-lg">
                    <h3 className="font-semibold text-lg mb-2">{recommendations.workoutName}</h3>
                    <p className="text-muted-foreground mb-3">{recommendations.description}</p>
                    
                    <div className="grid md:grid-cols-3 gap-4 mb-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold">{recommendations.duration}</div>
                        <div className="text-sm text-muted-foreground">Minutes</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold capitalize">{recommendations.intensity}</div>
                        <div className="text-sm text-muted-foreground">Intensity</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold">{recommendations.exercises?.length || 0}</div>
                        <div className="text-sm text-muted-foreground">Exercises</div>
                      </div>
                    </div>
                    
                    <div className="text-sm text-muted-foreground">
                      <strong>Reasoning:</strong> {recommendations.reasoning}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Predictive Health Analytics
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={generatePredictiveAnalytics}
                disabled={features.find(f => f.id === 'predictive-analytics')?.status === 'loading'}
                className="flex items-center gap-2"
              >
                <Activity className="h-4 w-4" />
                Generate Analytics
              </Button>
              
              {analytics && (
                <div className="space-y-4">
                  <div className="grid md:grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Injury Risk</h4>
                        <Badge variant={analytics.riskAssessment?.injuryRisk?.level === 'low' ? 'default' : 'destructive'}>
                          {analytics.riskAssessment?.injuryRisk?.level || 'Unknown'}
                        </Badge>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Burnout Risk</h4>
                        <Badge variant={analytics.riskAssessment?.burnoutRisk?.level === 'low' ? 'default' : 'secondary'}>
                          {analytics.riskAssessment?.burnoutRisk?.level || 'Unknown'}
                        </Badge>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Overall Confidence</h4>
                        <div className="text-2xl font-bold">
                          {Math.round((analytics.confidence?.overall || 0) * 100)}%
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="nutrition">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Utensils className="h-5 w-5" />
                AI Nutrition Assistant
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={generateNutritionPlan}
                disabled={features.find(f => f.id === 'nutrition-ai')?.status === 'loading'}
                className="flex items-center gap-2"
              >
                <Utensils className="h-4 w-4" />
                Generate Meal Plan
              </Button>
              
              {nutritionPlan && (
                <div className="space-y-4">
                  <div className="grid md:grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Daily Calories</h4>
                        <div className="text-2xl font-bold">
                          {nutritionPlan.dailyNutritionSummary?.totalCalories || 0}
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Protein</h4>
                        <div className="text-2xl font-bold">
                          {nutritionPlan.dailyNutritionSummary?.macroBreakdown?.protein?.grams || 0}g
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Supplements</h4>
                        <div className="text-2xl font-bold">
                          {nutritionPlan.supplementRecommendations?.length || 0}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="form">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Exercise Form Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center p-8 border-2 border-dashed border-muted rounded-lg">
                <Camera className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-medium mb-2">Upload Exercise Video/Photo</h3>
                <p className="text-muted-foreground mb-4">
                  AI will analyze your form and provide detailed feedback
                </p>
                <Button variant="outline" onClick={() => console.log("AIHub button clicked")}>
                  Upload Media
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AIHub;