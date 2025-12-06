import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/ui/atoms/Input';
import { Slider } from '@/components/ui/slider';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/contexts/AuthContext';
import { 
  Play, 
  Pause, 
  Square, 
  Timer, 
  Activity, 
  Zap, 
  Target,
  Heart,
  Camera,
  MessageSquare,
  TrendingUp,
  RotateCcw,
  CheckCircle,
  AlertTriangle,
  Volume2,
  VolumeX
} from 'lucide-react';

interface Exercise {
  id: string;
  name: string;
  sets: number;
  reps: string;
  weight: string;
  restTime: number;
  category: 'compound' | 'isolation' | 'cardio';
}

interface WorkoutSession {
  id: string;
  name: string;
  exercises: Exercise[];
  estimatedDuration: number;
}

interface LiveWorkoutState {
  isActive: boolean;
  currentExercise: number;
  currentSet: number;
  isResting: boolean;
  restTimeRemaining: number;
  sessionStartTime?: Date;
  completedExercises: string[];
  formRatings: number[];
  heartRate?: number;
}

interface CoachingMessage {
  type: 'motivation' | 'form' | 'warning' | 'achievement';
  message: string;
  timestamp: Date;
  severity: 'info' | 'warning' | 'success' | 'error';
}

interface WorkoutSummary {
  totalTime: number;
  exercisesCompleted: number;
  totalVolume: number;
  averageFormRating: number;
  completedExercises: Array<{ name: string; sets: number; reps: number; weight: number }>;
}

interface WebSocketMessage {
  type: string;
  data?: {
    category?: string;
    message?: string;
    severity?: CoachingMessage['severity'];
    feedback?: string;
  };
}

interface FormAnalysisData {
  confidence: number;
  corrections: string[];
  analysis: {
    score: number;
  };
}

interface LiveWorkoutTrackerProps {
  workoutSession: WorkoutSession;
  onComplete?: (summary: WorkoutSummary) => void;
}

export const LiveWorkoutTracker: React.FC<LiveWorkoutTrackerProps> = ({ 
  workoutSession,
  onComplete 
}) => {
  const { user } = useAuth();
  const { toast } = useToast();
  
  const [workoutState, setWorkoutState] = useState<LiveWorkoutState>({
    isActive: false,
    currentExercise: 0,
    currentSet: 1,
    isResting: false,
    restTimeRemaining: 0,
    completedExercises: [],
    formRatings: []
  });
  
  const [coachingMessages, setCoachingMessages] = useState<CoachingMessage[]>([]);
  const [currentReps, setCurrentReps] = useState('');
  const [currentWeight, setCurrentWeight] = useState('');
  const [formRating, setFormRating] = useState([7]);
  const [isSoundEnabled, setIsSoundEnabled] = useState(true);
  const [isCoachingEnabled, setIsCoachingEnabled] = useState(true);
  
  const websocketRef = useRef<WebSocket | null>(null);
  const restTimerRef = useRef<NodeJS.Timeout | null>(null);
  const speechSynthRef = useRef<SpeechSynthesisUtterance | null>(null);

  const currentExercise = workoutSession.exercises[workoutState.currentExercise];
  const totalExercises = workoutSession.exercises.length;
  const progressPercentage = ((workoutState.currentExercise + (workoutState.currentSet / currentExercise?.sets || 1)) / totalExercises) * 100;

  useEffect(() => {
    if (workoutState.isActive && !websocketRef.current) {
      connectToLiveSession();
    }
    
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
      if (restTimerRef.current) {
        clearInterval(restTimerRef.current);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workoutState.isActive]);

  const connectToLiveSession = useCallback(async () => {
    try {
      const wsUrl = `wss://lvmcumsfpjjcgnnovvzs.supabase.co/functions/v1/live-workout-coaching?token=${encodeURIComponent('bearer ' + await getAuthToken())}`;
      
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onopen = () => {
        console.log('Connected to live workout session');
        toast({
          title: "Live Session Active",
          description: "Real-time coaching enabled",
        });
      };
      
      websocketRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };
      
      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        toast({
          title: "Connection Error",
          description: "Live coaching temporarily unavailable",
          variant: "destructive"
        });
      };
      
    } catch (error) {
      console.error('Failed to connect to live session:', error);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getAuthToken = async () => {
    // This would get the actual auth token from Supabase
    return 'dummy-token'; // Placeholder
  };

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'coaching_message':
        addCoachingMessage({
          type: message.data.category,
          message: message.data.message,
          timestamp: new Date(),
          severity: message.data.severity
        });
        
        if (isSoundEnabled && isCoachingEnabled) {
          speakMessage(message.data.message);
        }
        break;
        
      case 'exercise_feedback':
        addCoachingMessage({
          type: 'form',
          message: message.data.feedback,
          timestamp: new Date(),
          severity: 'info'
        });
        break;
        
      case 'form_analysis':
        handleFormAnalysis(message.data);
        break;
        
      case 'rest_motivation':
        if (workoutState.isResting) {
          addCoachingMessage({
            type: 'motivation',
            message: message.data.message,
            timestamp: new Date(),
            severity: 'info'
          });
        }
        break;
        
      case 'workout_summary':
        if (onComplete) {
          onComplete(message.data);
        }
        break;
    }
  };

  const addCoachingMessage = (message: CoachingMessage) => {
    setCoachingMessages(prev => [message, ...prev].slice(0, 10)); // Keep last 10 messages
  };

  const speakMessage = (text: string) => {
    if (!isSoundEnabled || !window.speechSynthesis) return;
    
    window.speechSynthesis.cancel(); // Cancel any ongoing speech
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 0.7;
    
    speechSynthRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  };

  const startWorkout = () => {
    setWorkoutState(prev => ({
      ...prev,
      isActive: true,
      sessionStartTime: new Date()
    }));

    // Send workout start message via WebSocket
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'workout_start',
        data: {
          workoutPlanId: workoutSession.id,
          exerciseCount: totalExercises,
          estimatedDuration: workoutSession.estimatedDuration
        }
      }));
    }

    addCoachingMessage({
      type: 'motivation',
      message: `Let's start your ${workoutSession.name} workout! Focus on form and have fun!`,
      timestamp: new Date(),
      severity: 'success'
    });
  };

  const completeSet = () => {
    if (!currentExercise || !currentReps || !currentWeight) {
      toast({
        title: "Missing Information",
        description: "Please enter reps and weight for this set",
        variant: "destructive"
      });
      return;
    }

    const exerciseData = {
      exerciseName: currentExercise.name,
      sets: workoutState.currentSet,
      reps: currentReps,
      weight: currentWeight,
      formRating: formRating[0],
      exerciseType: currentExercise.category,
      intensity: formRating[0] >= 8 ? 8 : 6
    };

    // Send to WebSocket for AI feedback
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'exercise_complete',
        data: exerciseData
      }));
    }

    // Update form ratings
    setWorkoutState(prev => ({
      ...prev,
      formRatings: [...prev.formRatings, formRating[0]]
    }));

    if (workoutState.currentSet >= currentExercise.sets) {
      // Move to next exercise
      completeExercise();
    } else {
      // Start rest period
      startRestPeriod(currentExercise.restTime);
    }

    // Reset inputs
    setCurrentReps('');
    setCurrentWeight('');
    setFormRating([7]);
  };

  const startRestPeriod = (restTime: number) => {
    setWorkoutState(prev => ({
      ...prev,
      isResting: true,
      restTimeRemaining: restTime
    }));

    // Send rest timer message
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'rest_timer',
        data: {
          restTime,
          exerciseName: currentExercise.name
        }
      }));
    }

    restTimerRef.current = setInterval(() => {
      setWorkoutState(prev => {
        if (prev.restTimeRemaining <= 1) {
          clearInterval(restTimerRef.current!);
          return {
            ...prev,
            isResting: false,
            restTimeRemaining: 0,
            currentSet: prev.currentSet + 1
          };
        }
        return {
          ...prev,
          restTimeRemaining: prev.restTimeRemaining - 1
        };
      });
    }, 1000);

    addCoachingMessage({
      type: 'motivation',
      message: `Great set! Rest for ${restTime} seconds and prepare for the next one.`,
      timestamp: new Date(),
      severity: 'success'
    });
  };

  const completeExercise = () => {
    setWorkoutState(prev => ({
      ...prev,
      completedExercises: [...prev.completedExercises, currentExercise.id],
      currentExercise: prev.currentExercise + 1,
      currentSet: 1
    }));

    if (workoutState.currentExercise + 1 >= totalExercises) {
      completeWorkout();
    } else {
      addCoachingMessage({
        type: 'achievement',
        message: `${currentExercise.name} completed! Moving to next exercise.`,
        timestamp: new Date(),
        severity: 'success'
      });
    }
  };

  const completeWorkout = () => {
    const duration = workoutState.sessionStartTime 
      ? Math.floor((Date.now() - workoutState.sessionStartTime.getTime()) / 1000 / 60)
      : workoutSession.estimatedDuration;

    const workoutData = {
      duration,
      exercisesCompleted: workoutState.completedExercises.length,
      totalSets: workoutState.formRatings.length,
      averageFormRating: workoutState.formRatings.reduce((a, b) => a + b, 0) / workoutState.formRatings.length,
      performanceScore: calculatePerformanceScore(),
      plannedExercises: totalExercises,
      plannedDuration: workoutSession.estimatedDuration
    };

    // Send completion message
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'workout_complete',
        data: workoutData
      }));
    }

    setWorkoutState(prev => ({ ...prev, isActive: false }));

    addCoachingMessage({
      type: 'achievement',
      message: `Workout completed! Amazing effort today!`,
      timestamp: new Date(),
      severity: 'success'
    });
  };

  const calculatePerformanceScore = () => {
    const formAverage = workoutState.formRatings.reduce((a, b) => a + b, 0) / workoutState.formRatings.length;
    const completionRate = workoutState.completedExercises.length / totalExercises;
    return Math.round((formAverage + (completionRate * 10)) / 2);
  };

  const requestCoachHelp = () => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'coach_request',
        data: {
          exerciseName: currentExercise?.name,
          issue: 'form_check',
          urgency: 'normal'
        }
      }));
    }

    addCoachingMessage({
      type: 'motivation',
      message: "Coach has been notified! Keep going, help is on the way.",
      timestamp: new Date(),
      severity: 'info'
    });
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleFormAnalysis = (data: FormAnalysisData) => {
    if (data.confidence > 0.7 && data.corrections.length > 0) {
      addCoachingMessage({
        type: 'form',
        message: `Form check: ${data.corrections.join('. ')}`,
        timestamp: new Date(),
        severity: data.analysis.score < 7 ? 'warning' : 'info'
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Workout Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                {workoutSession.name}
              </CardTitle>
              <CardDescription>
                Live workout session with AI coaching
              </CardDescription>
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsSoundEnabled(!isSoundEnabled)}
              >
                {isSoundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsCoachingEnabled(!isCoachingEnabled)}
              >
                <MessageSquare className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Progress</span>
              <span className="text-sm font-medium">
                {workoutState.currentExercise + 1} of {totalExercises} exercises
              </span>
            </div>
            
            <Progress value={progressPercentage} className="h-2" />
            
            {!workoutState.isActive ? (
              <Button onClick={startWorkout} className="w-full" size="lg">
                <Play className="h-5 w-5 mr-2" />
                Start Workout
              </Button>
            ) : (
              <div className="flex gap-2">
                <Button onClick={completeWorkout} variant="destructive" className="flex-1">
                  <Square className="h-4 w-4 mr-2" />
                  End Workout
                </Button>
                <Button onClick={requestCoachHelp} variant="outline">
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Request Help
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {workoutState.isActive && currentExercise && (
        <>
          {/* Current Exercise */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                {currentExercise.name}
              </CardTitle>
              <CardDescription>
                Set {workoutState.currentSet} of {currentExercise.sets}
                {workoutState.isResting && (
                  <Badge variant="secondary" className="ml-2">
                    <Timer className="h-3 w-3 mr-1" />
                    Rest: {formatTime(workoutState.restTimeRemaining)}
                  </Badge>
                )}
              </CardDescription>
            </CardHeader>
            
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Reps Completed</label>
                  <Input
                    type="number"
                    value={currentReps}
                    onChange={(e) => setCurrentReps(e.target.value)}
                    placeholder={currentExercise.reps}
                    disabled={workoutState.isResting}
                  />
                </div>
                
                <div>
                  <label className="text-sm font-medium">Weight Used</label>
                  <Input
                    value={currentWeight}
                    onChange={(e) => setCurrentWeight(e.target.value)}
                    placeholder={currentExercise.weight}
                    disabled={workoutState.isResting}
                  />
                </div>
              </div>
              
              <div>
                <label className="text-sm font-medium">Form Rating (1-10)</label>
                <div className="mt-2">
                  <Slider
                    value={formRating}
                    onValueChange={setFormRating}
                    max={10}
                    min={1}
                    step={1}
                    disabled={workoutState.isResting}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>Poor</span>
                    <span className="font-medium">{formRating[0]}/10</span>
                    <span>Perfect</span>
                  </div>
                </div>
              </div>
              
              {!workoutState.isResting && (
                <Button onClick={completeSet} className="w-full">
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Complete Set
                </Button>
              )}
            </CardContent>
          </Card>

          {/* Live Coaching Messages */}
          {isCoachingEnabled && coachingMessages.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">AI Coach</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-60 overflow-y-auto">
                  {coachingMessages.map((message, index) => (
                    <div 
                      key={index}
                      className={`p-3 rounded-lg border-l-4 ${
                        message.severity === 'success' ? 'border-green-500 bg-green-50' :
                        message.severity === 'warning' ? 'border-yellow-500 bg-yellow-50' :
                        message.severity === 'error' ? 'border-red-500 bg-red-50' :
                        'border-blue-500 bg-blue-50'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {message.type === 'motivation' && <Zap className="h-4 w-4 text-blue-500 mt-0.5" />}
                        {message.type === 'form' && <Camera className="h-4 w-4 text-repz-orange mt-0.5" />}
                        {message.type === 'achievement' && <TrendingUp className="h-4 w-4 text-green-500 mt-0.5" />}
                        {message.type === 'warning' && <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />}
                        
                        <div className="flex-1">
                          <p className="text-sm">{message.message}</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {message.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
};