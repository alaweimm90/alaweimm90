import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { 
  Play, 
  Pause, 
  Square, 
  Timer, 
  Zap, 
  Target,
  TrendingUp,
  Smartphone,
  Wifi,
  WifiOff,
  ChevronLeft,
  ChevronRight,
  SkipForward
} from 'lucide-react';
import { useMobileCapacitor } from '@/hooks/useMobileCapacitor';
import { useGestures } from '@/hooks/useGestures';
import { useOfflineSupport } from '@/hooks/useOfflineSupport';

interface Exercise {
  id: string;
  name: string;
  sets: number;
  reps: string;
  weight?: string;
  restTime: number;
  instructions: string[];
}

interface MobileWorkoutInterfaceProps {
  exercises: Exercise[];
  onComplete: () => void;
  isLiveCoaching?: boolean;
}

const MobileWorkoutInterface: React.FC<MobileWorkoutInterfaceProps> = ({
  exercises,
  onComplete,
  isLiveCoaching = false
}) => {
  const { hapticFeedback, networkStatus, deviceInfo, isNative } = useMobileCapacitor();
  const { isOnline, cacheWorkout, addToPendingSync } = useOfflineSupport();
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [currentSet, setCurrentSet] = useState(1);
  const [isResting, setIsResting] = useState(false);
  const [restTimeLeft, setRestTimeLeft] = useState(0);
  const [workoutStarted, setWorkoutStarted] = useState(false);
  const [completedSets, setCompletedSets] = useState<Record<string, number>>({});
  const [workoutData, setWorkoutData] = useState<Array<{ exercise: string; sets: number; reps: number; weight: number }>>([]);

  // Gesture handlers
  const gestureHandlers = {
    onSwipeLeft: () => nextExercise(),
    onSwipeRight: () => previousExercise(),
    onSwipeUp: () => !isResting && completeSet(),
    onSwipeDown: () => isResting && skipRest(),
    onDoubleTap: () => !isResting && completeSet(),
    onLongPress: () => isResting && skipRest()
  };

  const { elementRef } = useGestures(gestureHandlers);

  const currentExercise = exercises[currentExerciseIndex];
  const progress = ((currentExerciseIndex + (currentSet / currentExercise?.sets || 1)) / exercises.length) * 100;

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isResting && restTimeLeft > 0) {
      timer = setTimeout(() => {
        setRestTimeLeft(restTimeLeft - 1);
        if (restTimeLeft === 1) {
          hapticFeedback.success();
        } else if (restTimeLeft <= 5) {
          hapticFeedback.light();
        }
      }, 1000);
    }
    return () => clearTimeout(timer);
  }, [isResting, restTimeLeft, hapticFeedback]);

  const startWorkout = () => {
    setWorkoutStarted(true);
    hapticFeedback.medium();
  };

  const nextExercise = () => {
    if (currentExerciseIndex < exercises.length - 1) {
      setCurrentExerciseIndex(currentExerciseIndex + 1);
      setCurrentSet(1);
      setIsResting(false);
      hapticFeedback.light();
    }
  };

  const previousExercise = () => {
    if (currentExerciseIndex > 0) {
      setCurrentExerciseIndex(currentExerciseIndex - 1);
      setCurrentSet(1);
      setIsResting(false);
      hapticFeedback.light();
    }
  };

  const completeSet = () => {
    const exerciseId = currentExercise.id;
    const newCompletedSets = { ...completedSets };
    newCompletedSets[exerciseId] = (newCompletedSets[exerciseId] || 0) + 1;
    setCompletedSets(newCompletedSets);
    
    // Log set completion
    const setData = {
      exerciseId,
      exerciseName: currentExercise.name,
      setNumber: currentSet,
      reps: currentExercise.reps,
      weight: currentExercise.weight,
      completedAt: new Date()
    };
    setWorkoutData(prev => [...prev, setData]);
    
    hapticFeedback.success();

    if (currentSet < currentExercise.sets) {
      // Start rest period
      setIsResting(true);
      setRestTimeLeft(currentExercise.restTime);
      setCurrentSet(currentSet + 1);
    } else {
      // Move to next exercise
      if (currentExerciseIndex < exercises.length - 1) {
        setCurrentExerciseIndex(currentExerciseIndex + 1);
        setCurrentSet(1);
        setIsResting(false);
      } else {
        // Workout complete
        completeWorkout();
      }
    }
  };

  const completeWorkout = () => {
    hapticFeedback.heavy();
    
    const fullWorkoutData = {
      exercises: workoutData,
      completedAt: new Date(),
      totalSets: workoutData.length,
      duration: Date.now() - (workoutData[0]?.completedAt || Date.now())
    };

    // Save offline if needed
    if (!isOnline) {
      cacheWorkout(fullWorkoutData);
      addToPendingSync({ type: 'workout_log', data: fullWorkoutData });
    }

    onComplete();
  };

  const skipRest = () => {
    setIsResting(false);
    setRestTimeLeft(0);
    hapticFeedback.light();
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!workoutStarted) {
  return (
    <div 
      ref={elementRef}
      className="min-h-screen bg-background p-4 flex flex-col"
    >
      {/* Device Info Bar */}
      {isNative && (
        <div className="flex items-center justify-between mb-4 p-2 bg-card rounded-lg">
          <div className="flex items-center gap-2">
            <Smartphone className="h-4 w-4" />
            <span className="text-sm text-muted-foreground">
              {deviceInfo?.platform} {deviceInfo?.osVersion}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {isOnline ? (
              <>
                <Wifi className="h-4 w-4 text-green-500" />
                <span className="text-sm text-green-500">Online</span>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4 text-red-500" />
                <span className="text-sm text-red-500">Offline</span>
              </>
            )}
          </div>
        </div>
      )}

        {/* Workout Overview */}
        <Card className="flex-1 flex flex-col">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl mb-2">Ready to Train?</CardTitle>
            <div className="flex justify-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <Target className="h-4 w-4" />
                {exercises.length} Exercises
              </div>
              <div className="flex items-center gap-1">
                <TrendingUp className="h-4 w-4" />
                {exercises.reduce((total, ex) => total + ex.sets, 0)} Sets
              </div>
              {isLiveCoaching && (
                <div className="flex items-center gap-1">
                  <Zap className="h-4 w-4 text-primary" />
                  Live AI Coach
                </div>
              )}
            </div>
          </CardHeader>
          
          <CardContent className="flex-1 space-y-4">
            {exercises.map((exercise, index) => (
              <div key={exercise.id} className="p-3 border rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-medium">{exercise.name}</h3>
                  <Badge variant="secondary">
                    {exercise.sets} × {exercise.reps}
                  </Badge>
                </div>
                {exercise.weight && (
                  <p className="text-sm text-muted-foreground mb-1">
                    Weight: {exercise.weight}
                  </p>
                )}
                <p className="text-sm text-muted-foreground">
                  Rest: {exercise.restTime}s
                </p>
              </div>
            ))}
          </CardContent>

          <div className="p-6">
            <Button 
              onClick={startWorkout} 
              className="w-full h-12 text-lg"
              size="lg"
            >
              <Play className="mr-2 h-5 w-5" />
              Start Workout
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div 
      ref={elementRef}
      className="min-h-screen bg-background p-4 flex flex-col"
    >
      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium">Workout Progress</span>
          <div className="flex items-center gap-2">
            {!isOnline && (
              <Badge variant="outline" className="text-xs">
                Offline
              </Badge>
            )}
            <span className="text-sm text-muted-foreground">
              {Math.round(progress)}%
            </span>
          </div>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {/* Exercise Navigation */}
      <div className="flex items-center justify-between mb-4">
        <Button 
          variant="outline" 
          onClick={previousExercise}
          disabled={currentExerciseIndex === 0}
          className="p-2"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        
        <div className="text-center">
          <p className="text-sm text-muted-foreground">Exercise</p>
          <p className="font-medium">
            {currentExerciseIndex + 1} of {exercises.length}
          </p>
        </div>
        
        <Button 
          variant="outline" 
          onClick={nextExercise}
          disabled={currentExerciseIndex === exercises.length - 1}
          className="p-2"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {/* Rest Timer */}
      {isResting && (
        <Card className="mb-6 bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
          <CardContent className="p-6 text-center">
            <Timer className="h-8 w-8 mx-auto mb-3 text-blue-600" />
            <h3 className="text-xl font-semibold mb-2">Rest Time</h3>
            <div className="text-3xl font-mono font-bold text-blue-600 mb-4">
              {formatTime(restTimeLeft)}
            </div>
            <div className="flex gap-2 justify-center">
              <Button variant="outline" onClick={skipRest}>
                <SkipForward className="mr-2 h-4 w-4" />
                Skip Rest
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Swipe down or long press to skip
            </p>
          </CardContent>
        </Card>
      )}

      {/* Current Exercise */}
      {!isResting && currentExercise && (
        <Card className="flex-1 flex flex-col">
          <CardHeader>
            <div className="flex justify-between items-start mb-2">
              <CardTitle className="text-xl">{currentExercise.name}</CardTitle>
              <Badge variant="default">
                Exercise {currentExerciseIndex + 1}/{exercises.length}
              </Badge>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">
                Set {currentSet} of {currentExercise.sets}
              </div>
              <div className="text-lg text-muted-foreground">
                {currentExercise.reps} reps
                {currentExercise.weight && ` @ ${currentExercise.weight}`}
              </div>
            </div>
          </CardHeader>

          <CardContent className="flex-1">
            <div className="space-y-3">
              <h4 className="font-medium">Instructions:</h4>
              {currentExercise.instructions.map((instruction, index) => (
                <div key={index} className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 flex items-center justify-center text-sm font-medium">
                    {index + 1}
                  </span>
                  <p className="text-sm">{instruction}</p>
                </div>
              ))}
            </div>

            {/* Set Completion Indicator */}
            <div className="mt-6 p-4 bg-muted rounded-lg">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Sets Completed:</span>
                <div className="flex gap-1">
                  {Array.from({ length: currentExercise.sets }, (_, i) => (
                    <div
                      key={i}
                      className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-sm font-medium ${
                        i < currentSet - 1
                          ? 'bg-green-500 border-green-500 text-white'
                          : i === currentSet - 1
                          ? 'bg-primary border-primary text-primary-foreground'
                          : 'border-muted-foreground/30'
                      }`}
                    >
                      {i + 1}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>

          <div className="p-6">
            <Button 
              onClick={completeSet} 
              className="w-full h-12 text-lg"
              size="lg"
            >
              Complete Set
            </Button>
            <p className="text-xs text-center text-muted-foreground mt-2">
              Swipe up or double tap to complete • Swipe left/right to navigate
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};

export default MobileWorkoutInterface;