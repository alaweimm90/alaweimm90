import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { TierGate } from '@/components/auth/TierGate';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';
import { 
  Play, 
  Pause, 
  SkipForward, 
  RotateCcw, 
  Timer,
  Activity,
  Target,
  CheckCircle
} from 'lucide-react';

interface Exercise {
  id: string;
  name: string;
  sets: number;
  reps: string;
  restTime: number;
  instructions: string[];
  formCues: string[];
  motivationalCues: string[];
}

interface WorkoutProgram {
  id: string;
  name: string;
  description: string;
  estimatedDuration: number;
  exercises: Exercise[];
}

interface WorkoutSession {
  currentExercise: number;
  currentSet: number;
  isResting: boolean;
  restTimeRemaining: number;
  completedExercises: string[];
  startTime: Date;
}

const SAMPLE_WORKOUT: WorkoutProgram = {
  id: 'upper_strength',
  name: 'Upper Body Strength Training',
  description: 'Comprehensive upper body workout focusing on strength and muscle development',
  estimatedDuration: 45,
  exercises: [
    {
      id: 'push_ups',
      name: 'Push-ups',
      sets: 3,
      reps: '8-12',
      restTime: 90,
      instructions: [
        'Start in a plank position with hands shoulder-width apart',
        'Lower your body until chest nearly touches the floor',
        'Push back up to starting position with control'
      ],
      formCues: [
        'Keep your core tight throughout the movement',
        'Maintain a straight line from head to heels',
        'Control the descent - don\'t drop down quickly'
      ],
      motivationalCues: [
        'You\'ve got this! Each rep makes you stronger!',
        'Feel those muscles working - this is where growth happens!',
        'Push through the burn - that\'s your body adapting!'
      ]
    },
    {
      id: 'pull_ups',
      name: 'Pull-ups',
      sets: 3,
      reps: '5-8',
      restTime: 120,
      instructions: [
        'Hang from the bar with palms facing away',
        'Pull your body up until chin clears the bar',
        'Lower yourself with control to full extension'
      ],
      formCues: [
        'Engage your lats and pull with your back muscles',
        'Avoid swinging or using momentum',
        'Full range of motion on every rep'
      ],
      motivationalCues: [
        'Every pull-up is a victory - you\'re incredibly strong!',
        'Your back is getting more powerful with each rep!',
        'Show that bar who\'s boss!'
      ]
    },
    {
      id: 'shoulder_press',
      name: 'Overhead Press',
      sets: 3,
      reps: '10-12',
      restTime: 90,
      instructions: [
        'Stand with feet shoulder-width apart',
        'Press weights overhead until arms are fully extended',
        'Lower weights back to shoulder level with control'
      ],
      formCues: [
        'Keep your core braced throughout the movement',
        'Press straight up, not forward',
        'Don\'t arch your back excessively'
      ],
      motivationalCues: [
        'Build those powerful shoulders - you\'re unstoppable!',
        'Press through any doubt - you\'re stronger than you think!',
        'Every rep is building the new, stronger you!'
      ]
    }
  ]
};

export default function VoiceWorkoutGuide() {
  const { toast } = useToast();
  const [workout] = useState<WorkoutProgram>(SAMPLE_WORKOUT);
  const [session, setSession] = useState<WorkoutSession | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [restTimer, setRestTimer] = useState<NodeJS.Timeout | null>(null);

  // Start workout session
  const startWorkout = useCallback(async () => {
    const newSession: WorkoutSession = {
      currentExercise: 0,
      currentSet: 1,
      isResting: false,
      restTimeRemaining: 0,
      completedExercises: [],
      startTime: new Date()
    };

    setSession(newSession);
    setIsActive(true);

    // Generate welcome message
    await generateVoiceMessage(
      `Welcome to your ${workout.name}! We'll be working through ${workout.exercises.length} exercises today. Let's start strong with ${workout.exercises[0].name}!`,
      'motivational'
    );

    // Provide exercise instructions
    const firstExercise = workout.exercises[0];
    await generateVoiceMessage(
      `First up: ${firstExercise.name}. ${firstExercise.instructions.join(' ')}`,
      'technical'
    );

    toast({
      title: "Workout Started",
      description: `${workout.name} - Let's do this!`
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workout]);

  // Complete current set
  const completeSet = useCallback(async () => {
    if (!session || !isActive) return;

    const currentExercise = workout.exercises[session.currentExercise];
    const isLastSet = session.currentSet >= currentExercise.sets;
    const isLastExercise = session.currentExercise >= workout.exercises.length - 1;

    if (isLastSet) {
      // Move to next exercise or complete workout
      if (isLastExercise) {
        // Workout complete
        await generateVoiceMessage(
          "Outstanding work! You've completed your entire workout. You should be proud of your dedication and effort today!",
          'supportive'
        );
        
        setIsActive(false);
        setSession(null);
        
        toast({
          title: "Workout Complete!",
          description: "Congratulations on finishing your session!"
        });
        return;
      } else {
        // Move to next exercise
        const nextExercise = workout.exercises[session.currentExercise + 1];
        
        setSession(prev => prev ? {
          ...prev,
          currentExercise: prev.currentExercise + 1,
          currentSet: 1,
          isResting: true,
          restTimeRemaining: 180, // 3 minutes between exercises
          completedExercises: [...prev.completedExercises, currentExercise.id]
        } : null);

        await generateVoiceMessage(
          `Excellent work on ${currentExercise.name}! Take a 3-minute rest before we move to ${nextExercise.name}.`,
          'supportive'
        );

        startRestTimer(180);
        return;
      }
    } else {
      // Start rest period between sets
      setSession(prev => prev ? {
        ...prev,
        currentSet: prev.currentSet + 1,
        isResting: true,
        restTimeRemaining: currentExercise.restTime
      } : null);

      await generateVoiceMessage(
        `Great set! Rest for ${currentExercise.restTime} seconds. ${currentExercise.motivationalCues[Math.floor(Math.random() * currentExercise.motivationalCues.length)]}`,
        'motivational'
      );

      startRestTimer(currentExercise.restTime);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, isActive, workout]);

  // Start rest timer
  const startRestTimer = useCallback((duration: number) => {
    if (restTimer) clearInterval(restTimer);

    const timer = setInterval(() => {
      setSession(prev => {
        if (!prev || prev.restTimeRemaining <= 1) {
          clearInterval(timer);
          return prev ? { ...prev, isResting: false, restTimeRemaining: 0 } : null;
        }
        return { ...prev, restTimeRemaining: prev.restTimeRemaining - 1 };
      });
    }, 1000);

    setRestTimer(timer);

    // Notify when rest is complete
    setTimeout(async () => {
      if (session && isActive) {
        const currentExercise = workout.exercises[session.currentExercise];
        await generateVoiceMessage(
          `Rest time is up! Ready for set ${session.currentSet} of ${currentExercise.name}? ${currentExercise.formCues[Math.floor(Math.random() * currentExercise.formCues.length)]}`,
          'technical'
        );
      }
    }, duration * 1000);
  }, [restTimer, session, isActive, workout, generateVoiceMessage]);

  // Generate voice coaching message
  const generateVoiceMessage = useCallback(async (text: string, personality: 'motivational' | 'technical' | 'supportive' | 'intense' = 'motivational') => {
    try {
      const { data, error } = await supabase.functions.invoke('elevenlabs-tts', {
        body: { 
          text, 
          coachPersonality: personality
        }
      });

      if (error) throw error;

      // Play audio
      const audio = new Audio(`data:audio/mpeg;base64,${data.audioContent}`);
      audio.volume = 0.8;
      audio.play().catch(console.error);

    } catch (error) {
      console.error('Error generating voice message:', error);
    }
  }, []);

  // Provide form cue
  const provideFormCue = useCallback(async () => {
    if (!session || !isActive) return;

    const currentExercise = workout.exercises[session.currentExercise];
    const randomCue = currentExercise.formCues[Math.floor(Math.random() * currentExercise.formCues.length)];
    
    await generateVoiceMessage(randomCue, 'technical');
  }, [session, isActive, workout, generateVoiceMessage]);

  // Provide motivation
  const provideMotivation = useCallback(async () => {
    if (!session || !isActive) return;

    const currentExercise = workout.exercises[session.currentExercise];
    const randomMotivation = currentExercise.motivationalCues[Math.floor(Math.random() * currentExercise.motivationalCues.length)];
    
    await generateVoiceMessage(randomMotivation, 'motivational');
  }, [session, isActive, workout, generateVoiceMessage]);

  // Calculate workout progress
  const getWorkoutProgress = useCallback(() => {
    if (!session) return 0;
    
    const totalSets = workout.exercises.reduce((total, exercise) => total + exercise.sets, 0);
    const completedSets = session.completedExercises.length * workout.exercises.find(e => e.id === session.completedExercises[0])?.sets || 0;
    const currentExerciseSets = session.currentSet - 1;
    
    return ((completedSets + currentExerciseSets) / totalSets) * 100;
  }, [session, workout]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (restTimer) clearInterval(restTimer);
    };
  }, [restTimer]);

  const currentExercise = session ? workout.exercises[session.currentExercise] : null;

  return (
    <TierGate requiredTier="performance" feature="voice_workout_guide">
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Activity className="h-6 w-6" />
            Voice-Guided Workout
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {!isActive ? (
            // Workout Preview
            <div className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">{workout.name}</h3>
                <p className="text-muted-foreground">{workout.description}</p>
                <div className="flex items-center gap-4 text-sm">
                  <Badge variant="outline">
                    <Timer className="h-3 w-3 mr-1" />
                    {workout.estimatedDuration} min
                  </Badge>
                  <Badge variant="outline">
                    <Target className="h-3 w-3 mr-1" />
                    {workout.exercises.length} exercises
                  </Badge>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-medium">Exercises</h4>
                {workout.exercises.map((exercise, index) => (
                  <div key={exercise.id} className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                    <div className="flex-shrink-0 w-8 h-8 bg-primary rounded-full flex items-center justify-center text-primary-foreground text-sm font-medium">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <h5 className="font-medium">{exercise.name}</h5>
                      <p className="text-sm text-muted-foreground">
                        {exercise.sets} sets × {exercise.reps} reps
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <Button onClick={startWorkout} className="w-full" size="lg">
                <Play className="h-4 w-4 mr-2" />
                Start Workout
              </Button>
            </div>
          ) : (
            // Active Workout
            <div className="space-y-6">
              {/* Progress */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Workout Progress</span>
                  <span>{Math.round(getWorkoutProgress())}%</span>
                </div>
                <Progress value={getWorkoutProgress()} className="w-full" />
              </div>

              {/* Current Exercise */}
              {currentExercise && (
                <div className="space-y-4">
                  <div className="text-center space-y-2">
                    <h3 className="text-xl font-semibold">{currentExercise.name}</h3>
                    <div className="flex justify-center gap-4">
                      <Badge variant="default">
                        Set {session.currentSet} of {currentExercise.sets}
                      </Badge>
                      <Badge variant="outline">
                        {currentExercise.reps} reps
                      </Badge>
                    </div>
                  </div>

                  {session.isResting ? (
                    // Rest Period
                    <div className="text-center space-y-4">
                      <div className="space-y-2">
                        <h4 className="text-lg font-medium">Rest Time</h4>
                        <div className="text-3xl font-bold text-primary">
                          {Math.floor(session.restTimeRemaining / 60)}:{(session.restTimeRemaining % 60).toString().padStart(2, '0')}
                        </div>
                      </div>
                      <Badge variant="secondary" className="text-sm">
                        Preparing for next set...
                      </Badge>
                    </div>
                  ) : (
                    // Active Exercise
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <Button onClick={completeSet} size="lg" className="col-span-full md:col-span-1">
                          <CheckCircle className="h-4 w-4 mr-2" />
                          Complete Set
                        </Button>
                        <Button variant="outline" onClick={provideFormCue}>
                          Form Cue
                        </Button>
                        <Button variant="outline" onClick={provideMotivation}>
                          Motivation
                        </Button>
                      </div>

                      {/* Exercise Instructions */}
                      <div className="space-y-3">
                        <h4 className="font-medium">Instructions</h4>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          {currentExercise.instructions.map((instruction, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <span className="flex-shrink-0 w-4 h-4 bg-primary/20 rounded-full flex items-center justify-center text-xs font-medium mt-0.5">
                                {index + 1}
                              </span>
                              {instruction}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Workout Controls */}
              <div className="flex justify-center gap-3 pt-4 border-t">
                <Button variant="outline" onClick={() => {
                  setIsActive(false);
                  setSession(null);
                  if (restTimer) clearInterval(restTimer);
                }}>
                  <Pause className="h-4 w-4 mr-2" />
                  End Workout
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </TierGate>
  );
}