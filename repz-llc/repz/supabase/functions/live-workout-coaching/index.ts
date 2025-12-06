import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.3';

// Type definitions
type SupabaseClient = ReturnType<typeof createClient>;

interface WorkoutData {
  workoutPlanId?: string;
  exerciseName?: string;
  sets?: number;
  reps?: number;
  weight?: number;
  restTime?: number;
  formRating?: number;
  duration?: number;
  exercisesCompleted?: number;
  performanceScore?: number;
  targetSets?: number;
  exerciseType?: string;
  intensity?: number;
  averageFormRating?: number;
  totalSets?: number;
  plannedExercises?: number;
  plannedDuration?: number;
}

interface WSMessage {
  type: string;
  data?: WorkoutData;
}

interface FormAnalysis {
  score: number;
  confidence: number;
  corrections: string[];
}

interface WorkoutSummary {
  duration: number;
  exercisesCompleted: number;
  totalSets: number;
  averageFormRating: number;
  caloriesBurned: number;
  achievements: string[];
  nextWorkoutRecommendation: string;
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const { headers } = req;
    const upgradeHeader = headers.get("upgrade") || "";

    if (upgradeHeader.toLowerCase() !== "websocket") {
      return new Response("Expected WebSocket connection", { status: 400 });
    }

    // Get auth token from query params or headers
    const url = new URL(req.url);
    const authToken = url.searchParams.get('token') || headers.get('authorization');
    
    if (!authToken) {
      return new Response("Authentication required", { status: 401 });
    }

    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Verify user authentication
    const { data: { user }, error: authError } = await supabase.auth.getUser(authToken.replace('Bearer ', ''));
    
    if (authError || !user) {
      return new Response("Invalid authentication", { status: 401 });
    }

    const { socket, response } = Deno.upgradeWebSocket(req);
    
    // Store active connections
    const connections = new Map();
    const sessionId = crypto.randomUUID();
    
    socket.onopen = () => {
      connections.set(sessionId, {
        socket,
        userId: user.id,
        connectedAt: new Date().toISOString()
      });
      
      console.log(`User ${user.id} connected to live workout session ${sessionId}`);
      
      // Send initial connection message
      socket.send(JSON.stringify({
        type: 'connected',
        sessionId,
        message: 'Connected to live workout session',
        timestamp: new Date().toISOString()
      }));
    };

    socket.onmessage = async (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data);
        console.log('Received message:', message);

        switch (message.type) {
          case 'workout_start':
            await handleWorkoutStart(user.id, message.data, socket, supabase);
            break;
            
          case 'exercise_complete':
            await handleExerciseComplete(user.id, message.data, socket, supabase);
            break;
            
          case 'form_check':
            await handleFormCheck(user.id, message.data, socket, supabase);
            break;
            
          case 'rest_timer':
            await handleRestTimer(user.id, message.data, socket, supabase);
            break;
            
          case 'workout_complete':
            await handleWorkoutComplete(user.id, message.data, socket, supabase);
            break;
            
          case 'coach_request':
            await handleCoachRequest(user.id, message.data, socket, supabase);
            break;
            
          default:
            socket.send(JSON.stringify({
              type: 'error',
              message: 'Unknown message type',
              timestamp: new Date().toISOString()
            }));
        }
      } catch (error) {
        console.error('Error processing message:', error);
        socket.send(JSON.stringify({
          type: 'error',
          message: 'Failed to process message',
          timestamp: new Date().toISOString()
        }));
      }
    };

    socket.onclose = () => {
      connections.delete(sessionId);
      console.log(`User ${user.id} disconnected from session ${sessionId}`);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      connections.delete(sessionId);
    };

    return response;

  } catch (error) {
    console.error('WebSocket connection error:', error);
    return new Response('Internal server error', { 
      status: 500,
      headers: corsHeaders 
    });
  }
});

async function handleWorkoutStart(userId: string, data: WorkoutData, socket: WebSocket, supabase: SupabaseClient) {
  // Log workout start
  const { error } = await supabase
    .from('live_workout_sessions')
    .insert({
      client_id: userId,
      workout_plan_id: data.workoutPlanId,
      started_at: new Date().toISOString(),
      status: 'active'
    });

  if (error) {
    console.error('Error starting workout session:', error);
  }

  // Send AI coaching message
  const coachingMessage = generateWorkoutStartMessage(data);
  
  socket.send(JSON.stringify({
    type: 'coaching_message',
    data: {
      message: coachingMessage,
      severity: 'info',
      category: 'motivation'
    },
    timestamp: new Date().toISOString()
  }));
  
  // Start heart rate monitoring if available
  socket.send(JSON.stringify({
    type: 'start_monitoring',
    data: {
      metrics: ['heart_rate', 'form_analysis'],
      interval: 5000 // 5 seconds
    },
    timestamp: new Date().toISOString()
  }));
}

async function handleExerciseComplete(userId: string, data: WorkoutData, socket: WebSocket, supabase: SupabaseClient) {
  // Log exercise completion
  const { error } = await supabase
    .from('exercise_logs')
    .insert({
      client_id: userId,
      exercise_name: data.exerciseName,
      sets_completed: data.sets,
      reps: data.reps,
      weight: data.weight,
      rest_time: data.restTime,
      form_rating: data.formRating,
      completed_at: new Date().toISOString()
    });

  if (error) {
    console.error('Error logging exercise:', error);
  }

  // Generate AI feedback
  const feedback = generateExerciseFeedback(data);
  
  socket.send(JSON.stringify({
    type: 'exercise_feedback',
    data: {
      exerciseName: data.exerciseName,
      feedback,
      nextAction: determineNextAction(data),
      restTime: calculateOptimalRestTime(data)
    },
    timestamp: new Date().toISOString()
  }));
}

async function handleFormCheck(userId: string, data: WorkoutData, socket: WebSocket, supabase: SupabaseClient) {
  // Analyze form data (this would integrate with computer vision in a real implementation)
  const formAnalysis = analyzeForm(data);
  
  socket.send(JSON.stringify({
    type: 'form_analysis',
    data: {
      exerciseName: data.exerciseName,
      analysis: formAnalysis,
      corrections: generateFormCorrections(formAnalysis),
      confidence: formAnalysis.confidence
    },
    timestamp: new Date().toISOString()
  }));
  
  // Log form check for coach review
  await supabase
    .from('form_checks')
    .insert({
      client_id: userId,
      exercise_name: data.exerciseName,
      form_score: formAnalysis.score,
      corrections_needed: formAnalysis.corrections,
      timestamp: new Date().toISOString()
    });
}

async function handleRestTimer(userId: string, data: WorkoutData, socket: WebSocket, supabase: SupabaseClient) {
  const { restTime, exerciseName } = data;
  const restTimeValue = restTime || 60;
  
  // Send motivational messages during rest
  const intervals = [
    { time: Math.floor(restTimeValue * 0.5), message: "You're halfway through your rest. Stay focused!" },
    { time: Math.floor(restTimeValue * 0.8), message: "Almost ready for the next set. Prepare yourself!" },
    { time: restTimeValue - 10, message: "10 seconds left! Get ready to crush this next set!" }
  ];
  
  intervals.forEach(interval => {
    setTimeout(() => {
      socket.send(JSON.stringify({
        type: 'rest_motivation',
        data: {
          message: interval.message,
          timeRemaining: restTimeValue - interval.time
        },
        timestamp: new Date().toISOString()
      }));
    }, interval.time * 1000);
  });
}

async function handleWorkoutComplete(userId: string, data: WorkoutData, socket: WebSocket, supabase: SupabaseClient) {
  // Update workout session
  const { error } = await supabase
    .from('live_workout_sessions')
    .update({
      completed_at: new Date().toISOString(),
      status: 'completed',
      total_duration: data.duration,
      exercises_completed: data.exercisesCompleted,
      performance_score: data.performanceScore
    })
    .eq('client_id', userId)
    .eq('status', 'active');

  if (error) {
    console.error('Error completing workout session:', error);
  }

  // Generate workout summary
  const summary = generateWorkoutSummary(data);
  
  socket.send(JSON.stringify({
    type: 'workout_summary',
    data: summary,
    timestamp: new Date().toISOString()
  }));
}

async function handleCoachRequest(userId: string, data: WorkoutData, socket: WebSocket, supabase: SupabaseClient) {
  // Notify available coaches
  const { data: coaches } = await supabase
    .from('coach_profiles')
    .select('auth_user_id')
    .eq('current_longevity_clients', 'lt', 'max_longevity_clients');

  // Send notification to coach (this would integrate with the notification system)
  await supabase
    .from('coach_notifications')
    .insert({
      title: 'Live Coaching Request',
      message: `Client needs live assistance with ${data.exerciseName}`,
      notification_type: 'alert',
      priority: 'high',
      subscriber_id: userId
    });

  socket.send(JSON.stringify({
    type: 'coach_request_sent',
    data: {
      message: 'Your coach has been notified and will assist you shortly.',
      estimatedResponse: '2-3 minutes'
    },
    timestamp: new Date().toISOString()
  }));
}

// AI Helper Functions
function generateWorkoutStartMessage(data: WorkoutData): string {
  const motivationalMessages = [
    "Let's crush this workout! Remember to focus on form over speed.",
    "You've got this! Stay consistent with your breathing and form.",
    "Time to show your strength! Listen to your body and push when you can.",
    "Another step closer to your goals. Let's make every rep count!"
  ];
  
  return motivationalMessages[Math.floor(Math.random() * motivationalMessages.length)];
}

function generateExerciseFeedback(data: WorkoutData): string {
  const { formRating, reps, weight } = data;
  
  if (formRating >= 8) {
    return "Excellent form! Your technique is on point. Consider increasing weight next time.";
  } else if (formRating >= 6) {
    return "Good effort! Focus on controlled movements and full range of motion.";
  } else {
    return "Let's work on form. Slow down the movement and focus on proper technique.";
  }
}

function determineNextAction(data: WorkoutData): string {
  const { sets, targetSets, formRating } = data;
  
  if (sets >= targetSets) {
    return "Great job completing all sets! Move to the next exercise.";
  } else if (formRating < 6) {
    return "Take an extra minute to rest and focus on form for the next set.";
  } else {
    return `Ready for set ${sets + 1}? Keep up the momentum!`;
  }
}

function calculateOptimalRestTime(data: WorkoutData): number {
  const { exerciseType, intensity, formRating } = data;
  
  let baseRestTime = 60; // Default 60 seconds
  
  if (exerciseType === 'compound') baseRestTime = 120;
  if (intensity >= 8) baseRestTime += 30;
  if (formRating < 6) baseRestTime += 15; // Extra rest for form focus
  
  return baseRestTime;
}

function analyzeForm(data: WorkoutData): FormAnalysis {
  // Placeholder for computer vision form analysis
  // In a real implementation, this would process video/sensor data
  
  const mockAnalysis = {
    score: Math.floor(Math.random() * 3) + 7, // Score between 7-10
    confidence: 0.85,
    corrections: []
  };
  
  if (mockAnalysis.score < 8) {
    mockAnalysis.corrections.push("Keep your back straight throughout the movement");
  }
  
  if (mockAnalysis.score < 7) {
    mockAnalysis.corrections.push("Control the eccentric (lowering) phase");
  }
  
  return mockAnalysis;
}

function generateFormCorrections(analysis: FormAnalysis): string[] {
  return analysis.corrections.length > 0 
    ? analysis.corrections 
    : ["Form looks good! Keep up the excellent technique."];
}

function generateWorkoutSummary(data: WorkoutData): WorkoutSummary {
  return {
    duration: data.duration,
    exercisesCompleted: data.exercisesCompleted,
    totalSets: data.totalSets,
    averageFormRating: data.averageFormRating,
    caloriesBurned: Math.floor(data.duration * 8), // Rough estimate
    achievements: generateAchievements(data),
    nextWorkoutRecommendation: generateNextWorkoutRecommendation(data)
  };
}

function generateAchievements(data: WorkoutData): string[] {
  const achievements = [];
  
  if (data.averageFormRating >= 8) {
    achievements.push("Form Master - Excellent technique throughout!");
  }
  
  if (data.exercisesCompleted >= data.plannedExercises) {
    achievements.push("Workout Warrior - Completed all planned exercises!");
  }
  
  if (data.duration >= data.plannedDuration * 0.9) {
    achievements.push("Endurance Champion - Maintained intensity!");
  }
  
  return achievements;
}

function generateNextWorkoutRecommendation(data: WorkoutData): string {
  if (data.averageFormRating >= 8 && data.performanceScore >= 8) {
    return "You're ready to increase intensity! Consider adding weight or reps next session.";
  } else if (data.averageFormRating < 7) {
    return "Focus on form refinement. Practice the same exercises with lighter weight.";
  } else {
    return "Great progress! Maintain current intensity and focus on consistency.";
  }
}