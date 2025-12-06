import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Mic, MicOff, Volume2, VolumeX, Activity } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import { useTierAccess } from '@/hooks/useTierAccess';
import { TierGate } from '@/components/auth/TierGate';
import { LiveMetricsDisplay } from './components/LiveMetricsDisplay';
import { CoachingCuesDisplay } from './components/CoachingCuesDisplay';

interface LiveMetrics {
  heartRate?: number;
  calories: number;
  duration: number;
  currentExercise: string;
  setCount: number;
  repCount: number;
}

interface CoachingCue {
  id: string;
  type: 'form' | 'motivation' | 'pacing' | 'safety';
  message: string;
  priority: 'low' | 'medium' | 'high';
  timestamp: number;
}

interface CoachingMessage {
  type: 'coaching_cue' | 'metrics_update' | 'audio_cue';
  cueType?: CoachingCue['type'];
  message?: string;
  priority?: CoachingCue['priority'];
  metrics?: Partial<LiveMetrics>;
  audioData?: ArrayBuffer;
}

interface LiveCoachingInterfaceProps {
  clientId: string;
  workoutId?: string;
  className?: string;
}

export const LiveCoachingInterface: React.FC<LiveCoachingInterfaceProps> = ({
  clientId,
  workoutId,
  className = ""
}) => {
  const [isActive, setIsActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [metrics, setMetrics] = useState<LiveMetrics>({
    calories: 0,
    duration: 0,
    currentExercise: 'Squats',
    setCount: 1,
    repCount: 0
  });
  const [recentCues, setRecentCues] = useState<CoachingCue[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const tierAccess = useTierAccess();

  useEffect(() => {
    return () => {
      stopLiveCoaching();
    };
  }, []);

  const startLiveCoaching = async () => {
    if (!tierAccess.hasMinimumTier('longevity')) {
      toast({
        title: "Upgrade Required",
        description: "Live coaching requires Longevity Concierge tier",
        variant: "destructive"
      });
      return;
    }

    try {
      setConnectionStatus('connecting');
      
      // Initialize audio context
      audioContextRef.current = new AudioContext();
      
      // Start WebSocket connection to live coaching edge function
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//lvmcumsfpjjcgnnovvzs.functions.supabase.co/functions/v1/live-workout-coaching`;
      
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        setIsActive(true);
        
        // Send initial session data
        wsRef.current?.send(JSON.stringify({
          type: 'session_start',
          clientId,
          workoutId,
          timestamp: Date.now()
        }));

        // Start metrics timer
        timerRef.current = setInterval(() => {
          setMetrics(prev => ({
            ...prev,
            duration: prev.duration + 1,
            calories: prev.calories + (Math.random() * 0.2) // Simulate calorie burn
          }));
        }, 1000);

        toast({
          title: "Live Coaching Started",
          description: "AI coach is now monitoring your workout",
        });
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleCoachingMessage(data);
      };

      wsRef.current.onerror = () => {
        setConnectionStatus('disconnected');
        toast({
          title: "Connection Error",
          description: "Failed to connect to live coaching. Using offline mode.",
          variant: "destructive"
        });
        
        // Fallback to demo mode
        setIsActive(true);
        startDemoMode();
      };

      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        setIsActive(false);
      };

    } catch (error) {
      console.error('Error starting live coaching:', error);
      setConnectionStatus('disconnected');
      
      // Fallback to demo mode
      toast({
        title: "Demo Mode",
        description: "Running in demo mode. Upgrade for real-time coaching.",
      });
      setIsActive(true);
      startDemoMode();
    }
  };

  const startDemoMode = () => {
    // Simulate coaching cues in demo mode
    const demoCues = [
      { type: 'form', message: 'Great depth on that squat! Keep your chest up.', priority: 'medium' },
      { type: 'motivation', message: 'You\'re crushing it! Two more reps.', priority: 'low' },
      { type: 'pacing', message: 'Slow down the eccentric phase for better muscle engagement.', priority: 'high' },
      { type: 'safety', message: 'Take a 30-second rest before the next set.', priority: 'high' }
    ];

    let cueIndex = 0;
    const demoInterval = setInterval(() => {
      if (cueIndex < demoCues.length && isActive) {
        const cue = demoCues[cueIndex];
        addCoachingCue(cue.type, cue.message, cue.priority);
        cueIndex++;
      } else {
        clearInterval(demoInterval);
      }
    }, 8000);

    timerRef.current = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        duration: prev.duration + 1,
        calories: prev.calories + (Math.random() * 0.2)
      }));
    }, 1000);
  };

  const stopLiveCoaching = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    
    setIsActive(false);
    setIsListening(false);
    setConnectionStatus('disconnected');
  };

  const handleCoachingMessage = (data: CoachingMessage) => {
    switch (data.type) {
      case 'coaching_cue':
        addCoachingCue(data.cueType, data.message, data.priority);
        break;
      case 'metrics_update':
        setMetrics(prev => ({ ...prev, ...data.metrics }));
        break;
      case 'audio_cue':
        playAudioCue(data.audioData);
        break;
    }
  };

  const addCoachingCue = (type: CoachingCue['type'], message: string, priority: CoachingCue['priority']) => {
    const newCue: CoachingCue = {
      id: Date.now().toString(),
      type,
      message,
      priority,
      timestamp: Date.now()
    };
    
    setRecentCues(prev => [newCue, ...prev.slice(0, 4)]); // Keep last 5 cues
    
    // Speak the cue if enabled
    if (isSpeaking) {
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 0.9;
      utterance.volume = 0.8;
      speechSynthesis.speak(utterance);
    }
  };

  const playAudioCue = async (audioData: string) => {
    if (!audioContextRef.current) return;
    
    try {
      const audioBuffer = await audioContextRef.current.decodeAudioData(
        Uint8Array.from(atob(audioData), c => c.charCodeAt(0)).buffer
      );
      
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      source.start();
    } catch (error) {
      console.error('Error playing audio cue:', error);
    }
  };

  const getCueColor = (type: CoachingCue['type'], priority: CoachingCue['priority']) => {
    if (priority === 'high') return 'border-red-500 bg-red-50';
    if (type === 'form') return 'border-blue-500 bg-blue-50';
    if (type === 'motivation') return 'border-green-500 bg-green-50';
    if (type === 'pacing') return 'border-yellow-500 bg-yellow-50';
    return 'border-gray-500 bg-gray-50';
  };

  const getCueIcon = (type: CoachingCue['type']) => {
    switch (type) {
      case 'form': return Activity;
      case 'motivation': return Activity;
      case 'pacing': return Activity;
      case 'safety': return Activity;
      default: return Activity;
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <TierGate
      requiredTier="longevity"
      feature="live_coaching"
      fallback={
        <Card className={className}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Mic className="h-5 w-5" />
              Live AI Coaching
            </CardTitle>
            <CardDescription>
              Upgrade to Longevity Concierge for real-time AI coaching during workouts
            </CardDescription>
          </CardHeader>
        </Card>
      }
    >
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Mic className="h-5 w-5" />
              Live AI Coaching
            </div>
            <Badge 
              variant={connectionStatus === 'connected' ? 'default' : 'secondary'}
              className={connectionStatus === 'connected' ? 'bg-green-600' : ''}
            >
              {connectionStatus}
            </Badge>
          </CardTitle>
          <CardDescription>
            Real-time AI coaching with form feedback and motivation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Control Buttons */}
          <div className="flex justify-center gap-3">
            {!isActive ? (
              <Button onClick={startLiveCoaching} className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Start Live Coaching
              </Button>
            ) : (
              <>
                <Button onClick={stopLiveCoaching} variant="destructive" className="flex items-center gap-2">
                  <MicOff className="h-4 w-4" />
                  Stop Session
                </Button>
                <Button
                  onClick={() => setIsSpeaking(!isSpeaking)}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  {isSpeaking ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                  {isSpeaking ? 'Mute' : 'Unmute'}
                </Button>
              </>
            )}
          </div>

          <LiveMetricsDisplay metrics={metrics} isActive={isActive} />
          <CoachingCuesDisplay cues={recentCues} isActive={isActive} />
        </CardContent>
      </Card>
    </TierGate>
  );
};