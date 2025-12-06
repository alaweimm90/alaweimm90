import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { TierGate } from '@/components/auth/TierGate';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Play, 
  Pause, 
  RotateCcw,
  Zap,
  Heart,
  Brain,
  Flame
} from 'lucide-react';

// Coach personality types
type CoachPersonality = 'motivational' | 'technical' | 'supportive' | 'intense';

interface CoachingMessage {
  id: string;
  text: string;
  personality: CoachPersonality;
  timestamp: Date;
  audioUrl?: string;
  isPlaying?: boolean;
}

interface VoiceSettings {
  personality: CoachPersonality;
  volume: number;
  speed: number;
  enabled: boolean;
}

interface WorkoutPhase {
  name: string;
  messages: string[];
}

const COACH_PERSONALITIES = {
  motivational: {
    name: 'Motivational Mike',
    icon: Zap,
    color: 'text-yellow-500',
    description: 'High-energy encouragement and motivation'
  },
  technical: {
    name: 'Technical Tom',
    icon: Brain,
    color: 'text-blue-500',
    description: 'Precise form cues and technique guidance'
  },
  supportive: {
    name: 'Supportive Sarah',
    icon: Heart,
    color: 'text-pink-500',
    description: 'Caring and understanding approach'
  },
  intense: {
    name: 'Intense Ivan',
    icon: Flame,
    color: 'text-red-500',
    description: 'High-intensity and demanding coaching'
  }
};

const WORKOUT_PHASES: Record<string, WorkoutPhase> = {
  warmup: {
    name: 'Warm-Up',
    messages: [
      "Let's start with a proper warm-up to prepare your body for the workout ahead.",
      "Focus on dynamic movements to increase blood flow and mobility.",
      "Take your time with each movement - quality over speed right now."
    ]
  },
  exercise: {
    name: 'Exercise Execution',
    messages: [
      "Perfect form! Keep that core engaged and maintain control.",
      "Feel the muscle working - slow and controlled on the negative.",
      "Breathe with the movement - exhale on exertion, inhale on the return.",
      "You've got this! Push through and maintain that perfect form."
    ]
  },
  rest: {
    name: 'Rest Period',
    messages: [
      "Great work! Use this rest time to recover and prepare for the next set.",
      "Hydrate and catch your breath - you're doing amazing.",
      "Next set coming up - visualize perfect execution."
    ]
  },
  cooldown: {
    name: 'Cool Down',
    messages: [
      "Excellent workout! Let's cool down properly to aid recovery.",
      "Focus on deep breathing and gentle stretching.",
      "You should be proud of the work you just put in."
    ]
  }
};

export default function EnhancedVoiceCoaching() {
  const { toast } = useToast();
  const [isActive, setIsActive] = useState(false);
  const [messages, setMessages] = useState<CoachingMessage[]>([]);
  const [settings, setSettings] = useState<VoiceSettings>({
    personality: 'motivational',
    volume: 80,
    speed: 1.0,
    enabled: true
  });
  const [currentPhase, setCurrentPhase] = useState<string>('warmup');
  const [isGenerating, setIsGenerating] = useState(false);
  
  const audioContext = useRef<AudioContext | null>(null);
  const audioQueue = useRef<HTMLAudioElement[]>([]);
  const isPlayingRef = useRef(false);

  // Initialize audio context
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (AudioContextCtor) {
        audioContext.current = new AudioContextCtor();
      }
    }
    return () => {
      audioContext.current?.close();
    };
  }, []);

  // Queue management for sequential playback
  const playNextInQueue = useCallback(async () => {
    if (audioQueue.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }

    isPlayingRef.current = true;
    const audio = audioQueue.current.shift()!;

    return new Promise<void>((resolve) => {
      audio.onended = () => {
        setMessages(prev => prev.map(m => ({ ...m, isPlaying: false })));
        resolve();
        playNextInQueue();
      };

      audio.onerror = () => {
        console.error('Audio playback error');
        resolve();
        playNextInQueue();
      };

      audio.play().catch((error) => {
        console.error('Failed to play audio:', error);
        resolve();
        playNextInQueue();
      });
    });
  }, []);

  // Play audio message with queue management
  const playAudioMessage = useCallback(async (message: CoachingMessage) => {
    if (!message.audioUrl || !settings.enabled) return;

    try {
      const audio = new Audio(message.audioUrl);
      audio.volume = settings.volume / 100;
      audio.playbackRate = settings.speed;

      // Add to queue
      audioQueue.current.push(audio);

      // Update message state
      setMessages(prev => prev.map(m =>
        m.id === message.id ? { ...m, isPlaying: true } : m
      ));

      // Play if not already playing
      if (!isPlayingRef.current) {
        await playNextInQueue();
      }

    } catch (error) {
      console.error('Error playing audio:', error);
      toast({
        title: "Audio Playback Error",
        description: "Unable to play coaching audio.",
        variant: "destructive"
      });
    }
  }, [settings.enabled, settings.volume, settings.speed, toast, playNextInQueue]);

  // Generate coaching message using ElevenLabs
  const generateCoachingMessage = useCallback(async (text: string, personality: CoachPersonality = settings.personality) => {
    if (!settings.enabled) return;

    setIsGenerating(true);
    try {
      const { data, error } = await supabase.functions.invoke('elevenlabs-tts', {
        body: {
          text,
          coachPersonality: personality,
          priority: 'normal'
        }
      });

      if (error) throw error;

      const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const newMessage: CoachingMessage = {
        id: messageId,
        text,
        personality,
        timestamp: new Date(),
        audioUrl: `data:audio/mpeg;base64,${data.audioContent}`
      };

      setMessages(prev => [newMessage, ...prev.slice(0, 19)]); // Keep last 20 messages

      // Auto-play if enabled
      if (settings.enabled) {
        await playAudioMessage(newMessage);
      }

    } catch (error) {
      console.error('Error generating coaching message:', error);
      toast({
        title: "Voice Generation Error",
        description: "Unable to generate voice coaching. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  }, [settings.enabled, settings.personality, toast, playAudioMessage]);

  // Start voice coaching session
  const startVoiceCoaching = useCallback(async () => {
    setIsActive(true);
    setMessages([]);
    
    toast({
      title: "Voice Coaching Started",
      description: `${COACH_PERSONALITIES[settings.personality].name} is ready to guide you!`
    });

    // Generate welcome message
    const welcomeMessages = {
      motivational: "Let's crush this workout together! I'm here to keep you motivated and push you to your limits!",
      technical: "Welcome to your session. I'll provide precise guidance to ensure perfect form and technique.",
      supportive: "I'm here to support you every step of the way. Let's work together at your pace.",
      intense: "Time to get serious! No excuses, no shortcuts - just pure dedication and results!"
    };

    await generateCoachingMessage(welcomeMessages[settings.personality]);
  }, [settings.personality, generateCoachingMessage, toast]);

  // Stop voice coaching
  const stopVoiceCoaching = useCallback(() => {
    setIsActive(false);
    isPlayingRef.current = false;
    audioQueue.current = [];
    
    // Stop all playing audio
    document.querySelectorAll('audio').forEach(audio => {
      audio.pause();
      audio.currentTime = 0;
    });

    setMessages(prev => prev.map(m => ({ ...m, isPlaying: false })));
    
    toast({
      title: "Voice Coaching Stopped",
      description: "Session ended. Great work!"
    });
  }, [toast]);

  // Generate phase-specific coaching
  const generatePhaseCoaching = useCallback(async (phase: string) => {
    const phaseData = WORKOUT_PHASES[phase];
    if (!phaseData || !isActive) return;

    const randomMessage = phaseData.messages[Math.floor(Math.random() * phaseData.messages.length)];
    await generateCoachingMessage(randomMessage);
  }, [isActive, generateCoachingMessage]);

  // Auto-generate coaching messages during workout
  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      generatePhaseCoaching(currentPhase);
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, [isActive, currentPhase, generatePhaseCoaching]);

  const PersonalityIcon = COACH_PERSONALITIES[settings.personality].icon;

  return (
    <TierGate requiredTier="performance" feature="voice_coaching">
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <PersonalityIcon className={`h-6 w-6 ${COACH_PERSONALITIES[settings.personality].color}`} />
            Enhanced Voice Coaching
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Coach Personality Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Coach Personality</label>
            <Select
              value={settings.personality}
              onValueChange={(value: CoachPersonality) => 
                setSettings(prev => ({ ...prev, personality: value }))
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(COACH_PERSONALITIES).map(([key, coach]) => {
                  const Icon = coach.icon;
                  return (
                    <SelectItem key={key} value={key}>
                      <div className="flex items-center gap-2">
                        <Icon className={`h-4 w-4 ${coach.color}`} />
                        <span>{coach.name}</span>
                      </div>
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground">
              {COACH_PERSONALITIES[settings.personality].description}
            </p>
          </div>

          {/* Voice Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Volume: {settings.volume}%</label>
              <Slider
                value={[settings.volume]}
                onValueChange={([value]) => setSettings(prev => ({ ...prev, volume: value }))}
                max={100}
                step={5}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Speed: {settings.speed}x</label>
              <Slider
                value={[settings.speed]}
                onValueChange={([value]) => setSettings(prev => ({ ...prev, speed: value }))}
                min={0.5}
                max={2.0}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>

          {/* Workout Phase Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Workout Phase</label>
            <Select value={currentPhase} onValueChange={setCurrentPhase}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(WORKOUT_PHASES).map(([key, phase]) => (
                  <SelectItem key={key} value={key}>
                    {phase.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Control Buttons */}
          <div className="flex flex-wrap gap-3">
            {!isActive ? (
              <Button onClick={startVoiceCoaching} className="flex items-center gap-2">
                <Mic className="h-4 w-4" />
                Start Voice Coaching
              </Button>
            ) : (
              <Button onClick={stopVoiceCoaching} variant="destructive" className="flex items-center gap-2">
                <MicOff className="h-4 w-4" />
                Stop Coaching
              </Button>
            )}

            <Button
              variant="outline"
              onClick={() => setSettings(prev => ({ ...prev, enabled: !prev.enabled }))}
              className="flex items-center gap-2"
            >
              {settings.enabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
              {settings.enabled ? 'Mute' : 'Unmute'}
            </Button>

            {isActive && (
              <Button
                variant="outline"
                onClick={() => generatePhaseCoaching(currentPhase)}
                disabled={isGenerating}
                className="flex items-center gap-2"
              >
                <RotateCcw className={`h-4 w-4 ${isGenerating ? 'animate-spin' : ''}`} />
                Generate Cue
              </Button>
            )}
          </div>

          {/* Status & Messages */}
          {isActive && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge variant={isActive ? "default" : "secondary"}>
                  {isActive ? 'Active' : 'Inactive'}
                </Badge>
                <Badge variant="outline">
                  {WORKOUT_PHASES[currentPhase].name}
                </Badge>
                <Badge variant="outline">
                  {COACH_PERSONALITIES[settings.personality].name}
                </Badge>
              </div>

              {/* Recent Messages */}
              <div className="space-y-2 max-h-60 overflow-y-auto">
                <h4 className="text-sm font-medium">Recent Coaching Messages</h4>
                {messages.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No messages yet...</p>
                ) : (
                  messages.map((message) => {
                    const CoachIcon = COACH_PERSONALITIES[message.personality].icon;
                    return (
                      <div key={message.id} className="flex items-start gap-3 p-3 bg-muted rounded-lg">
                        <CoachIcon className={`h-5 w-5 mt-0.5 ${COACH_PERSONALITIES[message.personality].color}`} />
                        <div className="flex-1 space-y-1">
                          <p className="text-sm">{message.text}</p>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>{message.timestamp.toLocaleTimeString()}</span>
                            <Badge variant="outline" className="text-xs">
                              {COACH_PERSONALITIES[message.personality].name}
                            </Badge>
                            {message.isPlaying && (
                              <Badge variant="default" className="text-xs">
                                Playing
                              </Badge>
                            )}
                          </div>
                        </div>
                        {message.audioUrl && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => playAudioMessage(message)}
                            className="h-6 w-6 p-0"
                          >
                            <Play className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </TierGate>
  );
}