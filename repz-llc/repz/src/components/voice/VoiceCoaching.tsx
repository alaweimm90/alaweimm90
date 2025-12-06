import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { Badge } from '@/ui/atoms/Badge';
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Settings, 
  Zap,
  AlertCircle,
  CheckCircle
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/hooks/use-toast';
import { useTierAccess } from '@/hooks/useTierAccess';
import { TierGate } from '@/components/auth/TierGate';

// Extend window interface for speech recognition
interface SpeechRecognitionConstructor {
  new (): SpeechRecognition;
}

declare global {
  interface Window {
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
    SpeechRecognition?: SpeechRecognitionConstructor;
  }
}

interface VoiceCoachingProps {
  clientId: string;
  className?: string;
}

interface VoiceSettings {
  voiceId: string;
  speed: number;
  volume: number;
  enabled: boolean;
}

interface CoachingMessage {
  id: string;
  type: 'motivation' | 'form' | 'pacing' | 'warning';
  message: string;
  timestamp: Date;
  spoken: boolean;
}

const VOICE_OPTIONS = [
  { id: 'aria', name: 'Aria (Calm & Clear)', voiceId: '9BWtsMINqrJLrRacOk9x' },
  { id: 'sarah', name: 'Sarah (Energetic)', voiceId: 'EXAVITQu4vr4xnSDxMaL' },
  { id: 'roger', name: 'Roger (Motivational)', voiceId: 'CwhRBWXzGAHq8TQ4Fs17' },
  { id: 'charlie', name: 'Charlie (Professional)', voiceId: 'IKne3meq5aSn9XLyUdCD' }
];

export const VoiceCoaching: React.FC<VoiceCoachingProps> = ({
  clientId,
  className = ""
}) => {
  const [isActive, setIsActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings>({
    voiceId: VOICE_OPTIONS[0].voiceId,
    speed: 1.0,
    volume: 0.8,
    enabled: true
  });
  
  const [messages, setMessages] = useState<CoachingMessage[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const audioContextRef = useRef<AudioContext | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognition | null>(null);


  const tierAccess = useTierAccess();

  useEffect(() => {
    return () => {
      stopVoiceCoaching();
    };
  }, []);

  const startVoiceCoaching = async () => {
    if (!tierAccess.hasMinimumTier('performance')) {
      toast({
        title: "Upgrade Required",
        description: "Voice coaching requires Performance+ tier",
        variant: "destructive"
      });
      return;
    }

    try {
      setConnectionStatus('connecting');
      
      // Initialize audio context
      audioContextRef.current = new AudioContext();
      
      // Initialize speech recognition
      if (typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        const SpeechRecognitionCtor = window.webkitSpeechRecognition || window.SpeechRecognition;
        if (SpeechRecognitionCtor) {
          speechRecognitionRef.current = new SpeechRecognitionCtor();
          speechRecognitionRef.current.continuous = true;
          speechRecognitionRef.current.interimResults = true;

          speechRecognitionRef.current.onresult = (event) => {
            const transcript = Array.from(event.results)
              .map(result => result[0].transcript)
              .join('');

            // Process voice commands
            processVoiceCommand(transcript.toLowerCase());
          };

          speechRecognitionRef.current.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
          };
        }
      }

      setConnectionStatus('connected');
      setIsActive(true);
      
      // Start generating coaching messages
      startCoachingSession();

      toast({
        title: "Voice Coaching Started",
        description: "AI coach is now listening and providing guidance",
      });

    } catch (error) {
      console.error('Error starting voice coaching:', error);
      setConnectionStatus('disconnected');
      
      // Fallback to demo mode
      setIsActive(true);
      setConnectionStatus('connected');
      startDemoCoaching();
      
      toast({
        title: "Demo Mode",
        description: "Running voice coaching in demo mode",
      });
    }
  };

  const stopVoiceCoaching = () => {
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.stop();
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    setIsActive(false);
    setIsListening(false);
    setConnectionStatus('disconnected');
    setMessages([]);
  };

  const startCoachingSession = () => {
    // Simulate real-time coaching messages
    const coachingMessages = [
      { type: 'motivation', message: 'Great form on that last set! Keep it up!' },
      { type: 'pacing', message: 'Remember to control your breathing during the eccentric phase.' },
      { type: 'form', message: 'Focus on engaging your core throughout the movement.' },
      { type: 'motivation', message: 'You\'re doing amazing! Two more reps.' },
      { type: 'warning', message: 'Take a moment to rest if you need it. Safety first!' }
    ];

    let messageIndex = 0;
    const interval = setInterval(() => {
      if (messageIndex < coachingMessages.length && isActive) {
        const message = coachingMessages[messageIndex];
        addCoachingMessage(message.type as 'coaching_cue' | 'metrics_update' | 'audio_cue', message.message);
        messageIndex++;
      } else {
        clearInterval(interval);
      }
    }, 15000); // New message every 15 seconds
  };

  const startDemoCoaching = () => {
    startCoachingSession();
  };

  const processVoiceCommand = (command: string) => {
    if (command.includes('next exercise')) {
      addCoachingMessage('form', 'Moving to the next exercise. Great job on completing that set!');
    } else if (command.includes('help') || command.includes('form check')) {
      addCoachingMessage('form', 'Remember to keep your core engaged and maintain proper alignment.');
    } else if (command.includes('tired') || command.includes('rest')) {
      addCoachingMessage('warning', 'Take the rest you need. Listen to your body.');
    }
  };

  const addCoachingMessage = (type: CoachingMessage['type'], message: string) => {
    const newMessage: CoachingMessage = {
      id: Date.now().toString(),
      type,
      message,
      timestamp: new Date(),
      spoken: false
    };
    
    setMessages(prev => [newMessage, ...prev.slice(0, 9)]); // Keep last 10 messages
    
    // Speak the message if voice is enabled
    if (voiceSettings.enabled) {
      speakMessage(message);
    }
  };

  const speakMessage = async (message: string) => {
    try {
      // For demo purposes, use browser's built-in speech synthesis
      // In production, this would call ElevenLabs API
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = voiceSettings.speed;
      utterance.volume = voiceSettings.volume;
      speechSynthesis.speak(utterance);
      
      // Mark message as spoken
      setMessages(prev => prev.map(msg => 
        msg.message === message ? { ...msg, spoken: true } : msg
      ));
      
    } catch (error) {
      console.error('Error speaking message:', error);
    }
  };

  const toggleListening = () => {
    if (!speechRecognitionRef.current) return;
    
    if (isListening) {
      speechRecognitionRef.current.stop();
      setIsListening(false);
    } else {
      speechRecognitionRef.current.start();
      setIsListening(true);
    }
  };

  const getMessageIcon = (type: CoachingMessage['type']) => {
    switch (type) {
      case 'motivation': return <Zap className="h-4 w-4 text-green-600" />;
      case 'form': return <CheckCircle className="h-4 w-4 text-blue-600" />;
      case 'pacing': return <Settings className="h-4 w-4 text-orange-600" />;
      case 'warning': return <AlertCircle className="h-4 w-4 text-red-600" />;
      default: return <CheckCircle className="h-4 w-4 text-gray-600" />;
    }
  };

  const getMessageColor = (type: CoachingMessage['type']) => {
    switch (type) {
      case 'motivation': return 'border-green-200 bg-green-50';
      case 'form': return 'border-blue-200 bg-blue-50';
      case 'pacing': return 'border-repz-orange/20 bg-repz-orange/10';
      case 'warning': return 'border-red-200 bg-red-50';
      default: return 'border-gray-200 bg-gray-50';
    }
  };

  return (
    <TierGate
      requiredTier="performance"
      feature="ai_coaching"
      fallback={
        <Card className={className}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Mic className="h-5 w-5" />
              Voice Coaching
            </CardTitle>
            <CardDescription>
              Upgrade to Performance+ to access AI voice coaching with real-time guidance
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
              AI Voice Coaching
            </div>
            <Badge 
              variant={connectionStatus === 'connected' ? 'default' : 'secondary'}
              className={connectionStatus === 'connected' ? 'bg-green-600' : ''}
            >
              {connectionStatus}
            </Badge>
          </CardTitle>
          <CardDescription>
            Real-time voice guidance and motivation during workouts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Voice Settings */}
          <div className="space-y-4">
            <h4 className="font-medium">Voice Settings</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Coach Voice</label>
                <Select 
                  value={voiceSettings.voiceId} 
                  onValueChange={(value) => setVoiceSettings(prev => ({ ...prev, voiceId: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VOICE_OPTIONS.map((voice) => (
                      <SelectItem key={voice.id} value={voice.voiceId}>
                        {voice.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Speaking Speed</label>
                <Select 
                  value={voiceSettings.speed.toString()} 
                  onValueChange={(value) => setVoiceSettings(prev => ({ ...prev, speed: parseFloat(value) }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0.8">Slow</SelectItem>
                    <SelectItem value="1.0">Normal</SelectItem>
                    <SelectItem value="1.2">Fast</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex justify-center gap-3">
            {!isActive ? (
              <Button onClick={startVoiceCoaching} className="flex items-center gap-2">
                <Mic className="h-4 w-4" />
                Start Voice Coaching
              </Button>
            ) : (
              <>
                <Button onClick={stopVoiceCoaching} variant="destructive" className="flex items-center gap-2">
                  <MicOff className="h-4 w-4" />
                  Stop
                </Button>
                <Button
                  onClick={toggleListening}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  {isListening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                  {isListening ? 'Stop Listening' : 'Start Listening'}
                </Button>
                <Button
                  onClick={() => setVoiceSettings(prev => ({ ...prev, enabled: !prev.enabled }))}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  {voiceSettings.enabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                  {voiceSettings.enabled ? 'Mute' : 'Unmute'}
                </Button>
              </>
            )}
          </div>

          {/* Coaching Messages */}
          {isActive && (
            <div className="space-y-3">
              <h4 className="font-medium">Recent Coaching Messages</h4>
              {messages.length === 0 ? (
                <div className="text-center py-6 text-muted-foreground">
                  <Mic className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>Listening for workout activity...</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`p-3 rounded-lg border ${getMessageColor(message.type)}`}
                    >
                      <div className="flex items-start gap-2">
                        {getMessageIcon(message.type)}
                        <div className="flex-1">
                          <p className="text-sm font-medium">{message.message}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-muted-foreground">
                              {message.timestamp.toLocaleTimeString()}
                            </span>
                            {message.spoken && (
                              <Badge variant="outline" className="text-xs">
                                <Volume2 className="h-3 w-3 mr-1" />
                                Spoken
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Voice Commands Help */}
          {isActive && (
            <div className="p-4 bg-muted/50 rounded-lg">
              <h5 className="font-medium mb-2">Voice Commands</h5>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>• Say "next exercise" to move to the next exercise</p>
                <p>• Say "help" or "form check" for technique reminders</p>
                <p>• Say "tired" or "rest" if you need a break</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </TierGate>
  );
};