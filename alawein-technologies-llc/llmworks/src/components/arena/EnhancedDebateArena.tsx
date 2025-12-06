import { memo, useState, useEffect, useReducer } from 'react';
import { AIPersonalityAvatar, type AIPersonality, type BattleState } from './AIPersonalityAvatar';
import { EpicConfrontationMoments } from './EpicConfrontationMoments';
import { ArgumentImpactSystem } from './ArgumentImpactSystem';
import { ComboDetectionSystem } from './ComboDetectionSystem';
import { EnvironmentalEffects } from './EnvironmentalEffects';
import { ModelEnergySignatures } from './ModelEnergySignatures';
import { CitationTracker } from './CitationTracker';
import { PlayCircle, PauseCircle, Settings, Volume2, VolumeX, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Debater {
  id: string;
  name: string;
  personality: AIPersonality;
  energy: number;
  battleState: BattleState;
  arguments: string[];
  position: 'left' | 'right';
}

interface DebateState {
  phase: 'setup' | 'opening' | 'exchange' | 'closing' | 'judgment' | 'complete';
  timeRemaining: number;
  currentSpeaker: 'left' | 'right' | null;
  arguments: Array<{
    id: string;
    speaker: 'left' | 'right';
    content: string;
    strength: number;
    timestamp: Date;
    type: 'opening' | 'argument' | 'counter' | 'closing';
  }>;
  score: { left: number; right: number };
  academicMode: boolean;
  soundEnabled: boolean;
}

type DebateAction = 
  | { type: 'SET_PHASE'; phase: DebateState['phase'] }
  | { type: 'ADD_ARGUMENT'; argument: DebateState['arguments'][0] }
  | { type: 'UPDATE_SCORE'; side: 'left' | 'right'; score: number }
  | { type: 'SET_SPEAKER'; speaker: 'left' | 'right' | null }
  | { type: 'TICK_TIME' }
  | { type: 'TOGGLE_ACADEMIC_MODE' }
  | { type: 'TOGGLE_SOUND' }
  | { type: 'RESET_DEBATE'; topic: string };

const debateReducer = (state: DebateState, action: DebateAction): DebateState => {
  switch (action.type) {
    case 'SET_PHASE':
      return { ...state, phase: action.phase };
    
    case 'ADD_ARGUMENT':
      return { 
        ...state, 
        arguments: [...state.arguments, action.argument]
      };
    
    case 'UPDATE_SCORE':
      return {
        ...state,
        score: { ...state.score, [action.side]: action.score }
      };
    
    case 'SET_SPEAKER':
      return { ...state, currentSpeaker: action.speaker };
    
    case 'TICK_TIME':
      return { 
        ...state, 
        timeRemaining: Math.max(0, state.timeRemaining - 1) 
      };
    
    case 'TOGGLE_ACADEMIC_MODE':
      return { ...state, academicMode: !state.academicMode };
    
    case 'TOGGLE_SOUND':
      return { ...state, soundEnabled: !state.soundEnabled };
    
    case 'RESET_DEBATE':
      return {
        ...state,
        phase: 'setup',
        timeRemaining: 300,
        currentSpeaker: null,
        arguments: [],
        score: { left: 0, right: 0 }
      };
    
    default:
      return state;
  }
};

interface EnhancedDebateArenaProps {
  topic: string;
  leftDebater: Omit<Debater, 'position'>;
  rightDebater: Omit<Debater, 'position'>;
  onDebateComplete?: (winner: 'left' | 'right' | 'tie', finalScore: { left: number; right: number }) => void;
}

const EnhancedDebateArenaComponent = ({
  topic,
  leftDebater,
  rightDebater,
  onDebateComplete
}: EnhancedDebateArenaProps) => {
  const [debateState, dispatch] = useReducer(debateReducer, {
    phase: 'setup',
    timeRemaining: 300,
    currentSpeaker: null,
    arguments: [],
    score: { left: 0, right: 0 },
    academicMode: false,
    soundEnabled: true
  });

  const [debaters, setDebaters] = useState<[Debater, Debater]>([
    { ...leftDebater, position: 'left' },
    { ...rightDebater, position: 'right' }
  ]);

  const [isPlaying, setIsPlaying] = useState(false);
  const [argumentIntensity, setArgumentIntensity] = useState(0);

  // Simulate debate progression
  useEffect(() => {
    if (!isPlaying || debateState.phase === 'complete') return;

    const interval = setInterval(() => {
      dispatch({ type: 'TICK_TIME' });
      
      // Simulate argument generation
      if (Math.random() < 0.3) { // 30% chance per second
        simulateArgument();
      }
      
      // Check for phase transitions
      if (debateState.timeRemaining <= 0) {
        advancePhase();
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isPlaying, debateState.phase, debateState.timeRemaining]);

  const simulateArgument = () => {
    const speaker = debateState.currentSpeaker || (Math.random() < 0.5 ? 'left' : 'right');
    const argumentStrength = Math.floor(Math.random() * 40) + 40; // 40-80 base strength
    
    // Add topic relevance and personality modifiers
    const debater = debaters.find(d => d.position === speaker);
    let modifiedStrength = argumentStrength;
    
    if (debater) {
      switch (debater.personality) {
        case 'analytical':
          modifiedStrength += Math.random() < 0.7 ? 15 : 0; // 70% chance for logic bonus
          break;
        case 'creative':
          modifiedStrength += Math.random() < 0.6 ? 20 : 0; // 60% chance for creativity bonus
          break;
        case 'speed':
          modifiedStrength += Math.random() < 0.8 ? 10 : 0; // 80% chance for quick response
          break;
      }
    }

    const argument = {
      id: `arg-${Date.now()}`,
      speaker,
      content: generateArgumentContent(speaker, debater?.personality || 'analytical'),
      strength: Math.min(100, modifiedStrength),
      timestamp: new Date(),
      type: 'argument' as const
    };

    dispatch({ type: 'ADD_ARGUMENT', argument });
    
    // Update argument intensity for environmental effects
    setArgumentIntensity(Math.min(100, modifiedStrength));
    
    // Update debater states
    setDebaters(prev => [
      prev[0].position === speaker ? {
        ...prev[0],
        battleState: 'arguing',
        energy: Math.min(100, prev[0].energy + 5)
      } : {
        ...prev[0],
        battleState: argument.strength > 70 ? 'defending' : 'idle',
        energy: argument.strength > 70 ? Math.max(0, prev[0].energy - 3) : prev[0].energy
      },
      prev[1].position === speaker ? {
        ...prev[1],
        battleState: 'arguing',
        energy: Math.min(100, prev[1].energy + 5)
      } : {
        ...prev[1],
        battleState: argument.strength > 70 ? 'defending' : 'idle',
        energy: argument.strength > 70 ? Math.max(0, prev[1].energy - 3) : prev[1].energy
      }
    ]);

    // Update scores
    const scoreIncrease = Math.floor(argument.strength / 10);
    dispatch({ type: 'UPDATE_SCORE', side: speaker, score: debateState.score[speaker] + scoreIncrease });

    // Switch speaker
    setTimeout(() => {
      dispatch({ type: 'SET_SPEAKER', speaker: speaker === 'left' ? 'right' : 'left' });
    }, 2000);
  };

  const generateArgumentContent = (speaker: 'left' | 'right', personality: AIPersonality): string => {
    const argumentStyles = {
      analytical: [
        "According to established research patterns...",
        "The logical framework suggests...",
        "Empirical evidence demonstrates...",
        "Systematic analysis reveals..."
      ],
      creative: [
        "Consider this innovative perspective...",
        "What if we reimagined...",
        "A fresh approach might be...",
        "Breaking conventional thinking..."
      ],
      speed: [
        "Rapid assessment shows...",
        "Quick analysis indicates...",
        "Immediate pattern recognition...",
        "Fast processing reveals..."
      ],
      conversational: [
        "In human terms, this means...",
        "From a relatable standpoint...",
        "Speaking practically...",
        "In everyday experience..."
      ],
      strategic: [
        "Tactical evaluation suggests...",
        "Strategic positioning indicates...",
        "From a planning perspective...",
        "Calculated approach shows..."
      ]
    };

    const styles = argumentStyles[personality];
    return styles[Math.floor(Math.random() * styles.length)];
  };

  const advancePhase = () => {
    switch (debateState.phase) {
      case 'setup':
        dispatch({ type: 'SET_PHASE', phase: 'opening' });
        dispatch({ type: 'SET_SPEAKER', speaker: 'left' });
        break;
      case 'opening':
        dispatch({ type: 'SET_PHASE', phase: 'exchange' });
        break;
      case 'exchange':
        dispatch({ type: 'SET_PHASE', phase: 'closing' });
        break;
      case 'closing':
        dispatch({ type: 'SET_PHASE', phase: 'judgment' });
        break;
      case 'judgment':
        dispatch({ type: 'SET_PHASE', phase: 'complete' });
        determineWinner();
        break;
    }
  };

  const determineWinner = () => {
    const { left, right } = debateState.score;
    const winner = left > right ? 'left' : right > left ? 'right' : 'tie';
    
    // Update debater states
    setDebaters(prev => [
      {
        ...prev[0],
        battleState: winner === prev[0].position ? 'victory' : winner === 'tie' ? 'idle' : 'defeat'
      },
      {
        ...prev[1],
        battleState: winner === prev[1].position ? 'victory' : winner === 'tie' ? 'idle' : 'defeat'
      }
    ]);

    onDebateComplete?.(winner, debateState.score);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getPhaseDescription = (): string => {
    switch (debateState.phase) {
      case 'setup': return 'Preparing for intellectual combat...';
      case 'opening': return 'Opening statements';
      case 'exchange': return 'Active debate exchange';
      case 'closing': return 'Closing arguments';
      case 'judgment': return 'Evaluating performance...';
      case 'complete': return 'Debate concluded';
      default: return '';
    }
  };

  return (
    <div className="relative">
      {/* Arena Background */}
      <div className="glass-panel p-8 rounded-xl border-2 border-primary/20 min-h-[600px]">
        
        {/* Arena Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <h2 className="heading-refined text-2xl">Neural Battlefield Arena</h2>
            <div className="glass-minimal px-4 py-2 rounded-full">
              <div className="flex items-center gap-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${
                  debateState.phase === 'exchange' ? 'bg-red-400 animate-pulse' :
                  debateState.phase === 'complete' ? 'bg-green-400' : 'bg-yellow-400'
                }`} />
                <span>{getPhaseDescription()}</span>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => dispatch({ type: 'TOGGLE_ACADEMIC_MODE' })}
              className={`glass-minimal ${debateState.academicMode ? 'bg-primary/10' : ''}`}
            >
              {debateState.academicMode ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => dispatch({ type: 'TOGGLE_SOUND' })}
              className="glass-minimal"
            >
              {debateState.soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
            </Button>

            <Button
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={debateState.phase === 'complete'}
              className="bg-primary hover:bg-primary/90"
            >
              {isPlaying ? <PauseCircle className="h-4 w-4" /> : <PlayCircle className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* Topic Display */}
        <div className="text-center mb-8 relative">
          <div className="glass-subtle p-4 rounded-lg border border-primary/20">
            <h3 className="heading-refined text-lg mb-2">Debate Topic</h3>
            <p className="text-muted-foreground italic">"{topic}"</p>
          </div>
          
          {/* Citation Tracker Overlay */}
          <div className="absolute inset-0">
            <CitationTracker
              arguments={debateState.arguments}
              onCitationVerified={(citation, isValid) => {
                console.log('Citation verified:', citation, isValid);
                // Could update UI indicators or scoring
              }}
              onFactCheckComplete={(factCheck) => {
                console.log('Fact check completed:', factCheck);
                // Could influence debate scoring
              }}
              academicMode={debateState.academicMode}
              realTimeChecking={isPlaying}
            />
          </div>
        </div>

        {/* Main Arena */}
        <div className="relative h-96 bg-gradient-to-b from-muted/5 to-muted/20 rounded-lg border border-primary/20 overflow-hidden">
          
          {/* Model Energy Signatures Layer (Background) */}
          <ModelEnergySignatures
            leftPersonality={debaters[0].personality}
            rightPersonality={debaters[1].personality}
            leftEnergy={debaters[0].energy}
            rightEnergy={debaters[1].energy}
            leftBattleState={debaters[0].battleState}
            rightBattleState={debaters[1].battleState}
            isActive={isPlaying}
            academicMode={debateState.academicMode}
          />
          
          {/* Environmental Effects Layer */}
          <EnvironmentalEffects
            topic={{
              text: topic,
              category: 'technology', // Could be dynamic based on topic analysis
              intensity: argumentIntensity > 80 ? 'explosive' : argumentIntensity > 60 ? 'heated' : argumentIntensity > 40 ? 'moderate' : 'calm'
            }}
            debatePhase={debateState.phase}
            argumentIntensity={argumentIntensity}
            academicMode={debateState.academicMode}
          />
          
          {/* Argument Impact System */}
          <ArgumentImpactSystem
            onImpact={(impact) => {
              console.log('Argument impact:', impact);
              // Could trigger sound effects here if enabled
            }}
            academicMode={debateState.academicMode}
            soundEnabled={debateState.soundEnabled}
          />
          
          {/* Combo Detection System */}
          <ComboDetectionSystem
            arguments={debateState.arguments}
            onComboDetected={(combo) => {
              console.log('Combo detected:', combo);
              // Could trigger sound effects here if enabled
            }}
            onComboComplete={(combo, finalScore) => {
              console.log('Combo complete:', combo, finalScore);
              // Add bonus points to the combo performer
              const bonusPoints = Math.floor(finalScore / 10);
              dispatch({ 
                type: 'UPDATE_SCORE', 
                side: combo.speaker, 
                score: debateState.score[combo.speaker] + bonusPoints 
              });
            }}
            academicMode={debateState.academicMode}
          />
          
          {/* Epic Confrontation Effects */}
          <EpicConfrontationMoments 
            argumentExchanges={debateState.arguments}
            academicMode={debateState.academicMode}
            onEpicMoment={(moment) => {
              console.log('Epic moment:', moment);
              // Could trigger sound effects here if enabled
            }}
          />

          {/* Debater Positions */}
          <div className="absolute inset-0 flex items-center justify-between p-8">
            <AIPersonalityAvatar
              personality={debaters[0].personality}
              battleState={debaters[0].battleState}
              energy={debaters[0].energy}
              name={debaters[0].name}
              isActive={debateState.currentSpeaker === 'left'}
              position="left"
            />

            <AIPersonalityAvatar
              personality={debaters[1].personality}
              battleState={debaters[1].battleState}
              energy={debaters[1].energy}
              name={debaters[1].name}
              isActive={debateState.currentSpeaker === 'right'}
              position="right"
            />
          </div>

          {/* Center Information */}
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
            <div className="glass-minimal px-4 py-2 rounded-full text-center">
              <div className="text-lg font-mono">{formatTime(debateState.timeRemaining)}</div>
              <div className="text-xs text-muted-foreground capitalize">{debateState.phase}</div>
            </div>
          </div>
        </div>

        {/* Score Display */}
        <div className="mt-8 grid grid-cols-3 gap-4">
          <div className="glass-minimal p-4 rounded-lg text-center">
            <div className="text-2xl font-mono text-primary">{debateState.score.left}</div>
            <div className="text-sm text-muted-foreground">{debaters[0].name}</div>
          </div>
          
          <div className="glass-minimal p-4 rounded-lg text-center">
            <div className="text-lg font-mono text-secondary">VS</div>
            <div className="text-xs text-muted-foreground">
              {debateState.arguments.length} exchanges
            </div>
          </div>
          
          <div className="glass-minimal p-4 rounded-lg text-center">
            <div className="text-2xl font-mono text-primary">{debateState.score.right}</div>
            <div className="text-sm text-muted-foreground">{debaters[1].name}</div>
          </div>
        </div>

        {/* Recent Arguments (Academic Mode) */}
        {debateState.academicMode && debateState.arguments.length > 0 && (
          <div className="mt-6">
            <h4 className="heading-refined text-sm mb-3">Recent Arguments</h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {debateState.arguments.slice(-3).map(argument => (
                <div key={argument.id} className="glass-minimal p-3 rounded-lg text-sm">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">
                      {argument.speaker === 'left' ? debaters[0].name : debaters[1].name}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      Strength: {argument.strength}
                    </span>
                  </div>
                  <p className="text-muted-foreground">{argument.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export const EnhancedDebateArena = memo(EnhancedDebateArenaComponent);