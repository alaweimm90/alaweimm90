import { memo, useState, useEffect, useCallback } from 'react';
import { Zap, Target, Star, Crown, Flame } from 'lucide-react';

interface Argument {
  id: string;
  speaker: 'left' | 'right';
  content: string;
  strength: number;
  timestamp: Date;
  type: 'opening' | 'argument' | 'counter' | 'closing';
  citations?: string[];
  logicalStructure?: 'deductive' | 'inductive' | 'abductive';
}

interface Combo {
  id: string;
  type: 'logical-chain' | 'evidence-stack' | 'counter-cascade' | 'precision-strike' | 'overwhelming-force';
  speaker: 'left' | 'right';
  arguments: string[];
  strength: number;
  multiplier: number;
  description: string;
  startTime: Date;
  duration: number;
}

interface ComboDetectionSystemProps {
  arguments: Argument[];
  onComboDetected: (combo: Combo) => void;
  onComboComplete: (combo: Combo, finalScore: number) => void;
  academicMode?: boolean;
}

const ComboDetectionSystemComponent = ({
  arguments: argumentList,
  onComboDetected,
  onComboComplete,
  academicMode = false
}: ComboDetectionSystemProps) => {
  const [activeCombos, setActiveCombos] = useState<Combo[]>([]);
  const [comboHistory, setComboHistory] = useState<Combo[]>([]);
  const [comboCounters, setComboCounters] = useState<Record<'left' | 'right', number>>({ left: 0, right: 0 });

  // Advanced pattern recognition for argument sequences
  const analyzeComboPatterns = useCallback((recentArgs: Argument[]) => {
    if (recentArgs.length < 2) return null;

    const speaker = recentArgs[recentArgs.length - 1].speaker;
    const speakerArgs = recentArgs.filter(arg => arg.speaker === speaker);
    
    if (speakerArgs.length < 2) return null;

    // Detect different combo types
    const comboType = detectComboType(speakerArgs);
    if (!comboType) return null;

    const combo: Combo = {
      id: `combo-${Date.now()}-${speaker}`,
      type: comboType.type,
      speaker,
      arguments: speakerArgs.map(arg => arg.id),
      strength: calculateComboStrength(speakerArgs, comboType.type),
      multiplier: comboType.multiplier,
      description: comboType.description,
      startTime: speakerArgs[0].timestamp,
      duration: comboType.expectedDuration
    };

    return combo;
  }, []);

  const detectComboType = (args: Argument[]) => {
    const avgStrength = args.reduce((sum, arg) => sum + arg.strength, 0) / args.length;
    const hasStrongProgression = args.every((arg, i) => i === 0 || arg.strength >= args[i - 1].strength - 5);
    const hasCitations = args.some(arg => arg.citations && arg.citations.length > 0);
    const hasLogicalStructure = args.some(arg => arg.logicalStructure);

    // Logical Chain: Sequential arguments with increasing strength and logical structure
    if (hasStrongProgression && hasLogicalStructure && args.length >= 3) {
      return {
        type: 'logical-chain' as const,
        multiplier: 1.5,
        description: 'Systematic logical progression building to strong conclusion',
        expectedDuration: 8000
      };
    }

    // Evidence Stack: Multiple arguments with strong citations
    if (hasCitations && avgStrength > 70 && args.length >= 2) {
      return {
        type: 'evidence-stack' as const,
        multiplier: 1.3,
        description: 'Compelling evidence accumulation with strong sources',
        expectedDuration: 6000
      };
    }

    // Counter Cascade: Rapid succession of counter-arguments
    if (args.every(arg => arg.type === 'counter') && args.length >= 3) {
      return {
        type: 'counter-cascade' as const,
        multiplier: 1.4,
        description: 'Devastating sequence of counter-arguments',
        expectedDuration: 5000
      };
    }

    // Precision Strike: Single extremely strong argument (90+)
    if (args.length === 1 && args[0].strength >= 90) {
      return {
        type: 'precision-strike' as const,
        multiplier: 2.0,
        description: 'Surgical precision argument with devastating impact',
        expectedDuration: 4000
      };
    }

    // Overwhelming Force: Many consecutive strong arguments
    if (args.length >= 4 && avgStrength > 65) {
      return {
        type: 'overwhelming-force' as const,
        multiplier: 1.8,
        description: 'Relentless barrage of powerful arguments',
        expectedDuration: 10000
      };
    }

    return null;
  };

  const calculateComboStrength = (args: Argument[], comboType: Combo['type']): number => {
    const baseStrength = args.reduce((sum, arg) => sum + arg.strength, 0) / args.length;
    
    switch (comboType) {
      case 'logical-chain': {
        // Bonus for logical progression
        const progressionBonus = args.every((arg, i) => i === 0 || arg.strength > args[i - 1].strength) ? 20 : 0;
        return Math.min(100, baseStrength + progressionBonus);
      }
      case 'evidence-stack': {
        // Bonus for citation density
        const citationCount = args.reduce((sum, arg) => sum + (arg.citations?.length || 0), 0);
        const citationBonus = Math.min(25, citationCount * 5);
        return Math.min(100, baseStrength + citationBonus);
      }
      case 'counter-cascade': {
        // Bonus for rapid succession
        const rapidityBonus = args.length >= 4 ? 15 : args.length >= 3 ? 10 : 5;
        return Math.min(100, baseStrength + rapidityBonus);
      }
      case 'precision-strike': {
        // Already high strength single argument
        return args[0].strength;
      }
      case 'overwhelming-force': {
        // Bonus for sustained pressure
        const sustainedBonus = Math.min(20, (args.length - 3) * 5);
        return Math.min(100, baseStrength + sustainedBonus);
      }
      default:
        return baseStrength;
    }
  };

  // Monitor for combo opportunities
  useEffect(() => {
    if (argumentList.length < 2) return;

    const recentArgs = argumentList.slice(-6); // Analyze last 6 arguments
    const detectedCombo = analyzeComboPatterns(recentArgs);
    
    if (detectedCombo) {
      setActiveCombos(prev => {
        // Prevent duplicate combos for same speaker
        const existingCombo = prev.find(combo => 
          combo.speaker === detectedCombo.speaker && 
          Date.now() - combo.startTime.getTime() < 5000
        );
        
        if (existingCombo) return prev;
        
        return [...prev, detectedCombo];
      });
      
      setComboCounters(prev => ({
        ...prev,
        [detectedCombo.speaker]: prev[detectedCombo.speaker] + 1
      }));
      
      onComboDetected(detectedCombo);
      
      // Auto-complete combo after duration
      setTimeout(() => {
        const finalScore = detectedCombo.strength * detectedCombo.multiplier;
        onComboComplete(detectedCombo, finalScore);
        
        setActiveCombos(prev => prev.filter(combo => combo.id !== detectedCombo.id));
        setComboHistory(prev => [...prev, detectedCombo]);
      }, detectedCombo.duration);
    }
  }, [argumentList, analyzeComboPatterns, onComboDetected, onComboComplete]);

  const getComboIcon = (type: Combo['type']) => {
    switch (type) {
      case 'logical-chain': return Target;
      case 'evidence-stack': return Star;
      case 'counter-cascade': return Zap;
      case 'precision-strike': return Crown;
      case 'overwhelming-force': return Flame;
    }
  };

  const getComboColor = (type: Combo['type']) => {
    switch (type) {
      case 'logical-chain': return 'from-blue-400 to-cyan-400';
      case 'evidence-stack': return 'from-green-400 to-emerald-400';
      case 'counter-cascade': return 'from-purple-400 to-violet-400';
      case 'precision-strike': return 'from-red-400 to-orange-400';
      case 'overwhelming-force': return 'from-yellow-400 to-amber-400';
    }
  };

  if (academicMode) {
    return (
      <div className="absolute top-4 left-4 right-4 pointer-events-none">
        {/* Academic combo indicators */}
        {activeCombos.map(combo => (
          <div key={combo.id} className="glass-minimal p-2 mb-2 rounded-lg">
            <div className="text-xs font-medium text-primary">
              {combo.description}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Multiplier: {combo.multiplier}x | Strength: {Math.round(combo.strength)}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      
      {/* Active Combo Visualizations */}
      {activeCombos.map(combo => {
        const IconComponent = getComboIcon(combo.type);
        const gradientColor = getComboColor(combo.type);
        const position = combo.speaker === 'left' ? '25%' : '75%';
        
        return (
          <div
            key={combo.id}
            className="absolute"
            style={{
              left: position,
              top: '30%',
              transform: 'translate(-50%, -50%)'
            }}
          >
            {/* Main combo visual */}
            <div className={`
              relative w-24 h-24 rounded-full 
              bg-gradient-to-br ${gradientColor}
              animate-pulse shadow-2xl
            `}>
              <div className="absolute inset-2 bg-gradient-to-br from-white/20 to-transparent rounded-full" />
              
              <div className="absolute inset-0 flex items-center justify-center">
                <IconComponent className="h-8 w-8 text-white drop-shadow-lg animate-bounce" />
              </div>
              
              {/* Combo energy rings */}
              {[...Array(3)].map((_, i) => (
                <div
                  key={i}
                  className={`absolute border-4 border-white/30 rounded-full`}
                  style={{
                    width: `${120 + i * 30}%`,
                    height: `${120 + i * 30}%`,
                    left: '50%',
                    top: '50%',
                    transform: 'translate(-50%, -50%)',
                    animation: `combo-ring ${2 + i * 0.5}s ease-out infinite`,
                    animationDelay: `${i * 0.3}s`
                  }}
                />
              ))}
            </div>
            
            {/* Combo information */}
            <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 text-center">
              <div className="glass-panel px-3 py-1 rounded-lg">
                <div className="text-sm font-bold text-white uppercase tracking-wide">
                  {combo.type.replace('-', ' ')}
                </div>
                <div className="text-xs text-white/80">
                  {combo.multiplier}x Multiplier
                </div>
              </div>
            </div>

            {/* Progress indicator */}
            <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
              <div className="w-16 h-1 bg-black/20 rounded-full overflow-hidden">
                <div 
                  className={`h-full bg-gradient-to-r ${gradientColor} transition-all duration-100`}
                  style={{
                    animation: `combo-progress ${combo.duration}ms linear`
                  }}
                />
              </div>
            </div>
          </div>
        );
      })}

      {/* Combo Counter Display */}
      <div className="absolute top-4 left-4 right-4 flex justify-between pointer-events-none">
        <div className="glass-minimal px-4 py-2 rounded-lg">
          <div className="text-sm font-bold text-primary">
            Left Combos: {comboCounters.left}
          </div>
        </div>
        <div className="glass-minimal px-4 py-2 rounded-lg">
          <div className="text-sm font-bold text-primary">
            Right Combos: {comboCounters.right}
          </div>
        </div>
      </div>

      {/* Combo History Trail (last 3 combos) */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
        <div className="flex gap-2">
          {comboHistory.slice(-3).map((combo, index) => {
            const IconComponent = getComboIcon(combo.type);
            const gradientColor = getComboColor(combo.type);
            
            return (
              <div
                key={combo.id}
                className={`
                  w-8 h-8 rounded-full bg-gradient-to-br ${gradientColor}
                  flex items-center justify-center opacity-60
                  transform transition-all duration-300
                `}
                style={{
                  scale: 1 - index * 0.1,
                  animationDelay: `${index * 100}ms`
                }}
              >
                <IconComponent className="h-4 w-4 text-white" />
              </div>
            );
          })}
        </div>
      </div>

      {/* Custom animations */}
      <style>{`
        @keyframes combo-ring {
          0% {
            transform: translate(-50%, -50%) scale(0.8);
            opacity: 0.8;
          }
          100% {
            transform: translate(-50%, -50%) scale(1.5);
            opacity: 0;
          }
        }

        @keyframes combo-progress {
          0% { width: 0%; }
          100% { width: 100%; }
        }
      `}</style>
    </div>
  );
};

export const ComboDetectionSystem = memo(ComboDetectionSystemComponent);