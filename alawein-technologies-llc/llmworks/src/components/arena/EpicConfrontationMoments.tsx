import { memo, useState, useEffect, useRef } from 'react';
import { Zap, Shield, Target, Swords, Crown, Star } from 'lucide-react';

interface ArgumentExchange {
  id: string;
  speaker: 'left' | 'right';
  content: string;
  strength: number;
  timestamp: Date;
  type: 'opening' | 'argument' | 'counter' | 'closing';
}

interface EpicMoment {
  id: string;
  type: 'clash' | 'critical-hit' | 'perfect-counter' | 'combo-breaker' | 'finishing-move';
  participants: ['left' | 'right', 'left' | 'right'];
  intensity: number; // 0-100
  description: string;
}

interface EpicConfrontationMomentsProps {
  argumentExchanges: ArgumentExchange[];
  onEpicMoment?: (moment: EpicMoment) => void;
  academicMode?: boolean;
}

const EpicConfrontationMomentsComponent = ({
  argumentExchanges,
  onEpicMoment,
  academicMode = false
}: EpicConfrontationMomentsProps) => {
  const [activeEffects, setActiveEffects] = useState<Array<{ id: string; type: string; position: { x: number; y: number } }>>([]);
  const [epicMoments, setEpicMoments] = useState<EpicMoment[]>([]);
  const [clashZones, setClashZones] = useState<Array<{ id: string; intensity: number; x: number; y: number }>>([]);
  const arenaRef = useRef<HTMLDivElement>(null);

  // Analyze argument patterns for epic moments
  useEffect(() => {
    if (argumentExchanges.length < 2) return;

    const recent = argumentExchanges.slice(-2);
    const [prev, current] = recent;

    // Detect epic moment types
    if (prev.speaker !== current.speaker) {
      const strengthDifference = Math.abs(prev.strength - current.strength);
      
      if (strengthDifference > 30 && current.strength > 80) {
        // Critical counter-attack
        const epicMoment: EpicMoment = {
          id: `epic-${Date.now()}`,
          type: 'critical-hit',
          participants: [prev.speaker, current.speaker],
          intensity: current.strength,
          description: 'Devastating counter-argument landed!'
        };
        
        triggerEpicMoment(epicMoment);
      } else if (prev.strength > 70 && current.strength > 70) {
        // Epic clash
        const epicMoment: EpicMoment = {
          id: `clash-${Date.now()}`,
          type: 'clash',
          participants: [prev.speaker, current.speaker],
          intensity: Math.max(prev.strength, current.strength),
          description: 'Intellectual titans clash!'
        };
        
        triggerClashEffect(epicMoment);
      }
    }

    // Detect combo sequences
    const recentSameSpeaker = argumentExchanges.slice(-3).filter(arg => arg.speaker === current.speaker);
    if (recentSameSpeaker.length >= 3 && recentSameSpeaker.every(arg => arg.strength > 60)) {
      const comboMoment: EpicMoment = {
        id: `combo-${Date.now()}`,
        type: 'finishing-move',
        participants: [current.speaker, current.speaker === 'left' ? 'right' : 'left'],
        intensity: 95,
        description: 'Unstoppable argument combo!'
      };
      
      triggerComboEffect(comboMoment);
    }
  }, [argumentExchanges]);

  const triggerEpicMoment = (moment: EpicMoment) => {
    if (academicMode) {
      // Subtle professional effects only
      setActiveEffects(prev => [...prev, {
        id: moment.id,
        type: 'academic-highlight',
        position: { x: 50, y: 50 }
      }]);
    } else {
      // Full gaming effects
      setEpicMoments(prev => [...prev, moment]);
      onEpicMoment?.(moment);
      
      // Generate impact effects
      generateImpactEffects(moment);
    }

    // Auto-remove after duration
    setTimeout(() => {
      setActiveEffects(prev => prev.filter(effect => effect.id !== moment.id));
      setEpicMoments(prev => prev.filter(m => m.id !== moment.id));
    }, 3000);
  };

  const triggerClashEffect = (moment: EpicMoment) => {
    const clashX = 50; // Center of arena
    const clashY = 50;
    
    setClashZones(prev => [...prev, {
      id: moment.id,
      intensity: moment.intensity,
      x: clashX,
      y: clashY
    }]);

    triggerEpicMoment(moment);

    setTimeout(() => {
      setClashZones(prev => prev.filter(zone => zone.id !== moment.id));
    }, 2000);
  };

  const triggerComboEffect = (moment: EpicMoment) => {
    // Multiple rapid-fire effects
    for (let i = 0; i < 5; i++) {
      setTimeout(() => {
        setActiveEffects(prev => [...prev, {
          id: `${moment.id}-${i}`,
          type: 'combo-hit',
          position: { 
            x: moment.participants[0] === 'left' ? 20 + i * 15 : 80 - i * 15, 
            y: 40 + Math.random() * 20 
          }
        }]);
      }, i * 200);
    }

    triggerEpicMoment(moment);
  };

  const generateImpactEffects = (moment: EpicMoment) => {
    const effectCount = Math.floor(moment.intensity / 20);
    
    for (let i = 0; i < effectCount; i++) {
      setTimeout(() => {
        setActiveEffects(prev => [...prev, {
          id: `${moment.id}-impact-${i}`,
          type: moment.type,
          position: {
            x: 30 + Math.random() * 40,
            y: 30 + Math.random() * 40
          }
        }]);
      }, i * 100);
    }
  };

  const getEffectIcon = (effectType: string) => {
    switch (effectType) {
      case 'critical-hit': return Crown;
      case 'clash': return Swords;
      case 'perfect-counter': return Shield;
      case 'combo-breaker': return Target;
      case 'finishing-move': return Star;
      default: return Zap;
    }
  };

  const getEffectColor = (effectType: string) => {
    switch (effectType) {
      case 'critical-hit': return 'text-red-400';
      case 'clash': return 'text-yellow-400';
      case 'perfect-counter': return 'text-blue-400';
      case 'combo-breaker': return 'text-purple-400';
      case 'finishing-move': return 'text-green-400';
      default: return 'text-primary';
    }
  };

  if (academicMode) {
    return (
      <div className="absolute inset-0 pointer-events-none">
        {/* Minimal academic highlighting */}
        {activeEffects.map(effect => (
          <div
            key={effect.id}
            className="absolute w-2 h-2 bg-primary/30 rounded-full animate-ping"
            style={{
              left: `${effect.position.x}%`,
              top: `${effect.position.y}%`,
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div ref={arenaRef} className="absolute inset-0 pointer-events-none overflow-hidden">
      
      {/* Clash Zones */}
      {clashZones.map(zone => (
        <div
          key={zone.id}
          className="absolute"
          style={{
            left: `${zone.x}%`,
            top: `${zone.y}%`,
            transform: 'translate(-50%, -50%)'
          }}
        >
          {/* Shockwave Effect */}
          <div className="relative">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="absolute w-32 h-32 border-4 border-yellow-400/50 rounded-full"
                style={{
                  animation: `shockwave ${1 + i * 0.3}s ease-out`,
                  animationDelay: `${i * 0.2}s`
                }}
              />
            ))}
            
            {/* Center Flash */}
            <div className="w-8 h-8 bg-yellow-400 rounded-full animate-pulse shadow-lg shadow-yellow-400/50" />
          </div>
        </div>
      ))}

      {/* Active Effects */}
      {activeEffects.map(effect => {
        const IconComponent = getEffectIcon(effect.type);
        const colorClass = getEffectColor(effect.type);
        
        return (
          <div
            key={effect.id}
            className="absolute"
            style={{
              left: `${effect.position.x}%`,
              top: `${effect.position.y}%`,
              transform: 'translate(-50%, -50%)'
            }}
          >
            <div className={`${colorClass} animate-bounce`}>
              <IconComponent className="h-8 w-8 drop-shadow-lg" />
            </div>
          </div>
        );
      })}

      {/* Epic Moment Announcements */}
      {epicMoments.map(moment => (
        <div
          key={moment.id}
          className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10"
        >
          <div className="glass-panel px-6 py-3 border-2 border-yellow-400/50 bg-gradient-to-r from-yellow-500/20 to-orange-500/20">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-yellow-400/20 rounded-full animate-pulse">
                {(() => {
                  const IconComponent = getEffectIcon(moment.type);
                  return <IconComponent className="h-5 w-5 text-yellow-400" />;
                })()}
              </div>
              <div>
                <div className="font-bold text-yellow-400 uppercase tracking-wide">
                  {moment.type.replace('-', ' ')}
                </div>
                <div className="text-sm text-white">{moment.description}</div>
              </div>
            </div>
          </div>
        </div>
      ))}

      {/* Argument Impact Visualization */}
      {argumentExchanges.slice(-1).map(exchange => (
        <div
          key={exchange.id}
          className="absolute"
          style={{
            left: exchange.speaker === 'left' ? '20%' : '80%',
            top: '50%',
            transform: 'translate(-50%, -50%)'
          }}
        >
          {/* Argument Strength Indicator */}
          <div className={`
            w-16 h-16 rounded-full border-4 transition-all duration-300
            ${exchange.strength > 80 ? 'border-red-400 bg-red-400/20 animate-pulse' :
              exchange.strength > 60 ? 'border-yellow-400 bg-yellow-400/20' :
              'border-blue-400 bg-blue-400/20'}
          `}>
            <div className="w-full h-full flex items-center justify-center text-white font-bold">
              {exchange.strength}
            </div>
          </div>

          {/* Impact Waves */}
          {exchange.strength > 70 && (
            <div className="absolute inset-0">
              {[...Array(2)].map((_, i) => (
                <div
                  key={i}
                  className={`
                    absolute w-20 h-20 border-2 rounded-full
                    ${exchange.strength > 80 ? 'border-red-400/30' : 'border-yellow-400/30'}
                  `}
                  style={{
                    animation: `impact-wave ${1.5 + i * 0.5}s ease-out`,
                    animationDelay: `${i * 0.3}s`
                  }}
                />
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Custom Animations */}
      <style>{`
        @keyframes shockwave {
          0% {
            transform: scale(0);
            opacity: 1;
          }
          100% {
            transform: scale(3);
            opacity: 0;
          }
        }

        @keyframes impact-wave {
          0% {
            transform: scale(1);
            opacity: 0.8;
          }
          100% {
            transform: scale(2);
            opacity: 0;
          }
        }

        @keyframes epic-flash {
          0%, 100% { opacity: 0; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export const EpicConfrontationMoments = memo(EpicConfrontationMomentsComponent);