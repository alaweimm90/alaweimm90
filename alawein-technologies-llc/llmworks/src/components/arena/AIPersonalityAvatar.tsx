import { memo, useState, useEffect, useRef } from 'react';
import { Brain, Sparkles, Cpu, MessageSquare, Zap, Shield, Swords, Target } from 'lucide-react';

export type AIPersonality = 'analytical' | 'creative' | 'speed' | 'conversational' | 'strategic';
export type BattleState = 'idle' | 'thinking' | 'arguing' | 'defending' | 'counter-attacking' | 'victory' | 'defeat';

interface AIPersonalityAvatarProps {
  personality: AIPersonality;
  battleState: BattleState;
  energy: number; // 0-100
  name: string;
  isActive: boolean;
  onStateChange?: (state: BattleState) => void;
  position?: 'left' | 'right';
}

const AIPersonalityAvatarComponent = ({
  personality,
  battleState,
  energy,
  name,
  isActive,
  onStateChange,
  position = 'left'
}: AIPersonalityAvatarProps) => {
  const [particleEffects, setParticleEffects] = useState<Array<{ id: string; x: number; y: number }>>([]);
  const [pulseIntensity, setPulseIntensity] = useState(0);
  const avatarRef = useRef<HTMLDivElement>(null);

  // Personality-specific visual configurations
  const getPersonalityConfig = () => {
    switch (personality) {
      case 'analytical':
        return {
          icon: Brain,
          primaryColor: 'from-blue-500 to-cyan-500',
          secondaryColor: 'from-blue-400/20 to-cyan-400/20',
          particleColor: 'bg-blue-400',
          shape: 'polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%)', // Pentagon
          description: 'Logic-driven, methodical approach'
        };
      case 'creative':
        return {
          icon: Sparkles,
          primaryColor: 'from-purple-500 to-pink-500',
          secondaryColor: 'from-purple-400/20 to-pink-400/20',
          particleColor: 'bg-purple-400',
          shape: 'ellipse(40% 50% at 50% 50%)', // Organic
          description: 'Innovative, artistic thinking'
        };
      case 'speed':
        return {
          icon: Zap,
          primaryColor: 'from-yellow-500 to-orange-500',
          secondaryColor: 'from-yellow-400/20 to-orange-400/20',
          particleColor: 'bg-yellow-400',
          shape: 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)', // Star
          description: 'Lightning-fast responses'
        };
      case 'conversational':
        return {
          icon: MessageSquare,
          primaryColor: 'from-green-500 to-emerald-500',
          secondaryColor: 'from-green-400/20 to-emerald-400/20',
          particleColor: 'bg-green-400',
          shape: 'circle(45%)', // Rounded
          description: 'Empathetic dialogue'
        };
      case 'strategic':
        return {
          icon: Target,
          primaryColor: 'from-red-500 to-rose-500',
          secondaryColor: 'from-red-400/20 to-rose-400/20',
          particleColor: 'bg-red-400',
          shape: 'polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)', // Hexagon
          description: 'Tactical precision'
        };
    }
  };

  const config = getPersonalityConfig();
  const IconComponent = config.icon;

  // Battle state animations
  useEffect(() => {
    let animationInterval: NodeJS.Timeout;

    switch (battleState) {
      case 'thinking':
        setPulseIntensity(0.3);
        animationInterval = setInterval(() => {
          generateThinkingParticles();
        }, 500);
        break;
      
      case 'arguing':
        setPulseIntensity(0.7);
        animationInterval = setInterval(() => {
          generateAttackParticles();
        }, 200);
        break;
      
      case 'defending':
        setPulseIntensity(0.5);
        generateShieldEffect();
        break;
      
      case 'counter-attacking':
        setPulseIntensity(1);
        animationInterval = setInterval(() => {
          generateCounterParticles();
        }, 150);
        break;
      
      case 'victory':
        setPulseIntensity(1);
        generateVictoryParticles();
        break;
      
      case 'defeat':
        setPulseIntensity(0.1);
        break;
      
      default:
        setPulseIntensity(0.2);
    }

    return () => {
      if (animationInterval) clearInterval(animationInterval);
    };
  }, [battleState]);

  const generateThinkingParticles = () => {
    const newParticles = Array.from({ length: 3 }, () => ({
      id: `think-${Date.now()}-${Math.random()}`,
      x: Math.random() * 100 - 50,
      y: -20
    }));
    
    setParticleEffects(prev => [...prev, ...newParticles]);
    setTimeout(() => {
      setParticleEffects(prev => prev.filter(p => !newParticles.find(np => np.id === p.id)));
    }, 2000);
  };

  const generateAttackParticles = () => {
    const newParticles = Array.from({ length: 5 }, () => ({
      id: `attack-${Date.now()}-${Math.random()}`,
      x: position === 'left' ? 100 : -100,
      y: Math.random() * 50 - 25
    }));
    
    setParticleEffects(prev => [...prev, ...newParticles]);
    setTimeout(() => {
      setParticleEffects(prev => prev.filter(p => !newParticles.find(np => np.id === p.id)));
    }, 1000);
  };

  const generateShieldEffect = () => {
    // Shield visualization handled by CSS animation
  };

  const generateCounterParticles = () => {
    const newParticles = Array.from({ length: 8 }, () => ({
      id: `counter-${Date.now()}-${Math.random()}`,
      x: Math.random() * 200 - 100,
      y: Math.random() * 200 - 100
    }));
    
    setParticleEffects(prev => [...prev, ...newParticles]);
    setTimeout(() => {
      setParticleEffects(prev => prev.filter(p => !newParticles.find(np => np.id === p.id)));
    }, 1500);
  };

  const generateVictoryParticles = () => {
    const newParticles = Array.from({ length: 20 }, () => ({
      id: `victory-${Date.now()}-${Math.random()}`,
      x: Math.random() * 300 - 150,
      y: Math.random() * 300 - 150
    }));
    
    setParticleEffects(prev => [...prev, ...newParticles]);
    setTimeout(() => {
      setParticleEffects(prev => prev.filter(p => !newParticles.find(np => np.id === p.id)));
    }, 3000);
  };

  const getBattleStateIcon = () => {
    switch (battleState) {
      case 'arguing': return <Swords className="h-4 w-4" />;
      case 'defending': return <Shield className="h-4 w-4" />;
      case 'counter-attacking': return <Target className="h-4 w-4" />;
      default: return null;
    }
  };

  return (
    <div 
      ref={avatarRef}
      className={`relative ${position === 'right' ? 'scale-x-[-1]' : ''}`}
    >
      {/* Main Avatar Container */}
      <div className={`
        relative w-32 h-32 transition-all duration-300
        ${isActive ? 'scale-110' : 'scale-100'}
        ${battleState === 'defeat' ? 'opacity-50 grayscale' : ''}
      `}>
        
        {/* Energy Shield (for defending state) */}
        {battleState === 'defending' && (
          <div className={`
            absolute inset-[-20px] rounded-full
            bg-gradient-to-r ${config.secondaryColor}
            animate-pulse border-2 border-current
          `} />
        )}

        {/* Main Avatar Shape */}
        <div 
          className={`
            absolute inset-0 bg-gradient-to-br ${config.primaryColor}
            shadow-2xl transition-all duration-300
            ${isActive ? 'shadow-current/50' : 'shadow-current/20'}
          `}
          style={{
            clipPath: config.shape,
            filter: `brightness(${1 + pulseIntensity * 0.5})`,
            transform: `scale(${1 + pulseIntensity * 0.1})`
          }}
        >
          {/* Inner Glow */}
          <div className={`
            absolute inset-2 bg-gradient-to-br ${config.secondaryColor}
            animate-pulse
          `} 
          style={{ 
            clipPath: config.shape,
            animationDuration: `${2 - pulseIntensity}s`
          }} />
        </div>

        {/* Central Icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className={`
            p-3 rounded-full glass-panel
            ${battleState === 'victory' ? 'animate-bounce' : ''}
            ${battleState === 'thinking' ? 'animate-pulse' : ''}
          `}>
            <IconComponent className={`
              h-8 w-8 text-white
              ${battleState === 'arguing' ? 'animate-pulse' : ''}
            `} />
          </div>
        </div>

        {/* Energy Level Bar */}
        <div className="absolute bottom-[-8px] left-1/2 transform -translate-x-1/2 w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
          <div 
            className={`h-full bg-gradient-to-r ${config.primaryColor} transition-all duration-300`}
            style={{ width: `${energy}%` }}
          />
        </div>

        {/* Battle State Indicator */}
        {getBattleStateIcon() && (
          <div className="absolute top-0 right-0 p-1 bg-background rounded-full border-2 border-current">
            {getBattleStateIcon()}
          </div>
        )}

        {/* Particle Effects */}
        {particleEffects.map(particle => (
          <div
            key={particle.id}
            className={`absolute w-2 h-2 ${config.particleColor} rounded-full`}
            style={{
              left: '50%',
              top: '50%',
              transform: `translate(${particle.x}px, ${particle.y}px)`,
              animation: battleState === 'arguing' 
                ? 'particle-attack 1s ease-out forwards'
                : battleState === 'victory'
                ? 'particle-explode 3s ease-out forwards'
                : 'particle-float 2s ease-out forwards'
            }}
          />
        ))}
      </div>

      {/* Name and Status */}
      <div className={`text-center mt-4 ${position === 'right' ? 'scale-x-[-1]' : ''}`}>
        <div className="font-bold text-sm">{name}</div>
        <div className="text-xs text-muted-foreground capitalize">{battleState}</div>
        <div className="text-xs text-muted-foreground mt-1">{config.description}</div>
      </div>

      {/* Custom animations */}
      <style>{`
        @keyframes particle-float {
          0% {
            opacity: 1;
            transform: translate(0, 0) scale(1);
          }
          100% {
            opacity: 0;
            transform: translate(var(--end-x, 0), -50px) scale(0.5);
          }
        }

        @keyframes particle-attack {
          0% {
            opacity: 1;
            transform: translate(0, 0) scale(1);
          }
          100% {
            opacity: 0;
            transform: translate(${position === 'left' ? '200px' : '-200px'}, 0) scale(2);
          }
        }

        @keyframes particle-explode {
          0% {
            opacity: 1;
            transform: translate(0, 0) scale(1);
          }
          50% {
            opacity: 1;
            transform: translate(var(--end-x, 0), var(--end-y, 0)) scale(1.5);
          }
          100% {
            opacity: 0;
            transform: translate(calc(var(--end-x, 0) * 2), calc(var(--end-y, 0) * 2)) scale(0);
          }
        }
      `}</style>
    </div>
  );
};

export const AIPersonalityAvatar = memo(AIPersonalityAvatarComponent);