import { memo, useState, useEffect, useRef } from 'react';
import { Shield, Zap, Target, Swords, Crown } from 'lucide-react';

interface ArgumentImpact {
  id: string;
  type: 'direct-hit' | 'deflection' | 'critical-strike' | 'combo-chain' | 'devastating-blow';
  strength: number; // 0-100
  position: { x: number; y: number };
  direction: 'left-to-right' | 'right-to-left';
  timestamp: Date;
  decayTime: number;
}

interface ArgumentImpactSystemProps {
  onImpact: (impact: ArgumentImpact) => void;
  academicMode?: boolean;
  soundEnabled?: boolean;
}

const ArgumentImpactSystemComponent = ({ 
  onImpact, 
  academicMode = false, 
  soundEnabled = true 
}: ArgumentImpactSystemProps) => {
  const [activeImpacts, setActiveImpacts] = useState<ArgumentImpact[]>([]);
  const [shockwaves, setShockwaves] = useState<Array<{ id: string; x: number; y: number; intensity: number }>>([]);
  const [deflectionShields, setDeflectionShields] = useState<Array<{ id: string; side: 'left' | 'right'; strength: number }>>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // Simulate argument impacts for demonstration
  useEffect(() => {
    if (academicMode) return;

    const impactInterval = setInterval(() => {
      if (Math.random() < 0.3) { // 30% chance per 2 seconds
        triggerArgumentImpact();
      }
    }, 2000);

    return () => clearInterval(impactInterval);
  }, [academicMode]);

  const triggerArgumentImpact = () => {
    const impactTypes: ArgumentImpact['type'][] = [
      'direct-hit', 'deflection', 'critical-strike', 'combo-chain', 'devastating-blow'
    ];
    
    const type = impactTypes[Math.floor(Math.random() * impactTypes.length)];
    const strength = Math.floor(Math.random() * 40) + 60; // 60-100 strength
    const direction: ArgumentImpact['direction'] = Math.random() < 0.5 ? 'left-to-right' : 'right-to-left';
    
    const impact: ArgumentImpact = {
      id: `impact-${Date.now()}`,
      type,
      strength,
      position: {
        x: direction === 'left-to-right' ? 20 : 80,
        y: 50 + (Math.random() - 0.5) * 30
      },
      direction,
      timestamp: new Date(),
      decayTime: type === 'devastating-blow' ? 4000 : type === 'critical-strike' ? 3000 : 2000
    };

    setActiveImpacts(prev => [...prev, impact]);
    onImpact(impact);

    // Generate visual effects based on impact type
    switch (type) {
      case 'direct-hit':
        generateDirectHitEffect(impact);
        break;
      case 'deflection':
        generateDeflectionEffect(impact);
        break;
      case 'critical-strike':
        generateCriticalStrikeEffect(impact);
        break;
      case 'combo-chain':
        generateComboChainEffect(impact);
        break;
      case 'devastating-blow':
        generateDevastatingBlowEffect(impact);
        break;
    }

    // Remove impact after decay time
    setTimeout(() => {
      setActiveImpacts(prev => prev.filter(i => i.id !== impact.id));
    }, impact.decayTime);
  };

  const generateDirectHitEffect = (impact: ArgumentImpact) => {
    const targetX = impact.direction === 'left-to-right' ? 80 : 20;
    const shockwave = {
      id: `shockwave-${impact.id}`,
      x: targetX,
      y: impact.position.y,
      intensity: impact.strength
    };
    
    setShockwaves(prev => [...prev, shockwave]);
    
    setTimeout(() => {
      setShockwaves(prev => prev.filter(s => s.id !== shockwave.id));
    }, 1500);
  };

  const generateDeflectionEffect = (impact: ArgumentImpact) => {
    const side = impact.direction === 'left-to-right' ? 'right' : 'left';
    const shield = {
      id: `shield-${impact.id}`,
      side: side as 'left' | 'right',
      strength: impact.strength
    };
    
    setDeflectionShields(prev => [...prev, shield]);
    
    setTimeout(() => {
      setDeflectionShields(prev => prev.filter(s => s.id !== shield.id));
    }, 2000);
  };

  const generateCriticalStrikeEffect = (impact: ArgumentImpact) => {
    // Multiple shockwaves for critical strikes
    for (let i = 0; i < 3; i++) {
      setTimeout(() => {
        generateDirectHitEffect({ ...impact, id: `${impact.id}-${i}` });
      }, i * 200);
    }
  };

  const generateComboChainEffect = (impact: ArgumentImpact) => {
    // Rapid succession of smaller impacts
    const comboCount = Math.floor(impact.strength / 20);
    for (let i = 0; i < comboCount; i++) {
      setTimeout(() => {
        const comboImpact = {
          ...impact,
          id: `${impact.id}-combo-${i}`,
          position: {
            x: impact.position.x,
            y: impact.position.y + (Math.random() - 0.5) * 20
          },
          strength: Math.floor(impact.strength / comboCount)
        };
        generateDirectHitEffect(comboImpact);
      }, i * 150);
    }
  };

  const generateDevastatingBlowEffect = (impact: ArgumentImpact) => {
    // Screen-wide shockwave effect
    for (let i = 0; i < 5; i++) {
      setTimeout(() => {
        const devastationWave = {
          id: `devastation-${impact.id}-${i}`,
          x: 50,
          y: 50,
          intensity: impact.strength + i * 10
        };
        setShockwaves(prev => [...prev, devastationWave]);
        
        setTimeout(() => {
          setShockwaves(prev => prev.filter(s => s.id !== devastationWave.id));
        }, 2000 + i * 300);
      }, i * 100);
    }
  };

  const getImpactIcon = (type: ArgumentImpact['type']) => {
    switch (type) {
      case 'direct-hit': return Target;
      case 'deflection': return Shield;
      case 'critical-strike': return Swords;
      case 'combo-chain': return Zap;
      case 'devastating-blow': return Crown;
    }
  };

  const getImpactColor = (type: ArgumentImpact['type']) => {
    switch (type) {
      case 'direct-hit': return 'text-yellow-400';
      case 'deflection': return 'text-blue-400';
      case 'critical-strike': return 'text-red-400';
      case 'combo-chain': return 'text-purple-400';
      case 'devastating-blow': return 'text-orange-400';
    }
  };

  if (academicMode) {
    return (
      <div className="absolute inset-0 pointer-events-none">
        {/* Minimal academic indicators */}
        {activeImpacts.map(impact => (
          <div
            key={impact.id}
            className="absolute w-3 h-3 bg-primary/30 rounded-full animate-pulse"
            style={{
              left: `${impact.position.x}%`,
              top: `${impact.position.y}%`,
              transform: 'translate(-50%, -50%)'
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {/* Canvas for GPU-accelerated effects */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ mixBlendMode: 'screen' }}
      />

      {/* Active Argument Impacts */}
      {activeImpacts.map(impact => {
        const IconComponent = getImpactIcon(impact.type);
        const colorClass = getImpactColor(impact.type);
        
        return (
          <div
            key={impact.id}
            className="absolute"
            style={{
              left: `${impact.position.x}%`,
              top: `${impact.position.y}%`,
              transform: 'translate(-50%, -50%)',
              animation: `impact-${impact.type} ${impact.decayTime}ms ease-out`
            }}
          >
            <div className={`${colorClass} animate-bounce`}>
              <IconComponent 
                className="h-8 w-8 drop-shadow-lg" 
                style={{ filter: `brightness(${impact.strength / 50})` }}
              />
            </div>
            
            {/* Impact trajectory line */}
            <div
              className={`absolute top-1/2 h-0.5 bg-gradient-to-r ${
                impact.direction === 'left-to-right' 
                  ? 'from-primary via-accent to-transparent' 
                  : 'from-transparent via-accent to-primary'
              }`}
              style={{
                width: '200px',
                left: impact.direction === 'left-to-right' ? '0' : '-200px',
                transform: 'translateY(-50%)',
                animation: 'trajectory-line 1s ease-out'
              }}
            />
          </div>
        );
      })}

      {/* Shockwave Effects */}
      {shockwaves.map(wave => (
        <div
          key={wave.id}
          className="absolute"
          style={{
            left: `${wave.x}%`,
            top: `${wave.y}%`,
            transform: 'translate(-50%, -50%)'
          }}
        >
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className={`absolute border-4 rounded-full ${
                wave.intensity > 80 ? 'border-red-400/40' : 
                wave.intensity > 60 ? 'border-yellow-400/40' : 'border-blue-400/40'
              }`}
              style={{
                width: `${80 + i * 40}px`,
                height: `${80 + i * 40}px`,
                left: '50%',
                top: '50%',
                transform: 'translate(-50%, -50%)',
                animation: `shockwave-expand ${1.5 + i * 0.3}s ease-out`,
                animationDelay: `${i * 0.1}s`
              }}
            />
          ))}
          
          {/* Center flash */}
          <div className={`w-6 h-6 rounded-full animate-pulse shadow-lg ${
            wave.intensity > 80 ? 'bg-red-400 shadow-red-400/50' : 
            wave.intensity > 60 ? 'bg-yellow-400 shadow-yellow-400/50' : 'bg-blue-400 shadow-blue-400/50'
          }`} />
        </div>
      ))}

      {/* Deflection Shields */}
      {deflectionShields.map(shield => (
        <div
          key={shield.id}
          className="absolute"
          style={{
            left: shield.side === 'left' ? '15%' : '85%',
            top: '50%',
            transform: 'translate(-50%, -50%)'
          }}
        >
          <div className={`
            w-20 h-32 rounded-lg border-4 border-blue-400/60
            bg-gradient-to-b from-blue-400/20 to-cyan-400/10
            animate-pulse shadow-2xl shadow-blue-400/30
          `}>
            <div className="absolute inset-2 rounded-md bg-gradient-to-b from-white/20 to-transparent" />
            
            {/* Shield energy ripples */}
            {[...Array(shield.strength > 70 ? 4 : 2)].map((_, i) => (
              <div
                key={i}
                className="absolute inset-0 border-2 border-blue-400/30 rounded-lg"
                style={{
                  animation: `shield-ripple ${1 + i * 0.2}s ease-out infinite`,
                  animationDelay: `${i * 0.1}s`
                }}
              />
            ))}
          </div>
        </div>
      ))}

      {/* Custom CSS animations */}
      <style>{`
        @keyframes impact-direct-hit {
          0% { 
            transform: translate(-50%, -50%) scale(0.5);
            opacity: 0;
          }
          20% { 
            transform: translate(-50%, -50%) scale(1.5);
            opacity: 1;
          }
          100% { 
            transform: translate(-50%, -50%) scale(1);
            opacity: 0;
          }
        }

        @keyframes impact-critical-strike {
          0% { 
            transform: translate(-50%, -50%) scale(0.3) rotate(0deg);
            opacity: 0;
          }
          30% { 
            transform: translate(-50%, -50%) scale(2) rotate(180deg);
            opacity: 1;
          }
          100% { 
            transform: translate(-50%, -50%) scale(1.2) rotate(360deg);
            opacity: 0;
          }
        }

        @keyframes impact-devastating-blow {
          0% { 
            transform: translate(-50%, -50%) scale(0.2);
            opacity: 0;
            filter: brightness(2) saturate(2);
          }
          50% { 
            transform: translate(-50%, -50%) scale(3);
            opacity: 1;
            filter: brightness(3) saturate(1.5);
          }
          100% { 
            transform: translate(-50%, -50%) scale(1.5);
            opacity: 0;
            filter: brightness(1) saturate(1);
          }
        }

        @keyframes shockwave-expand {
          0% {
            transform: translate(-50%, -50%) scale(0);
            opacity: 1;
          }
          100% {
            transform: translate(-50%, -50%) scale(2);
            opacity: 0;
          }
        }

        @keyframes shield-ripple {
          0% { 
            transform: scale(1);
            opacity: 0.8;
          }
          100% { 
            transform: scale(1.3);
            opacity: 0;
          }
        }

        @keyframes trajectory-line {
          0% { 
            transform: translateY(-50%) scaleX(0);
            opacity: 0;
          }
          50% { 
            transform: translateY(-50%) scaleX(1);
            opacity: 1;
          }
          100% { 
            transform: translateY(-50%) scaleX(1);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
};

export const ArgumentImpactSystem = memo(ArgumentImpactSystemComponent);