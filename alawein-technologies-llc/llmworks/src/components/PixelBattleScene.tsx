import { memo, useState, useEffect } from 'react';
import { Zap, Shield, Cpu, Target } from 'lucide-react';

interface BattleContender {
  id: string;
  name: string;
  type: 'reasoning' | 'creative' | 'analytical' | 'conversational';
  energy: number; // 0-100
  status: 'ready' | 'engaging' | 'analyzing' | 'completed';
  tacticalAdvantages: string[];
}

interface PixelBattleSceneProps {
  contenders: BattleContender[];
  battlePhase: 'preparation' | 'engagement' | 'analysis' | 'results';
  onPhaseComplete?: (phase: string) => void;
}

const PixelBattleSceneComponent = ({ 
  contenders, 
  battlePhase, 
  onPhaseComplete 
}: PixelBattleSceneProps) => {
  const [animationState, setAnimationState] = useState<'idle' | 'charging' | 'attacking' | 'defending'>('idle');
  const [energyFlows, setEnergyFlows] = useState<Array<{ id: string; intensity: number }>>([]);

  // Simulate battle progression
  useEffect(() => {
    const phaseTimer = setTimeout(() => {
      switch (battlePhase) {
        case 'preparation':
          setAnimationState('charging');
          simulateEnergyBuildup();
          break;
        case 'engagement':
          setAnimationState('attacking');
          simulateTacticalExchanges();
          break;
        case 'analysis':
          setAnimationState('defending');
          simulateAnalysisPhase();
          break;
        default:
          setAnimationState('idle');
      }
    }, 1000);

    return () => clearTimeout(phaseTimer);
  }, [battlePhase]);

  const simulateEnergyBuildup = () => {
    const flows = contenders.map(contender => ({
      id: contender.id,
      intensity: Math.random() * 0.5 + 0.5 // 50-100% intensity
    }));
    setEnergyFlows(flows);
  };

  const simulateTacticalExchanges = () => {
    const interval = setInterval(() => {
      setEnergyFlows(prev => prev.map(flow => ({
        ...flow,
        intensity: Math.random() * 0.8 + 0.2 // Dynamic fluctuation
      })));
    }, 800);

    setTimeout(() => clearInterval(interval), 5000);
  };

  const simulateAnalysisPhase = () => {
    // Gradual energy stabilization during analysis
    const stabilizationInterval = setInterval(() => {
      setEnergyFlows(prev => prev.map(flow => ({
        ...flow,
        intensity: flow.intensity * 0.95 + 0.05 // Converge to stable state
      })));
    }, 200);

    setTimeout(() => {
      clearInterval(stabilizationInterval);
      onPhaseComplete?.(battlePhase);
    }, 3000);
  };

  const getContenderIcon = (type: string) => {
    switch (type) {
      case 'reasoning': return Cpu;
      case 'creative': return Zap;
      case 'analytical': return Target;
      case 'conversational': return Shield;
      default: return Cpu;
    }
  };

  const getBattlePhaseDescription = () => {
    switch (battlePhase) {
      case 'preparation': return 'Systems Initializing...';
      case 'engagement': return 'Tactical Exchange Active';
      case 'analysis': return 'Performance Analysis';
      case 'results': return 'Evaluation Complete';
      default: return 'Standby Mode';
    }
  };

  return (
    <div className="pixel-battle-arena relative">
      {/* Arena Frame */}
      <div className="arena-frame p-6 rounded-lg bg-gradient-to-br from-background via-muted/10 to-background">
        
        {/* Battle Phase Indicator */}
        <div className="text-center mb-6">
          <div className="glass-minimal px-4 py-2 rounded-full inline-block">
            <div className="flex items-center gap-2 text-sm">
              <div className={`w-2 h-2 rounded-full animate-pulse ${
                battlePhase === 'engagement' ? 'bg-red-400' :
                battlePhase === 'analysis' ? 'bg-yellow-400' :
                battlePhase === 'results' ? 'bg-green-400' : 'bg-blue-400'
              }`} />
              <span className="font-mono text-primary">{getBattlePhaseDescription()}</span>
            </div>
          </div>
        </div>

        {/* Battle Arena */}
        <div className="relative h-48 bg-gradient-to-b from-muted/5 to-muted/20 rounded-lg border border-primary/20 overflow-hidden">
          
          {/* Animated Grid Background */}
          <div className="absolute inset-0 opacity-20">
            <svg width="100%" height="100%" className="absolute inset-0">
              <defs>
                <pattern 
                  id="battle-grid" 
                  width="20" 
                  height="20" 
                  patternUnits="userSpaceOnUse"
                  className="text-primary/30"
                >
                  <path 
                    d="M 20 0 L 0 0 0 20" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="0.5"
                  />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#battle-grid)" />
            </svg>
          </div>

          {/* Energy Flow Visualization */}
          <div className="absolute inset-0">
            {energyFlows.map((flow, index) => (
              <div
                key={flow.id}
                className="absolute w-1 bg-gradient-to-t from-primary to-secondary rounded-full"
                style={{
                  left: `${20 + index * 30}%`,
                  height: `${flow.intensity * 80}%`,
                  bottom: '10%',
                  opacity: flow.intensity,
                  animation: `energy-pulse ${2 + Math.random()}s ease-in-out infinite`
                }}
              />
            ))}
          </div>

          {/* Contender Positions */}
          <div className="absolute inset-0 flex items-end justify-around p-4">
            {contenders.map((contender, index) => {
              const IconComponent = getContenderIcon(contender.type);
              const energyFlow = energyFlows.find(f => f.id === contender.id);
              
              return (
                <div key={contender.id} className="text-center space-y-2">
                  
                  {/* Contender Avatar */}
                  <div className={`
                    relative p-4 rounded-lg glass-minimal border transition-all duration-500
                    ${animationState === 'charging' ? 'animate-pulse shadow-lg shadow-primary/20' :
                      animationState === 'attacking' ? 'animate-bounce shadow-xl shadow-red-500/20' :
                      animationState === 'defending' ? 'animate-pulse shadow-lg shadow-yellow-500/20' : ''}
                  `}>
                    <IconComponent className={`h-8 w-8 transition-colors duration-300 ${
                      contender.status === 'engaging' ? 'text-red-400' :
                      contender.status === 'analyzing' ? 'text-yellow-400' :
                      contender.status === 'completed' ? 'text-green-400' : 'text-primary'
                    }`} />
                    
                    {/* Energy Level Indicator */}
                    <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-background border border-primary/30 flex items-center justify-center">
                      <div 
                        className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                          (energyFlow?.intensity || 0) > 0.7 ? 'bg-green-400' :
                          (energyFlow?.intensity || 0) > 0.4 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}
                      />
                    </div>
                  </div>

                  {/* Contender Info */}
                  <div className="text-xs">
                    <div className="font-mono text-primary font-semibold">{contender.name}</div>
                    <div className="text-muted-foreground capitalize">{contender.type}</div>
                    
                    {/* Tactical Advantages */}
                    {contender.tacticalAdvantages.length > 0 && (
                      <div className="mt-1 flex flex-wrap justify-center gap-1">
                        {contender.tacticalAdvantages.slice(0, 2).map(advantage => (
                          <div 
                            key={advantage}
                            className="px-1 py-0.5 bg-primary/10 text-primary rounded text-xs font-mono"
                          >
                            {advantage}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Tactical Overlay Effects */}
          {animationState === 'attacking' && (
            <div className="absolute inset-0 pointer-events-none">
              {[...Array(3)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-2 h-2 bg-red-400 rounded-full animate-ping"
                  style={{
                    left: `${20 + i * 30}%`,
                    top: `${30 + i * 15}%`,
                    animationDelay: `${i * 0.2}s`
                  }}
                />
              ))}
            </div>
          )}
        </div>

        {/* Battle Statistics */}
        <div className="mt-4 grid grid-cols-3 gap-4 text-center">
          <div className="glass-minimal p-3 rounded-lg">
            <div className="text-lg font-mono text-primary">
              {battlePhase === 'engagement' ? 'ACTIVE' : 
               battlePhase === 'analysis' ? 'EVAL' :
               battlePhase === 'results' ? 'DONE' : 'READY'}
            </div>
            <div className="text-xs text-muted-foreground">Phase</div>
          </div>
          
          <div className="glass-minimal p-3 rounded-lg">
            <div className="text-lg font-mono text-secondary">
              {contenders.length}v{contenders.length > 1 ? contenders.length : 1}
            </div>
            <div className="text-xs text-muted-foreground">Format</div>
          </div>
          
          <div className="glass-minimal p-3 rounded-lg">
            <div className="text-lg font-mono text-accent">
              {energyFlows.reduce((sum, flow) => sum + flow.intensity, 0).toFixed(1)}
            </div>
            <div className="text-xs text-muted-foreground">Energy</div>
          </div>
        </div>
      </div>

      {/* Custom CSS for Arena Effects */}
      <style>{`
        .arena-frame {
          background: 
            linear-gradient(135deg, transparent 15px, hsl(var(--primary) / 0.1) 15px),
            linear-gradient(-135deg, transparent 15px, hsl(var(--secondary) / 0.05) 15px);
          border: 2px solid hsl(var(--primary) / 0.3);
          position: relative;
        }

        .arena-frame::before {
          content: "";
          position: absolute;
          inset: -3px;
          background: conic-gradient(from 0deg, 
            hsl(var(--primary) / 0.2) 0deg, 
            transparent 90deg, 
            hsl(var(--secondary) / 0.2) 180deg, 
            transparent 270deg);
          z-index: -1;
          border-radius: inherit;
          animation: rotate-border 10s linear infinite;
        }

        @keyframes energy-pulse {
          0%, 100% { opacity: 0.6; transform: scaleY(1); }
          50% { opacity: 1; transform: scaleY(1.2); }
        }

        @keyframes rotate-border {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .pixel-battle-arena {
          image-rendering: -moz-crisp-edges;
          image-rendering: -webkit-crisp-edges;
          image-rendering: pixelated;
          image-rendering: crisp-edges;
        }
      `}</style>
    </div>
  );
};

export const PixelBattleScene = memo(PixelBattleSceneComponent);