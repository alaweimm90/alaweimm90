import { memo, useState, useEffect, useMemo } from 'react';
import { Cloud, Zap, Leaf, Brain, Shield, Globe, Cpu, Heart } from 'lucide-react';

interface DebateTopic {
  text: string;
  category: 'technology' | 'environment' | 'philosophy' | 'politics' | 'science' | 'ethics' | 'economics' | 'health';
  intensity: 'calm' | 'moderate' | 'heated' | 'explosive';
}

interface EnvironmentalEffect {
  id: string;
  type: 'particle-field' | 'background-shift' | 'ambient-glow' | 'dynamic-elements' | 'contextual-icons';
  category: DebateTopic['category'];
  intensity: number;
  duration: number;
}

interface EnvironmentalEffectsProps {
  topic: DebateTopic;
  debatePhase: 'setup' | 'opening' | 'exchange' | 'closing' | 'judgment' | 'complete';
  argumentIntensity: number; // 0-100 based on recent argument strength
  academicMode?: boolean;
}

const EnvironmentalEffectsComponent = ({
  topic,
  debatePhase,
  argumentIntensity,
  academicMode = false
}: EnvironmentalEffectsProps) => {
  const [activeEffects, setActiveEffects] = useState<EnvironmentalEffect[]>([]);
  const [backgroundParticles, setBackgroundParticles] = useState<Array<{ id: string; x: number; y: number; type: string }>>([]);

  // Category-specific visual configurations
  const categoryConfig = useMemo(() => {
    switch (topic.category) {
      case 'technology':
        return {
          primaryColor: 'from-blue-500 to-cyan-500',
          secondaryColor: 'from-blue-400/20 to-cyan-400/20',
          particleColor: 'bg-blue-400',
          ambientColor: 'rgba(37, 99, 235, 0.1)',
          icon: Cpu,
          particles: ['binary', 'circuit', 'data-flow']
        };
      case 'environment':
        return {
          primaryColor: 'from-green-500 to-emerald-500',
          secondaryColor: 'from-green-400/20 to-emerald-400/20',
          particleColor: 'bg-green-400',
          ambientColor: 'rgba(34, 197, 94, 0.1)',
          icon: Leaf,
          particles: ['leaf', 'water-drop', 'wind']
        };
      case 'philosophy':
        return {
          primaryColor: 'from-purple-500 to-violet-500',
          secondaryColor: 'from-purple-400/20 to-violet-400/20',
          particleColor: 'bg-purple-400',
          ambientColor: 'rgba(147, 51, 234, 0.1)',
          icon: Brain,
          particles: ['thought-bubble', 'abstract-symbol', 'infinity']
        };
      case 'politics':
        return {
          primaryColor: 'from-red-500 to-rose-500',
          secondaryColor: 'from-red-400/20 to-rose-400/20',
          particleColor: 'bg-red-400',
          ambientColor: 'rgba(239, 68, 68, 0.1)',
          icon: Shield,
          particles: ['flag', 'balance', 'vote']
        };
      case 'science':
        return {
          primaryColor: 'from-indigo-500 to-blue-600',
          secondaryColor: 'from-indigo-400/20 to-blue-400/20',
          particleColor: 'bg-indigo-400',
          ambientColor: 'rgba(99, 102, 241, 0.1)',
          icon: Zap,
          particles: ['atom', 'molecule', 'formula']
        };
      case 'ethics':
        return {
          primaryColor: 'from-amber-500 to-yellow-500',
          secondaryColor: 'from-amber-400/20 to-yellow-400/20',
          particleColor: 'bg-amber-400',
          ambientColor: 'rgba(245, 158, 11, 0.1)',
          icon: Heart,
          particles: ['balance-scale', 'heart', 'light-ray']
        };
      case 'economics':
        return {
          primaryColor: 'from-emerald-600 to-green-600',
          secondaryColor: 'from-emerald-400/20 to-green-400/20',
          particleColor: 'bg-emerald-400',
          ambientColor: 'rgba(16, 185, 129, 0.1)',
          icon: Globe,
          particles: ['coin', 'chart', 'trend-line']
        };
      case 'health':
        return {
          primaryColor: 'from-pink-500 to-rose-500',
          secondaryColor: 'from-pink-400/20 to-rose-400/20',
          particleColor: 'bg-pink-400',
          ambientColor: 'rgba(236, 72, 153, 0.1)',
          icon: Heart,
          particles: ['pulse', 'DNA', 'cell']
        };
      default:
        return {
          primaryColor: 'from-gray-500 to-slate-500',
          secondaryColor: 'from-gray-400/20 to-slate-400/20',
          particleColor: 'bg-gray-400',
          ambientColor: 'rgba(107, 114, 128, 0.1)',
          icon: Brain,
          particles: ['dot', 'line', 'circle']
        };
    }
  }, [topic.category]);

  // Generate contextual particles based on topic and debate intensity
  useEffect(() => {
    if (academicMode) return;

    const generateParticles = () => {
      const particleCount = Math.floor(argumentIntensity / 20) + 2; // 2-7 particles
      const newParticles = Array.from({ length: particleCount }, () => ({
        id: `particle-${Date.now()}-${Math.random()}`,
        x: Math.random() * 100,
        y: Math.random() * 100,
        type: categoryConfig.particles[Math.floor(Math.random() * categoryConfig.particles.length)]
      }));

      setBackgroundParticles(prev => [...prev, ...newParticles]);

      // Remove particles after animation
      setTimeout(() => {
        setBackgroundParticles(prev => 
          prev.filter(p => !newParticles.find(np => np.id === p.id))
        );
      }, 8000);
    };

    // Generate particles based on debate phase intensity
    const phaseMultiplier = {
      'setup': 0.2,
      'opening': 0.5,
      'exchange': 1.0,
      'closing': 0.8,
      'judgment': 0.3,
      'complete': 0.1
    }[debatePhase];

    const shouldGenerate = Math.random() < (argumentIntensity / 100) * phaseMultiplier;
    
    if (shouldGenerate) {
      generateParticles();
    }
  }, [argumentIntensity, debatePhase, academicMode, categoryConfig.particles]);

  // Create dynamic environmental effects based on topic intensity
  const getIntensityModifier = () => {
    const baseIntensity = {
      'calm': 0.3,
      'moderate': 0.6,
      'heated': 0.8,
      'explosive': 1.0
    }[topic.intensity];

    const phaseIntensity = {
      'setup': 0.2,
      'opening': 0.4,
      'exchange': 1.0,
      'closing': 0.7,
      'judgment': 0.5,
      'complete': 0.2
    }[debatePhase];

    return baseIntensity * phaseIntensity * (argumentIntensity / 100);
  };

  const renderCategorySpecificElements = () => {
    const IconComponent = categoryConfig.icon;
    const intensity = getIntensityModifier();

    switch (topic.category) {
      case 'technology':
        return (
          <>
            {/* Digital grid overlay */}
            <div 
              className="absolute inset-0 opacity-10"
              style={{
                backgroundImage: `
                  linear-gradient(rgba(37, 99, 235, ${intensity * 0.3}) 1px, transparent 1px),
                  linear-gradient(90deg, rgba(37, 99, 235, ${intensity * 0.3}) 1px, transparent 1px)
                `,
                backgroundSize: '40px 40px',
                animation: 'grid-pulse 4s ease-in-out infinite'
              }}
            />
            {/* Circuit trace lines */}
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className={`absolute h-px bg-gradient-to-r ${categoryConfig.primaryColor} opacity-30`}
                style={{
                  width: '200px',
                  top: `${30 + i * 20}%`,
                  left: `-100px`,
                  animation: `circuit-flow ${3 + i}s linear infinite`
                }}
              />
            ))}
          </>
        );

      case 'environment':
        return (
          <>
            {/* Organic flowing patterns */}
            <div className="absolute inset-0">
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  className={`absolute w-2 h-2 rounded-full ${categoryConfig.particleColor} opacity-20`}
                  style={{
                    left: `${10 + i * 20}%`,
                    top: `${20 + Math.sin(i) * 30}%`,
                    animation: `organic-float ${8 + i * 2}s ease-in-out infinite ${i * 0.5}s`
                  }}
                />
              ))}
            </div>
            {/* Wind effect lines */}
            {intensity > 0.5 && (
              <div className="absolute inset-0">
                {[...Array(4)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute h-px bg-gradient-to-r from-transparent via-green-400/30 to-transparent"
                    style={{
                      width: '150px',
                      top: `${20 + i * 15}%`,
                      right: '-150px',
                      animation: `wind-sweep ${2 + i * 0.3}s ease-out infinite ${i * 0.2}s`
                    }}
                  />
                ))}
              </div>
            )}
          </>
        );

      case 'philosophy':
        return (
          <>
            {/* Abstract geometric patterns */}
            <div className="absolute inset-0">
              {[...Array(6)].map((_, i) => (
                <div
                  key={i}
                  className={`absolute border border-purple-400/20 rotate-45`}
                  style={{
                    width: `${20 + i * 10}px`,
                    height: `${20 + i * 10}px`,
                    left: `${30 + Math.cos(i) * 40}%`,
                    top: `${40 + Math.sin(i) * 30}%`,
                    animation: `philosophical-spin ${10 + i * 2}s linear infinite`,
                    transformOrigin: 'center'
                  }}
                />
              ))}
            </div>
          </>
        );

      case 'science':
        return (
          <>
            {/* Atomic orbital patterns */}
            <div className="absolute inset-0">
              <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2">
                {[...Array(3)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute border-2 border-indigo-400/20 rounded-full"
                    style={{
                      width: `${100 + i * 50}px`,
                      height: `${100 + i * 50}px`,
                      left: '50%',
                      top: '50%',
                      transform: `translate(-50%, -50%) rotate(${i * 60}deg)`,
                      animation: `orbital-rotation ${15 + i * 5}s linear infinite`
                    }}
                  />
                ))}
                {/* Nucleus */}
                <div className={`absolute w-4 h-4 rounded-full ${categoryConfig.particleColor} left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 animate-pulse`} />
              </div>
            </div>
          </>
        );

      default:
        return null;
    }
  };

  if (academicMode) {
    return (
      <div className="absolute inset-0 pointer-events-none">
        {/* Minimal category indicator */}
        <div className="absolute top-4 right-4">
          <div className="glass-minimal p-2 rounded-lg flex items-center gap-2">
            <categoryConfig.icon className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground capitalize">{topic.category}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      
      {/* Dynamic background ambient glow */}
      <div 
        className="absolute inset-0 transition-all duration-2000"
        style={{
          background: `radial-gradient(ellipse at center, ${categoryConfig.ambientColor} 0%, transparent 70%)`,
          opacity: getIntensityModifier() * 0.5
        }}
      />

      {/* Category-specific environmental elements */}
      {renderCategorySpecificElements()}

      {/* Floating contextual particles */}
      {backgroundParticles.map(particle => (
        <div
          key={particle.id}
          className={`absolute w-1 h-1 ${categoryConfig.particleColor} rounded-full`}
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            animation: 'particle-drift 8s linear forwards',
            opacity: 0.6
          }}
        />
      ))}

      {/* Debate phase indicator */}
      <div className="absolute bottom-4 left-4">
        <div className="glass-minimal px-3 py-1 rounded-lg flex items-center gap-2">
          <div 
            className={`w-2 h-2 rounded-full ${categoryConfig.particleColor} animate-pulse`}
            style={{ animationDuration: `${2 - getIntensityModifier()}s` }}
          />
          <span className="text-xs text-muted-foreground capitalize">
            {debatePhase} | {topic.intensity}
          </span>
        </div>
      </div>

      {/* Topic category badge */}
      <div className="absolute top-4 left-4">
        <div className={`glass-panel px-4 py-2 rounded-lg bg-gradient-to-r ${categoryConfig.secondaryColor}`}>
          <div className="flex items-center gap-2">
            <categoryConfig.icon className={`h-5 w-5 text-current`} />
            <div>
              <div className="text-sm font-bold capitalize">{topic.category}</div>
              <div className="text-xs opacity-80">Environmental Effects Active</div>
            </div>
          </div>
        </div>
      </div>

      {/* Custom animations */}
      <style>{`
        @keyframes grid-pulse {
          0%, 100% { opacity: 0.1; }
          50% { opacity: 0.3; }
        }

        @keyframes circuit-flow {
          0% { 
            transform: translateX(0);
            opacity: 0;
          }
          20% { 
            opacity: 0.8;
          }
          80% { 
            opacity: 0.8;
          }
          100% { 
            transform: translateX(300px);
            opacity: 0;
          }
        }

        @keyframes organic-float {
          0%, 100% { 
            transform: translateY(0) scale(1);
          }
          50% { 
            transform: translateY(-20px) scale(1.1);
          }
        }

        @keyframes wind-sweep {
          0% { 
            transform: translateX(0);
            opacity: 0;
          }
          20% { 
            opacity: 0.6;
          }
          80% { 
            opacity: 0.6;
          }
          100% { 
            transform: translateX(-300px);
            opacity: 0;
          }
        }

        @keyframes philosophical-spin {
          0% { 
            transform: rotate(45deg) scale(1);
            opacity: 0.2;
          }
          50% { 
            transform: rotate(225deg) scale(1.1);
            opacity: 0.4;
          }
          100% { 
            transform: rotate(405deg) scale(1);
            opacity: 0.2;
          }
        }

        @keyframes orbital-rotation {
          0% { 
            transform: translate(-50%, -50%) rotate(0deg);
          }
          100% { 
            transform: translate(-50%, -50%) rotate(360deg);
          }
        }

        @keyframes particle-drift {
          0% { 
            transform: translateY(0) translateX(0) rotate(0deg);
            opacity: 0.6;
          }
          50% { 
            transform: translateY(-100px) translateX(50px) rotate(180deg);
            opacity: 0.8;
          }
          100% { 
            transform: translateY(-200px) translateX(-30px) rotate(360deg);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
};

export const EnvironmentalEffects = memo(EnvironmentalEffectsComponent);