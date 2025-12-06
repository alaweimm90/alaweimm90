import { memo, useEffect, useState } from 'react';
import { Trophy, Star, Crown, Sparkles, Zap } from 'lucide-react';

interface Particle {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  size: number;
  type: 'star' | 'spark' | 'circle';
}

interface VictoryCelebrationProps {
  show: boolean;
  title: string;
  subtitle?: string;
  onComplete?: () => void;
}

const VictoryCelebrationComponent = ({ 
  show, 
  title, 
  subtitle = "Strategic Victory Achieved!",
  onComplete 
}: VictoryCelebrationProps) => {
  const [particles, setParticles] = useState<Particle[]>([]);
  const [showContent, setShowContent] = useState(false);

  // Generate particles
  useEffect(() => {
    if (!show) {
      setParticles([]);
      setShowContent(false);
      return;
    }

    setShowContent(true);

    const colors = [
      'hsl(var(--victory-gold))', // Victory Gold
      'hsl(var(--accent))', // Strategic Accent
      'hsl(var(--danger))', // Alert Red
      'hsl(var(--intel-cyan))', // Intel Cyan
      'hsl(var(--primary))', // Tactical Blue
      'hsl(var(--success))', // Success Green
      'hsl(var(--warning))', // Warning Yellow
      'hsl(var(--secondary))', // Secondary
    ];

    const particleTypes: Array<'star' | 'spark' | 'circle'> = ['star', 'spark', 'circle'];

    // Create initial burst of particles
    const initialParticles: Particle[] = [];
    for (let i = 0; i < 50; i++) {
      const angle = (Math.PI * 2 * i) / 50;
      const velocity = 3 + Math.random() * 4;
      const life = 60 + Math.random() * 40;
      
      initialParticles.push({
        id: `particle-${i}`,
        x: 50, // Center of screen
        y: 50,
        vx: Math.cos(angle) * velocity,
        vy: Math.sin(angle) * velocity,
        life: life,
        maxLife: life,
        color: colors[Math.floor(Math.random() * colors.length)],
        size: 2 + Math.random() * 4,
        type: particleTypes[Math.floor(Math.random() * particleTypes.length)]
      });
    }

    setParticles(initialParticles);

    // Animation loop
    const animationInterval = setInterval(() => {
      setParticles(prevParticles => {
        const updatedParticles = prevParticles
          .map(particle => ({
            ...particle,
            x: particle.x + particle.vx,
            y: particle.y + particle.vy,
            vy: particle.vy + 0.1, // Gravity
            vx: particle.vx * 0.99, // Air resistance
            life: particle.life - 1,
          }))
          .filter(particle => particle.life > 0 && particle.y < 100 && particle.x > -10 && particle.x < 110);

        return updatedParticles;
      });
    }, 16); // ~60fps

    // Add continuous sparkles
    const sparkleInterval = setInterval(() => {
      if (!show) return;
      
      setParticles(prev => {
        const newSparkles: Particle[] = [];
        for (let i = 0; i < 3; i++) {
          newSparkles.push({
            id: `sparkle-${Date.now()}-${i}`,
            x: 30 + Math.random() * 40,
            y: 30 + Math.random() * 40,
            vx: (Math.random() - 0.5) * 2,
            vy: -Math.random() * 2 - 1,
            life: 30 + Math.random() * 20,
            maxLife: 50,
            color: colors[Math.floor(Math.random() * colors.length)],
            size: 1 + Math.random() * 2,
            type: 'star'
          });
        }
        return [...prev, ...newSparkles].slice(-100); // Limit total particles
      });
    }, 200);

    // Auto-complete after 4 seconds
    const completeTimeout = setTimeout(() => {
      if (onComplete) onComplete();
    }, 4000);

    return () => {
      clearInterval(animationInterval);
      clearInterval(sparkleInterval);
      clearTimeout(completeTimeout);
    };
  }, [show, onComplete]);

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/20 backdrop-blur-sm"></div>
      
      {/* Particles Container */}
      <div className="absolute inset-0 overflow-hidden">
        {particles.map((particle) => {
          const opacity = particle.life / particle.maxLife;
          const scale = Math.min(1, particle.life / (particle.maxLife * 0.3));
          
          return (
            <div
              key={particle.id}
              className="absolute pointer-events-none"
              style={{
                left: `${particle.x}%`,
                top: `${particle.y}%`,
                transform: `translate(-50%, -50%) scale(${scale})`,
                opacity,
              }}
            >
              {particle.type === 'star' && (
                <Star 
                  style={{ 
                    color: particle.color, 
                    width: particle.size * 4, 
                    height: particle.size * 4 
                  }} 
                  fill="currentColor"
                />
              )}
              {particle.type === 'spark' && (
                <Sparkles 
                  style={{ 
                    color: particle.color, 
                    width: particle.size * 3, 
                    height: particle.size * 3 
                  }} 
                />
              )}
              {particle.type === 'circle' && (
                <div
                  style={{
                    width: particle.size * 2,
                    height: particle.size * 2,
                    backgroundColor: particle.color,
                    borderRadius: '50%',
                  }}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Victory Content */}
      {showContent && (
        <div className="relative z-10 text-center animate-bounce">
          <div className="glass-panel p-8 rounded-2xl border-2 border-primary/50 max-w-md mx-4">
            {/* Victory Icon */}
            <div className="mb-6 relative">
              <div className="absolute inset-0 animate-ping">
                <Crown className="h-16 w-16 mx-auto text-yellow-400 opacity-75" />
              </div>
              <Crown className="h-16 w-16 mx-auto text-yellow-400 relative z-10" />
            </div>

            {/* Victory Text */}
            <div className="space-y-4">
              <h2 className="heading-display text-3xl text-primary font-bold">
                {title}
              </h2>
              <p className="body-elegant text-lg text-muted-foreground">
                {subtitle}
              </p>
            </div>

            {/* Animated Trophy */}
            <div className="mt-6 flex justify-center items-center gap-3">
              <Trophy className="h-8 w-8 text-yellow-400 animate-pulse" />
              <span className="heading-refined text-lg text-accent">MISSION COMPLETE</span>
              <Trophy className="h-8 w-8 text-yellow-400 animate-pulse" />
            </div>

            {/* Energy Bars */}
            <div className="mt-6 flex justify-center gap-1">
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  className="w-2 h-8 bg-primary/30 rounded-full overflow-hidden"
                >
                  <div 
                    className="w-full bg-primary rounded-full animate-pulse"
                    style={{ 
                      height: `${100 - i * 10}%`,
                      animationDelay: `${i * 100}ms`
                    }}
                  ></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Lightning Effects */}
      <div className="absolute inset-0 pointer-events-none">
        <Zap className="absolute top-1/4 left-1/4 h-12 w-12 text-yellow-400 animate-ping opacity-60" />
        <Zap className="absolute top-3/4 right-1/4 h-8 w-8 text-cyan-400 animate-pulse opacity-60" />
        <Sparkles className="absolute top-1/2 left-1/6 h-10 w-10 text-purple-400 animate-bounce opacity-60" />
        <Sparkles className="absolute bottom-1/4 right-1/6 h-6 w-6 text-green-400 animate-spin opacity-60" />
      </div>
    </div>
  );
};

export const VictoryCelebration = memo(VictoryCelebrationComponent);