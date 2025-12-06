import { memo, useState, useEffect } from 'react';
import { Trophy, Star, Crown, Shield, Target, Zap, Award, Gem } from 'lucide-react';

interface Achievement {
  id: string;
  title: string;
  description: string;
  category: 'exploration' | 'mastery' | 'legendary' | 'milestone';
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  icon: React.ComponentType<{ className?: string }>;
  unlockedAt: Date;
  progress?: {
    current: number;
    total: number;
  };
}

interface AchievementPopupProps {
  achievement: Achievement;
  onDismiss: () => void;
}

const AchievementPopup = ({ achievement, onDismiss }: AchievementPopupProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const [animationPhase, setAnimationPhase] = useState<'enter' | 'display' | 'exit'>('enter');

  useEffect(() => {
    // Enter animation
    setTimeout(() => setIsVisible(true), 100);
    setTimeout(() => setAnimationPhase('display'), 600);
    
    // Auto dismiss after display time
    const dismissTimer = setTimeout(() => {
      setAnimationPhase('exit');
      setTimeout(onDismiss, 500);
    }, 4000);

    return () => clearTimeout(dismissTimer);
  }, [onDismiss]);

  const getRarityStyles = () => {
    switch (achievement.rarity) {
      case 'legendary':
        return {
          border: 'border-yellow-400/50',
          bg: 'bg-gradient-to-br from-yellow-500/10 via-amber-500/5 to-orange-500/10',
          glow: 'shadow-yellow-400/20',
          particle: 'bg-yellow-400'
        };
      case 'epic':
        return {
          border: 'border-purple-400/50',
          bg: 'bg-gradient-to-br from-purple-500/10 via-violet-500/5 to-indigo-500/10',
          glow: 'shadow-purple-400/20',
          particle: 'bg-purple-400'
        };
      case 'rare':
        return {
          border: 'border-blue-400/50',
          bg: 'bg-gradient-to-br from-blue-500/10 via-cyan-500/5 to-sky-500/10',
          glow: 'shadow-blue-400/20',
          particle: 'bg-blue-400'
        };
      default:
        return {
          border: 'border-primary/30',
          bg: 'bg-gradient-to-br from-primary/10 to-secondary/5',
          glow: 'shadow-primary/20',
          particle: 'bg-primary'
        };
    }
  };

  const styles = getRarityStyles();
  const IconComponent = achievement.icon;

  return (
    <div
      className={`
        fixed top-20 right-4 z-50 transform transition-all duration-500 ease-out
        ${isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
        ${animationPhase === 'enter' ? 'scale-95' : 'scale-100'}
        ${animationPhase === 'exit' ? 'translate-x-full opacity-0' : ''}
      `}
    >
      <div className={`
        relative glass-panel border-2 ${styles.border} ${styles.bg} 
        shadow-2xl ${styles.glow} p-4 rounded-xl max-w-sm
        transform transition-transform duration-300 hover:scale-105
      `}>
        
        {/* Particle Effects */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-xl">
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              className={`
                absolute w-1 h-1 ${styles.particle} rounded-full
                animate-bounce opacity-60
              `}
              style={{
                left: `${20 + i * 10}%`,
                top: `${15 + (i % 3) * 20}%`,
                animationDelay: `${i * 0.1}s`,
                animationDuration: `${1.5 + Math.random()}s`
              }}
            />
          ))}
        </div>

        {/* Achievement Content */}
        <div className="relative z-10">
          <div className="flex items-start gap-3">
            
            {/* Achievement Icon */}
            <div className={`
              p-3 rounded-xl ${styles.bg} border ${styles.border}
              flex items-center justify-center shadow-lg
              transform transition-transform duration-300 hover:rotate-12
            `}>
              <IconComponent className="h-6 w-6 text-primary" />
            </div>

            {/* Achievement Details */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="heading-refined font-semibold text-sm text-primary">
                  Achievement Unlocked!
                </h3>
                <div className={`
                  px-2 py-0.5 rounded-full text-xs font-medium uppercase tracking-wide
                  ${achievement.rarity === 'legendary' ? 'bg-yellow-500/20 text-yellow-300' :
                    achievement.rarity === 'epic' ? 'bg-purple-500/20 text-purple-300' :
                    achievement.rarity === 'rare' ? 'bg-blue-500/20 text-blue-300' :
                    'bg-primary/20 text-primary'}
                `}>
                  {achievement.rarity}
                </div>
              </div>
              
              <h4 className="heading-refined font-bold mb-1">{achievement.title}</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {achievement.description}
              </p>

              {/* Progress Indicator (if applicable) */}
              {achievement.progress && (
                <div className="mt-2">
                  <div className="flex justify-between text-xs text-muted-foreground mb-1">
                    <span>Progress</span>
                    <span>{achievement.progress.current}/{achievement.progress.total}</span>
                  </div>
                  <div className="glass-minimal h-1.5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${styles.particle} transition-all duration-1000 ease-out`}
                      style={{ width: `${(achievement.progress.current / achievement.progress.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Dismiss Button */}
            <button
              onClick={onDismiss}
              className="glass-minimal p-1 rounded-lg hover:bg-muted/20 transition-colors opacity-50 hover:opacity-100"
              aria-label="Dismiss achievement"
            >
              <div className="w-4 h-4 flex items-center justify-center">
                <div className="w-2 h-0.5 bg-current rotate-45 absolute" />
                <div className="w-2 h-0.5 bg-current -rotate-45 absolute" />
              </div>
            </button>
          </div>
        </div>

        {/* Special Legendary Effect */}
        {achievement.rarity === 'legendary' && (
          <div className="absolute inset-0 rounded-xl pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-r from-yellow-400/20 via-transparent to-yellow-400/20 animate-pulse" />
            <div className="absolute top-2 right-2">
              <Crown className="h-4 w-4 text-yellow-400 animate-bounce" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const AchievementSystemComponent = () => {
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [currentPopup, setCurrentPopup] = useState<Achievement | null>(null);

  // Predefined achievements that can be unlocked
  const achievementDatabase: Omit<Achievement, 'unlockedAt'>[] = [
    {
      id: 'first-visit',
      title: 'Strategic Reconnaissance',
      description: 'Welcome to the Strategic Command Center! Your evaluation journey begins.',
      category: 'exploration',
      rarity: 'common',
      icon: Shield
    },
    {
      id: 'demo-complete',
      title: 'Feature Explorer',
      description: 'Completed the interactive showcase demonstration. You know the terrain!',
      category: 'exploration',
      rarity: 'common',
      icon: Target,
      progress: { current: 11, total: 11 }
    },
    {
      id: 'tour-master',
      title: 'Tactical Awareness',
      description: 'Completed the guided feature tour. Strategic knowledge acquired!',
      category: 'mastery',
      rarity: 'rare',
      icon: Star
    },
    {
      id: 'theme-customizer',
      title: 'Command Personalization',
      description: 'Customized your command center theme. The environment adapts to you!',
      category: 'mastery',
      rarity: 'rare',
      icon: Zap
    },
    {
      id: 'portfolio-viewer',
      title: 'Strategic Intelligence',
      description: 'Accessed portfolio presentation mode. You understand the mission!',
      category: 'milestone',
      rarity: 'epic',
      icon: Award
    },
    {
      id: 'keyboard-master',
      title: 'Hotkey Commander',
      description: 'Used keyboard shortcuts extensively. Efficiency is your weapon!',
      category: 'mastery',
      rarity: 'epic',
      icon: Zap,
      progress: { current: 5, total: 10 }
    },
    {
      id: 'legendary-explorer',
      title: 'Command Center Mastery',
      description: 'Explored every feature of the Strategic Command Center. True mastery achieved!',
      category: 'legendary',
      rarity: 'legendary',
      icon: Crown
    }
  ];

  // Simulate achievement unlocking based on user interactions
  useEffect(() => {
    const unlockTimers: NodeJS.Timeout[] = [];

    // Welcome achievement after 2 seconds
    unlockTimers.push(setTimeout(() => {
      unlockAchievement('first-visit');
    }, 2000));

    // Simulate various achievements being unlocked over time
    unlockTimers.push(setTimeout(() => {
      unlockAchievement('demo-complete');
    }, 30000)); // 30 seconds

    unlockTimers.push(setTimeout(() => {
      unlockAchievement('tour-master');
    }, 60000)); // 1 minute

    unlockTimers.push(setTimeout(() => {
      unlockAchievement('theme-customizer');
    }, 90000)); // 1.5 minutes

    unlockTimers.push(setTimeout(() => {
      unlockAchievement('portfolio-viewer');
    }, 120000)); // 2 minutes

    unlockTimers.push(setTimeout(() => {
      unlockAchievement('keyboard-master');
    }, 150000)); // 2.5 minutes

    unlockTimers.push(setTimeout(() => {
      unlockAchievement('legendary-explorer');
    }, 180000)); // 3 minutes

    return () => {
      unlockTimers.forEach(timer => clearTimeout(timer));
    };
  }, []);

  const unlockAchievement = (achievementId: string) => {
    const template = achievementDatabase.find(a => a.id === achievementId);
    if (!template) return;

    // Check if already unlocked
    if (achievements.some(a => a.id === achievementId)) return;

    const newAchievement: Achievement = {
      ...template,
      unlockedAt: new Date()
    };

    setAchievements(prev => [...prev, newAchievement]);
    setCurrentPopup(newAchievement);
  };

  const dismissPopup = () => {
    setCurrentPopup(null);
  };

  return (
    <>
      {/* Achievement Popup */}
      {currentPopup && (
        <AchievementPopup 
          achievement={currentPopup} 
          onDismiss={dismissPopup}
        />
      )}

      {/* Achievement Counter (bottom right, subtle) */}
      {achievements.length > 0 && (
        <div className="fixed bottom-20 right-4 glass-minimal px-3 py-2 rounded-full text-xs text-muted-foreground z-40">
          <div className="flex items-center gap-2">
            <Trophy className="h-3 w-3" />
            <span>{achievements.length} achievements</span>
          </div>
        </div>
      )}
    </>
  );
};

export const AchievementSystem = memo(AchievementSystemComponent);