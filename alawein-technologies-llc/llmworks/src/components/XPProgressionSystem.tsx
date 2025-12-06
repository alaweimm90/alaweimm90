import { memo, useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Trophy, 
  Star, 
  Zap, 
  Shield, 
  Target, 
  Crown, 
  Award, 
  Sparkles,
  TrendingUp,
  Brain,
  Cpu
} from 'lucide-react';

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  unlocked: boolean;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  xpReward: number;
}

interface PlayerLevel {
  level: number;
  title: string;
  xpCurrent: number;
  xpRequired: number;
  badge: 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond';
  perks: string[];
}

const XPProgressionSystemComponent = () => {
  const [playerLevel, setPlayerLevel] = useState<PlayerLevel>({
    level: 12,
    title: 'Strategic Commander',
    xpCurrent: 2847,
    xpRequired: 3200,
    badge: 'gold',
    perks: ['Extended Battle Duration', 'Advanced Analytics', 'Priority Queue Access']
  });

  const [recentXpGain, setRecentXpGain] = useState<number>(0);
  const [showXpAnimation, setShowXpAnimation] = useState(false);

  const achievements: Achievement[] = [
    {
      id: 'first-victory',
      name: 'First Victory',
      description: 'Win your first strategic evaluation',
      icon: Trophy,
      unlocked: true,
      rarity: 'common',
      xpReward: 100,
    },
    {
      id: 'model-expert',
      name: 'Model Expert',
      description: 'Test 25 different AI models',
      icon: Brain,
      unlocked: true,
      rarity: 'rare',
      xpReward: 250,
    },
    {
      id: 'strategic-mastermind',
      name: 'Strategic Mastermind',
      description: 'Complete 100 strategic evaluations',
      icon: Crown,
      unlocked: false,
      rarity: 'epic',
      xpReward: 500,
    },
    {
      id: 'legendary-commander',
      name: 'Legendary Commander',
      description: 'Reach top 10 global leaderboard',
      icon: Star,
      unlocked: false,
      rarity: 'legendary',
      xpReward: 1000,
    },
  ];

  const getRarityColor = (rarity: string) => {
    switch (rarity) {
      case 'common': return 'text-gray-400 border-gray-400/30 bg-gray-400/10';
      case 'rare': return 'text-blue-400 border-blue-400/30 bg-blue-400/10';
      case 'epic': return 'text-purple-400 border-purple-400/30 bg-purple-400/10';
      case 'legendary': return 'text-yellow-400 border-yellow-400/30 bg-yellow-400/10';
      default: return 'text-gray-400 border-gray-400/30 bg-gray-400/10';
    }
  };

  const getBadgeColor = (badge: string) => {
    switch (badge) {
      case 'bronze': return 'rank-bronze';
      case 'silver': return 'rank-silver';
      case 'gold': return 'rank-gold';
      case 'platinum': return 'rank-platinum';
      case 'diamond': return 'text-cyan-400 bg-cyan-400/20 border-cyan-400/30';
      default: return 'performance-standard';
    }
  };

  // Simulate XP gain animation
  useEffect(() => {
    const xpGainInterval = setInterval(() => {
      if (Math.random() > 0.7) {
        const gain = Math.floor(Math.random() * 50) + 10;
        setRecentXpGain(gain);
        setShowXpAnimation(true);
        
        setPlayerLevel(prev => ({
          ...prev,
          xpCurrent: Math.min(prev.xpCurrent + gain, prev.xpRequired)
        }));

        setTimeout(() => setShowXpAnimation(false), 2000);
      }
    }, 8000);

    return () => clearInterval(xpGainInterval);
  }, []);

  const progressPercentage = (playerLevel.xpCurrent / playerLevel.xpRequired) * 100;

  return (
    <div className="space-y-6">
      {/* Player Level Card */}
      <Card className="glass-panel border-border/20 overflow-hidden">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="glass-subtle p-2 rounded-xl">
                <Crown className="h-5 w-5 text-primary" />
              </div>
              <div>
                <CardTitle className="heading-refined text-lg">Commander Profile</CardTitle>
                <p className="text-xs text-muted-foreground">Strategic Operations Division</p>
              </div>
            </div>
            <Badge className={`strategic-rank ${getBadgeColor(playerLevel.badge)}`}>
              LVL {playerLevel.level}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Level Title and XP */}
          <div className="flex items-center justify-between">
            <div>
              <h3 className="heading-refined text-sm text-primary">{playerLevel.title}</h3>
              <p className="text-xs text-muted-foreground">
                {playerLevel.xpCurrent.toLocaleString()} / {playerLevel.xpRequired.toLocaleString()} XP
              </p>
            </div>
            {showXpAnimation && (
              <div className="animate-bounce">
                <Badge className="performance-elite strategic-rank">
                  +{recentXpGain} XP
                </Badge>
              </div>
            )}
          </div>

          {/* XP Progress Bar */}
          <div className="space-y-2">
            <Progress 
              value={progressPercentage} 
              className="h-3 bg-muted/30"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Progress to Level {playerLevel.level + 1}</span>
              <span>{Math.round(progressPercentage)}%</span>
            </div>
          </div>

          {/* Perks */}
          <div>
            <h4 className="heading-refined text-xs mb-2 flex items-center gap-2">
              <Sparkles className="h-3 w-3 text-accent" />
              Commander Perks
            </h4>
            <div className="flex flex-wrap gap-1">
              {playerLevel.perks.map((perk, index) => (
                <Badge key={index} variant="outline" className="text-xs glass-minimal">
                  {perk}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Achievements Grid */}
      <Card className="glass-panel border-border/20">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="glass-subtle p-2 rounded-xl">
              <Award className="h-5 w-5 text-secondary" />
            </div>
            <div>
              <CardTitle className="heading-refined text-lg">Strategic Achievements</CardTitle>
              <p className="text-xs text-muted-foreground">
                {achievements.filter(a => a.unlocked).length} of {achievements.length} unlocked
              </p>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {achievements.map((achievement) => {
              const IconComponent = achievement.icon;
              return (
                <div
                  key={achievement.id}
                  className={`glass-subtle p-4 rounded-lg transition-all duration-300 border ${
                    achievement.unlocked 
                      ? `${getRarityColor(achievement.rarity)} hover:shadow-md` 
                      : 'opacity-50 border-muted/30'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      achievement.unlocked 
                        ? getRarityColor(achievement.rarity)
                        : 'bg-muted/20'
                    }`}>
                      <IconComponent className="h-4 w-4" />
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="heading-refined text-sm">{achievement.name}</h4>
                        {achievement.unlocked && (
                          <Badge className="text-xs performance-superior strategic-rank">
                            +{achievement.xpReward}
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mb-2">
                        {achievement.description}
                      </p>
                      <Badge 
                        variant="outline" 
                        className={`text-xs capitalize ${getRarityColor(achievement.rarity)}`}
                      >
                        {achievement.rarity}
                      </Badge>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="glass-subtle border-border/20 text-center">
          <CardContent className="pt-4 pb-4">
            <div className="flex flex-col items-center gap-2">
              <Target className="h-5 w-5 text-primary" />
              <div className="heading-display text-lg">247</div>
              <div className="text-xs text-muted-foreground">Battles Won</div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-subtle border-border/20 text-center">
          <CardContent className="pt-4 pb-4">
            <div className="flex flex-col items-center gap-2">
              <TrendingUp className="h-5 w-5 text-secondary" />
              <div className="heading-display text-lg">89%</div>
              <div className="text-xs text-muted-foreground">Win Rate</div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-subtle border-border/20 text-center">
          <CardContent className="pt-4 pb-4">
            <div className="flex flex-col items-center gap-2">
              <Cpu className="h-5 w-5 text-accent" />
              <div className="heading-display text-lg">47</div>
              <div className="text-xs text-muted-foreground">Models Tested</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export const XPProgressionSystem = memo(XPProgressionSystemComponent);