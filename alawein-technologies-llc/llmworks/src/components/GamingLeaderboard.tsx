import { memo, useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Crown, 
  Trophy, 
  Star, 
  Sword, 
  Shield, 
  Zap, 
  Target, 
  TrendingUp, 
  ChevronDown,
  Medal,
  Award,
  Sparkles,
  Flame,
  Bot
} from 'lucide-react';
import { VictoryCelebration } from './VictoryCelebration';

interface LeaderboardEntry {
  id: string;
  name: string;
  model: string;
  elo: number;
  tier: 'legendary' | 'diamond' | 'platinum' | 'gold' | 'silver' | 'bronze';
  wins: number;
  losses: number;
  streak: number;
  avatar: string;
  isHuman?: boolean;
  specialTitle?: string;
  battleStats: {
    accuracy: number;
    creativity: number;
    strategy: number;
    speed: number;
  };
}

interface TierInfo {
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  minElo: number;
  maxElo: number;
  gradient: string;
}

const GamingLeaderboardComponent = () => {
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [showCelebration, setShowCelebration] = useState(false);
  const [animatingEntry, setAnimatingEntry] = useState<string | null>(null);

  const tiers: Record<string, TierInfo> = {
    legendary: {
      name: 'Legendary Commander',
      icon: Crown,
      color: 'text-yellow-400',
      minElo: 2400,
      maxElo: 3000,
      gradient: 'from-yellow-400 to-orange-500'
    },
    diamond: {
      name: 'Diamond Strategist', 
      icon: Star,
      color: 'text-cyan-400',
      minElo: 2000,
      maxElo: 2399,
      gradient: 'from-cyan-400 to-blue-500'
    },
    platinum: {
      name: 'Platinum Tactician',
      icon: Award,
      color: 'text-gray-300',
      minElo: 1600,
      maxElo: 1999,
      gradient: 'from-gray-300 to-gray-400'
    },
    gold: {
      name: 'Gold Elite',
      icon: Medal,
      color: 'text-yellow-500',
      minElo: 1200,
      maxElo: 1599,
      gradient: 'from-yellow-500 to-yellow-600'
    },
    silver: {
      name: 'Silver Operative',
      icon: Shield,
      color: 'text-gray-400',
      minElo: 800,
      maxElo: 1199,
      gradient: 'from-gray-400 to-gray-500'
    },
    bronze: {
      name: 'Bronze Recruit',
      icon: Target,
      color: 'text-orange-600',
      minElo: 0,
      maxElo: 799,
      gradient: 'from-orange-600 to-orange-700'
    }
  };

  const leaderboardData: LeaderboardEntry[] = [
    {
      id: '1',
      name: 'QuantumMind_Pro',
      model: 'GPT-4 Turbo',
      elo: 2847,
      tier: 'legendary',
      wins: 247,
      losses: 23,
      streak: 15,
      avatar: '🤖',
      specialTitle: 'Strategic Overlord',
      battleStats: { accuracy: 95, creativity: 92, strategy: 98, speed: 87 }
    },
    {
      id: '2',
      name: 'Dr_Nexus',
      model: 'Claude-3 Opus',
      elo: 2756,
      tier: 'legendary',
      wins: 198,
      losses: 31,
      streak: 8,
      avatar: '🧠',
      isHuman: true,
      specialTitle: 'AI Whisperer',
      battleStats: { accuracy: 91, creativity: 96, strategy: 89, speed: 84 }
    },
    {
      id: '3',
      name: 'TacticalGenius',
      model: 'Gemini Ultra',
      elo: 2634,
      tier: 'legendary',
      wins: 156,
      losses: 28,
      streak: 12,
      avatar: '⚡',
      specialTitle: 'Combat Virtuoso',
      battleStats: { accuracy: 89, creativity: 87, strategy: 94, speed: 91 }
    },
    {
      id: '4',
      name: 'AICommander_X',
      model: 'GPT-3.5 Turbo',
      elo: 2298,
      tier: 'diamond',
      wins: 134,
      losses: 45,
      streak: 5,
      avatar: '💎',
      battleStats: { accuracy: 86, creativity: 82, strategy: 88, speed: 89 }
    },
    {
      id: '5',
      name: 'StrategicMaster',
      model: 'PaLM-2',
      elo: 2156,
      tier: 'diamond',
      wins: 112,
      losses: 39,
      streak: 3,
      avatar: '🎯',
      isHuman: true,
      battleStats: { accuracy: 84, creativity: 88, strategy: 85, speed: 82 }
    },
    {
      id: '6',
      name: 'CyberTactician',
      model: 'LLaMA-2 70B',
      elo: 1987,
      tier: 'platinum',
      wins: 89,
      losses: 34,
      streak: 7,
      avatar: '🛡️',
      battleStats: { accuracy: 81, creativity: 79, strategy: 83, speed: 86 }
    }
  ];

  const getRankIcon = (position: number) => {
    switch (position) {
      case 1: return <Crown className="h-5 w-5 text-yellow-400" />;
      case 2: return <Trophy className="h-5 w-5 text-gray-300" />;
      case 3: return <Medal className="h-5 w-5 text-orange-600" />;
      default: return <span className="text-sm font-bold text-muted-foreground">#{position}</span>;
    }
  };

  const getTierIcon = (tier: string) => {
    const TierIcon = tiers[tier].icon;
    return <TierIcon className={`h-4 w-4 ${tiers[tier].color}`} />;
  };

  const filteredData = selectedTier === 'all' 
    ? leaderboardData 
    : leaderboardData.filter(entry => entry.tier === selectedTier);

  // Simulate rank changes
  useEffect(() => {
    const rankChangeInterval = setInterval(() => {
      if (Math.random() > 0.8) {
        const randomEntry = leaderboardData[Math.floor(Math.random() * leaderboardData.length)];
        setAnimatingEntry(randomEntry.id);
        
        if (Math.random() > 0.7) {
          setShowCelebration(true);
        }

        setTimeout(() => setAnimatingEntry(null), 2000);
      }
    }, 8000);

    return () => clearInterval(rankChangeInterval);
  }, []);

  return (
    <>
      <div className="space-y-6">
        {/* Tier Selector */}
        <Card className="glass-panel border-border/20">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="glass-subtle p-2 rounded-xl">
                  <Sword className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle className="heading-refined text-lg">Strategic Command Rankings</CardTitle>
                  <p className="text-xs text-muted-foreground">Elite commanders by tactical superiority</p>
                </div>
              </div>
              <Badge className="performance-elite strategic-rank">
                LIVE RANKINGS
              </Badge>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* Tier Filter */}
            <div className="flex flex-wrap gap-2">
              <Button
                variant={selectedTier === 'all' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedTier('all')}
                className="glass-minimal"
              >
                All Tiers
              </Button>
              {Object.entries(tiers).map(([tierKey, tierInfo]) => {
                const TierIcon = tierInfo.icon;
                return (
                  <Button
                    key={tierKey}
                    variant={selectedTier === tierKey ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedTier(tierKey)}
                    className="glass-minimal"
                  >
                    <TierIcon className={`h-3 w-3 mr-1 ${tierInfo.color}`} />
                    {tierInfo.name.split(' ')[0]}
                  </Button>
                );
              })}
            </div>

            {/* Leaderboard */}
            <div className="space-y-3">
              {filteredData.map((entry, index) => {
                const tierInfo = tiers[entry.tier];
                const winRate = Math.round((entry.wins / (entry.wins + entry.losses)) * 100);
                const isAnimating = animatingEntry === entry.id;

                return (
                  <Card 
                    key={entry.id} 
                    className={`glass-subtle border-border/20 transition-all duration-500 ${
                      isAnimating ? 'scale-105 shadow-lg border-primary/50' : 'hover:shadow-md'
                    }`}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center gap-4">
                        {/* Rank & Avatar */}
                        <div className="flex items-center gap-3">
                          <div className="flex items-center justify-center w-8 h-8">
                            {getRankIcon(index + 1)}
                          </div>
                          <div className="text-2xl">{entry.avatar}</div>
                        </div>

                        {/* Player Info */}
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="heading-refined text-sm">{entry.name}</h3>
                            {entry.isHuman && (
                              <Badge variant="outline" className="text-xs glass-minimal border-green-400/30 text-green-400">
                                HUMAN
                              </Badge>
                            )}
                            {entry.specialTitle && (
                              <Badge className="text-xs performance-superior strategic-rank">
                                {entry.specialTitle}
                              </Badge>
                            )}
                          </div>
                          
                          <div className="flex items-center gap-3 text-xs text-muted-foreground">
                            <span>{entry.model}</span>
                            <div className="flex items-center gap-1">
                              {getTierIcon(entry.tier)}
                              <span>{tierInfo.name}</span>
                            </div>
                          </div>

                          {/* Battle Stats Mini-Bars */}
                          <div className="grid grid-cols-4 gap-1 mt-2">
                            <div className="space-y-1">
                              <div className="text-xs text-muted-foreground">ACC</div>
                              <div className="w-full bg-muted/20 rounded-full h-1">
                                <div 
                                  className="h-1 bg-blue-400 rounded-full transition-all"
                                  style={{ width: `${entry.battleStats.accuracy}%` }}
                                />
                              </div>
                            </div>
                            <div className="space-y-1">
                              <div className="text-xs text-muted-foreground">CRE</div>
                              <div className="w-full bg-muted/20 rounded-full h-1">
                                <div 
                                  className="h-1 bg-purple-400 rounded-full transition-all"
                                  style={{ width: `${entry.battleStats.creativity}%` }}
                                />
                              </div>
                            </div>
                            <div className="space-y-1">
                              <div className="text-xs text-muted-foreground">STR</div>
                              <div className="w-full bg-muted/20 rounded-full h-1">
                                <div 
                                  className="h-1 bg-red-400 rounded-full transition-all"
                                  style={{ width: `${entry.battleStats.strategy}%` }}
                                />
                              </div>
                            </div>
                            <div className="space-y-1">
                              <div className="text-xs text-muted-foreground">SPD</div>
                              <div className="w-full bg-muted/20 rounded-full h-1">
                                <div 
                                  className="h-1 bg-green-400 rounded-full transition-all"
                                  style={{ width: `${entry.battleStats.speed}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Stats */}
                        <div className="text-right space-y-1">
                          <div className="heading-display text-lg text-primary">{entry.elo}</div>
                          <div className="text-xs text-muted-foreground">ELO Rating</div>
                          
                          <div className="flex items-center gap-2 text-xs">
                            <span className="text-green-400">{entry.wins}W</span>
                            <span className="text-red-400">{entry.losses}L</span>
                            <Badge variant="outline" className="text-xs">
                              {winRate}%
                            </Badge>
                          </div>

                          {entry.streak > 0 && (
                            <div className="flex items-center gap-1 text-xs">
                              <Flame className="h-3 w-3 text-orange-400" />
                              <span className="text-orange-400">{entry.streak} streak</span>
                            </div>
                          )}

                          {isAnimating && (
                            <div className="animate-bounce">
                              <Badge className="text-xs performance-elite strategic-rank">
                                RANK UP!
                              </Badge>
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Tier Distribution Chart */}
        <Card className="glass-panel border-border/20">
          <CardHeader className="pb-4">
            <CardTitle className="heading-refined text-lg flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-secondary" />
              Tier Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(tiers).reverse().map(([tierKey, tierInfo]) => {
                const TierIcon = tierInfo.icon;
                const count = leaderboardData.filter(entry => entry.tier === tierKey).length;
                const percentage = (count / leaderboardData.length) * 100;

                return (
                  <div key={tierKey} className="flex items-center gap-3">
                    <TierIcon className={`h-4 w-4 ${tierInfo.color}`} />
                    <span className="text-sm min-w-[120px]">{tierInfo.name}</span>
                    <div className="flex-1">
                      <Progress value={percentage} className="h-2" />
                    </div>
                    <span className="text-xs text-muted-foreground min-w-[60px] text-right">
                      {count} ({Math.round(percentage)}%)
                    </span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Victory Celebration */}
      <VictoryCelebration
        show={showCelebration}
        title="Legendary Achievement!"
        subtitle="New commander has joined the elite ranks!"
        onComplete={() => setShowCelebration(false)}
      />
    </>
  );
};

export const GamingLeaderboard = memo(GamingLeaderboardComponent);