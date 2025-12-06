import { memo, useState, useEffect } from 'react';
import { Trophy, TrendingUp, TrendingDown, Minus, Crown, Star, Shield, Zap } from 'lucide-react';

interface Contender {
  id: string;
  name: string;
  category: 'reasoning' | 'creative' | 'conversational' | 'analytical';
  currentRank: number;
  previousRank: number;
  winStreak: number;
  specialization: string[];
  lastActive: Date;
  momentum: 'rising' | 'falling' | 'stable';
}

const DynamicLeaderboardComponent = () => {
  const [contenders, setContenders] = useState<Contender[]>([]);
  const [displayMode, setDisplayMode] = useState<'overall' | 'category' | 'recent'>('overall');

  // Simulate dynamic ranking changes
  useEffect(() => {
    const initialContenders: Contender[] = [
      {
        id: 'strategic-analyst',
        name: 'Strategic Analyst Alpha',
        category: 'analytical',
        currentRank: 1,
        previousRank: 2,
        winStreak: 7,
        specialization: ['Logic', 'Data Analysis', 'Pattern Recognition'],
        lastActive: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
        momentum: 'rising'
      },
      {
        id: 'creative-nexus',
        name: 'Creative Nexus Pro',
        category: 'creative',
        currentRank: 2,
        previousRank: 1,
        winStreak: 5,
        specialization: ['Storytelling', 'Innovation', 'Artistic Vision'],
        lastActive: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
        momentum: 'falling'
      },
      {
        id: 'reasoning-engine',
        name: 'Reasoning Engine X1',
        category: 'reasoning',
        currentRank: 3,
        previousRank: 4,
        winStreak: 12,
        specialization: ['Problem Solving', 'Logic Chains', 'Deduction'],
        lastActive: new Date(Date.now() - 1000 * 60 * 5), // 5 minutes ago
        momentum: 'rising'
      },
      {
        id: 'conversation-master',
        name: 'Conversation Master',
        category: 'conversational',
        currentRank: 4,
        previousRank: 3,
        winStreak: 3,
        specialization: ['Dialogue', 'Context Retention', 'Empathy'],
        lastActive: new Date(Date.now() - 1000 * 60 * 45), // 45 minutes ago
        momentum: 'stable'
      },
      {
        id: 'tactical-commander',
        name: 'Tactical Commander',
        category: 'analytical',
        currentRank: 5,
        previousRank: 6,
        winStreak: 9,
        specialization: ['Strategy', 'Risk Assessment', 'Optimization'],
        lastActive: new Date(Date.now() - 1000 * 60 * 20), // 20 minutes ago
        momentum: 'rising'
      },
      {
        id: 'creative-spark',
        name: 'Creative Spark V2',
        category: 'creative',
        currentRank: 6,
        previousRank: 5,
        winStreak: 1,
        specialization: ['Brainstorming', 'Ideation', 'Inspiration'],
        lastActive: new Date(Date.now() - 1000 * 60 * 60), // 1 hour ago
        momentum: 'falling'
      }
    ];

    setContenders(initialContenders);

    // Simulate ranking changes every 30 seconds
    const rankingUpdates = setInterval(() => {
      setContenders(prevContenders => {
        return prevContenders.map(contender => {
          const shouldUpdate = Math.random() < 0.3; // 30% chance of change
          if (!shouldUpdate) return contender;

          // Simulate small ranking fluctuations
          const rankChange = Math.floor(Math.random() * 3) - 1; // -1, 0, or 1
          const newRank = Math.max(1, Math.min(6, contender.currentRank + rankChange));
          
          // Update momentum based on rank change
          let newMomentum: 'rising' | 'falling' | 'stable' = contender.momentum;
          if (newRank < contender.currentRank) newMomentum = 'rising';
          else if (newRank > contender.currentRank) newMomentum = 'falling';
          else newMomentum = 'stable';

          // Simulate activity
          const activityChance = Math.random();
          let lastActive = contender.lastActive;
          if (activityChance < 0.2) { // 20% chance of recent activity
            lastActive = new Date();
          }

          return {
            ...contender,
            previousRank: contender.currentRank,
            currentRank: newRank,
            momentum: newMomentum,
            lastActive,
            winStreak: newMomentum === 'rising' ? contender.winStreak + 1 : 
                      newMomentum === 'falling' ? Math.max(0, contender.winStreak - 1) : 
                      contender.winStreak
          };
        }).sort((a, b) => a.currentRank - b.currentRank);
      });
    }, 30000); // Update every 30 seconds

    return () => clearInterval(rankingUpdates);
  }, []);

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1: return <Crown className="h-5 w-5 text-yellow-400" />;
      case 2: return <Trophy className="h-5 w-5 text-gray-400" />;
      case 3: return <Star className="h-5 w-5 text-amber-600" />;
      default: return <Shield className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getMomentumIcon = (momentum: string, currentRank: number, previousRank: number) => {
    if (currentRank < previousRank) return <TrendingUp className="h-4 w-4 text-green-400" />;
    if (currentRank > previousRank) return <TrendingDown className="h-4 w-4 text-red-400" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'reasoning': return <Zap className="h-4 w-4" />;
      case 'creative': return <Star className="h-4 w-4" />;
      case 'analytical': return <Shield className="h-4 w-4" />;
      case 'conversational': return <Trophy className="h-4 w-4" />;
      default: return <Shield className="h-4 w-4" />;
    }
  };

  const getTimeAgo = (date: Date) => {
    const minutes = Math.floor((Date.now() - date.getTime()) / (1000 * 60));
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  const getRankChangeDescription = (current: number, previous: number) => {
    if (current < previous) return `↑${previous - current}`;
    if (current > previous) return `↓${current - previous}`;
    return '—';
  };

  return (
    <div className="space-y-6">
      {/* Leaderboard Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="glass-subtle p-2 rounded-xl">
            <Trophy className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h3 className="heading-refined text-xl">Strategic Leaderboard</h3>
            <p className="text-sm text-muted-foreground">Live rankings • Updates every 30s</p>
          </div>
        </div>

        {/* Display Mode Toggle */}
        <div className="glass-minimal p-1 rounded-lg">
          <div className="flex gap-1">
            {['overall', 'category', 'recent'].map(mode => (
              <button
                key={mode}
                onClick={() => setDisplayMode(mode as 'overall' | 'category' | 'recent')}
                className={`px-3 py-1 text-xs rounded transition-colors capitalize ${
                  displayMode === mode
                    ? 'bg-primary text-primary-foreground'
                    : 'hover:bg-muted/20'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Leaderboard Entries */}
      <div className="space-y-3">
        {contenders.map((contender, index) => {
          const isTopThree = contender.currentRank <= 3;
          const hasRecentActivity = (Date.now() - contender.lastActive.getTime()) < 1000 * 60 * 30; // 30 minutes

          return (
            <div
              key={contender.id}
              className={`
                relative overflow-hidden rounded-lg transition-all duration-500
                ${isTopThree ? 'glass-panel border border-primary/20' : 'glass-minimal'}
                ${hasRecentActivity ? 'ring-2 ring-green-400/20 shadow-green-400/10' : ''}
                hover:shadow-lg group
              `}
            >
              {/* Rank Position Indicator */}
              <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-primary to-secondary" />
              
              <div className="p-4">
                <div className="flex items-center justify-between">
                  
                  {/* Left: Rank and Info */}
                  <div className="flex items-center gap-4">
                    
                    {/* Rank Display */}
                    <div className="flex items-center gap-2">
                      <div className={`
                        w-10 h-10 rounded-lg flex items-center justify-center font-bold
                        ${isTopThree ? 'bg-gradient-to-br from-primary/20 to-secondary/10' : 'bg-muted/10'}
                      `}>
                        {getRankIcon(contender.currentRank)}
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-mono text-primary">#{contender.currentRank}</div>
                        <div className="flex items-center text-xs text-muted-foreground">
                          {getMomentumIcon(contender.momentum, contender.currentRank, contender.previousRank)}
                          <span className="ml-1">{getRankChangeDescription(contender.currentRank, contender.previousRank)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Contender Info */}
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <h4 className="heading-refined font-semibold">{contender.name}</h4>
                        {hasRecentActivity && (
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                        )}
                      </div>
                      
                      <div className="flex items-center gap-3 text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          {getCategoryIcon(contender.category)}
                          <span className="capitalize">{contender.category}</span>
                        </div>
                        <div>•</div>
                        <div>{contender.winStreak} win streak</div>
                        <div>•</div>
                        <div>{getTimeAgo(contender.lastActive)}</div>
                      </div>

                      {/* Specializations */}
                      <div className="flex gap-2 mt-2">
                        {contender.specialization.slice(0, 3).map(spec => (
                          <div 
                            key={spec}
                            className="px-2 py-1 bg-primary/10 text-primary rounded text-xs font-mono"
                          >
                            {spec}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Right: Performance Indicators */}
                  <div className="text-right space-y-2">
                    
                    {/* Momentum Indicator */}
                    <div className={`
                      px-3 py-1 rounded-full text-xs font-medium
                      ${contender.momentum === 'rising' ? 'bg-green-500/10 text-green-400' :
                        contender.momentum === 'falling' ? 'bg-red-500/10 text-red-400' :
                        'bg-muted/10 text-muted-foreground'}
                    `}>
                      {contender.momentum === 'rising' ? '↗ Rising' :
                       contender.momentum === 'falling' ? '↘ Falling' : '— Stable'}
                    </div>

                    {/* Win Streak Display */}
                    {contender.winStreak > 5 && (
                      <div className="flex items-center gap-1 text-xs text-yellow-400">
                        <Star className="h-3 w-3" />
                        <span>Hot Streak!</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Top 3 Special Effects */}
              {isTopThree && (
                <div className="absolute inset-0 pointer-events-none">
                  <div className={`
                    absolute inset-0 rounded-lg opacity-20
                    ${contender.currentRank === 1 ? 'bg-gradient-to-r from-yellow-400/10 to-amber-500/10' :
                      contender.currentRank === 2 ? 'bg-gradient-to-r from-gray-400/10 to-slate-500/10' :
                      'bg-gradient-to-r from-amber-600/10 to-orange-500/10'}
                  `} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Leaderboard Statistics */}
      <div className="glass-minimal p-4 rounded-lg">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-lg font-mono text-primary">
              {contenders.filter(c => c.momentum === 'rising').length}
            </div>
            <div className="text-xs text-muted-foreground">Rising</div>
          </div>
          <div>
            <div className="text-lg font-mono text-secondary">
              {Math.max(...contenders.map(c => c.winStreak))}
            </div>
            <div className="text-xs text-muted-foreground">Max Streak</div>
          </div>
          <div>
            <div className="text-lg font-mono text-accent">
              {contenders.filter(c => (Date.now() - c.lastActive.getTime()) < 1000 * 60 * 30).length}
            </div>
            <div className="text-xs text-muted-foreground">Active</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const DynamicLeaderboard = memo(DynamicLeaderboardComponent);