import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import { Crown, TrendingUp, Award, Star } from "lucide-react";

interface RankingItem {
  position: number;
  name: string;
  score: number;
  tier: 'platinum' | 'gold' | 'silver' | 'bronze' | 'standard';
  performance: 'elite' | 'superior' | 'standard';
  change: number;
}

const mockRankings: RankingItem[] = [
  { position: 1, name: "GPT-4 Turbo", score: 1847, tier: 'platinum', performance: 'elite', change: +23 },
  { position: 2, name: "Claude-3 Opus", score: 1823, tier: 'gold', performance: 'elite', change: +15 },
  { position: 3, name: "Gemini Ultra", score: 1789, tier: 'gold', performance: 'superior', change: -8 },
  { position: 4, name: "GPT-3.5 Turbo", score: 1756, tier: 'silver', performance: 'superior', change: +12 },
  { position: 5, name: "Claude-3 Sonnet", score: 1734, tier: 'bronze', performance: 'standard', change: +5 },
];

const getTierClasses = (tier: string) => {
  switch (tier) {
    case 'platinum': return 'rank-platinum strategic-rank';
    case 'gold': return 'rank-gold strategic-rank';
    case 'silver': return 'rank-silver strategic-rank';
    case 'bronze': return 'rank-bronze strategic-rank';
    default: return 'strategic-rank performance-standard';
  }
};

const getPerformanceClasses = (performance: string) => {
  switch (performance) {
    case 'elite': return 'performance-elite strategic-rank';
    case 'superior': return 'performance-superior strategic-rank';
    default: return 'performance-standard strategic-rank';
  }
};

const getRankIcon = (position: number) => {
  switch (position) {
    case 1: return <Crown className="h-4 w-4" />;
    case 2: case 3: return <Award className="h-4 w-4" />;
    default: return <Star className="h-4 w-4" />;
  }
};

const StrategicRankingsComponent = () => {
  return (
    <div className="glass-panel p-6 rounded-xl">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="glass-subtle p-2 rounded-lg">
            <TrendingUp className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="heading-refined text-lg">Strategic Rankings</h3>
            <p className="text-xs text-muted-foreground">Elite AI Command Leaderboard</p>
          </div>
        </div>
        <Badge className="glass-minimal border-0 px-3 py-1">
          <span className="text-xs text-primary font-mono">LIVE</span>
        </Badge>
      </div>

      <div className="space-y-3">
        {mockRankings.map((item) => (
          <div key={item.name} className="card-sophisticated p-4 hover:transform hover:scale-[1.02] transition-all duration-300">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Badge className={getTierClasses(item.tier)}>
                    {getRankIcon(item.position)}
                    <span className="ml-1 font-mono">#{item.position}</span>
                  </Badge>
                  <Badge className={getPerformanceClasses(item.performance)}>
                    {item.performance}
                  </Badge>
                </div>
                
                <div>
                  <h4 className="heading-refined text-sm">{item.name}</h4>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>ELO: {item.score.toLocaleString()}</span>
                    <span className={`font-mono ${item.change > 0 ? 'text-green-600' : 'text-red-500'}`}>
                      {item.change > 0 ? '+' : ''}{item.change}
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-right">
                <div className="text-sm font-semibold text-primary">
                  {(item.score / 10).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Win Rate</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-border/30">
        <div className="flex justify-between items-center text-xs text-muted-foreground">
          <span>Updated 2 minutes ago</span>
          <span className="font-mono">5,247 evaluations today</span>
        </div>
      </div>
    </div>
  );
};

export const StrategicRankings = memo(StrategicRankingsComponent);