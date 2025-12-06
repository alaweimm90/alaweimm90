import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Progress } from '@/components/ui/progress';
import { Trophy, Users, Target, Calendar, Award, TrendingUp } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface Challenge {
  id: string;
  title: string;
  description: string;
  challenge_type: string;
  target_value: number;
  target_unit: string;
  start_date: string;
  end_date: string;
  reward_points: number;
  participant_count?: number;
  user_progress?: number;
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon_name: string;
  points_value: number;
  unlocked_at?: string;
}

interface LeaderboardEntry {
  id: string;
  user_id: string;
  display_name: string;
  total_points: number;
  rank: number;
}

export const CommunityHub: React.FC = () => {
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [userPoints, setUserPoints] = useState(0);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    fetchCommunityData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchCommunityData = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      // For now, use mock data until types are updated
      const mockChallenges: Challenge[] = [
        {
          id: '1',
          title: 'Weekly Workout Challenge',
          description: 'Complete 5 workouts this week',
          challenge_type: 'workout_frequency',
          target_value: 5,
          target_unit: 'workouts',
          start_date: new Date().toISOString(),
          end_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
          reward_points: 100,
          participant_count: 15,
          user_progress: 3
        }
      ];

      const mockAchievements: Achievement[] = [
        {
          id: '1',
          title: 'First Workout',
          description: 'Complete your first workout',
          icon_name: 'trophy',
          points_value: 50,
          unlocked_at: new Date().toISOString()
        },
        {
          id: '2',
          title: 'Week Warrior',
          description: 'Complete 7 consecutive days of workouts',
          icon_name: 'medal',
          points_value: 200
        }
      ];

      const mockLeaderboard: LeaderboardEntry[] = [
        {
          id: '1',
          user_id: user.id,
          display_name: 'You',
          total_points: 350,
          rank: 5
        },
        {
          id: '2',
          user_id: '2',
          display_name: 'Top Performer',
          total_points: 1250,
          rank: 1
        }
      ];

      setChallenges(mockChallenges);
      setAchievements(mockAchievements);
      setLeaderboard(mockLeaderboard);
      setUserPoints(350);
    } catch (error) {
      console.error('Error fetching community data:', error);
      toast({
        title: "Error",
        description: "Failed to load community data.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const joinChallenge = async (challengeId: string) => {
    try {
      // Mock functionality for now
      toast({
        title: "Challenge Joined!",
        description: "You've successfully joined the challenge.",
      });

      // Update the challenges state to show user participation
      setChallenges(prev => prev.map(challenge => 
        challenge.id === challengeId 
          ? { ...challenge, participant_count: (challenge.participant_count || 0) + 1 }
          : challenge
      ));
    } catch (error) {
      console.error('Error joining challenge:', error);
      toast({
        title: "Error",
        description: "Failed to join challenge.",
        variant: "destructive",
      });
    }
  };

  const renderChallenge = (challenge: Challenge) => {
    const isActive = new Date(challenge.end_date) > new Date();
    const progress = challenge.user_progress || 0;
    const progressPercentage = (progress / challenge.target_value) * 100;

    return (
      <Card key={challenge.id} className="mb-4">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">{challenge.title}</CardTitle>
            <div className="flex gap-2">
              <Badge variant={isActive ? "default" : "secondary"}>
                {isActive ? "Active" : "Ended"}
              </Badge>
              <Badge variant="outline">
                {challenge.reward_points} pts
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">{challenge.description}</p>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Progress</span>
              <span>{progress} / {challenge.target_value} {challenge.target_unit}</span>
            </div>
            <Progress value={progressPercentage} className="h-2" />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Users className="h-4 w-4" />
              <span>{challenge.participant_count || 0} participants</span>
            </div>
            
            {isActive && (
              <Button
                size="sm"
                onClick={() => joinChallenge(challenge.id)}
              >
                Join Challenge
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderAchievement = (achievement: Achievement) => {
    const isUnlocked = !!achievement.unlocked_at;
    
    return (
      <Card key={achievement.id} className={`mb-3 ${!isUnlocked ? 'opacity-50' : ''}`}>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-full ${isUnlocked ? 'bg-primary' : 'bg-muted'}`}>
              <Award className={`h-5 w-5 ${isUnlocked ? 'text-primary-foreground' : 'text-muted-foreground'}`} />
            </div>
            <div className="flex-1">
              <h4 className="font-medium">{achievement.title}</h4>
              <p className="text-sm text-muted-foreground">{achievement.description}</p>
            </div>
            <div className="text-right">
              <Badge variant={isUnlocked ? "default" : "secondary"}>
                {achievement.points_value} pts
              </Badge>
              {isUnlocked && (
                <div className="text-xs text-muted-foreground mt-1">
                  Unlocked {new Date(achievement.unlocked_at!).toLocaleDateString()}
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-center space-y-2">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
          <p className="text-sm text-muted-foreground">Loading community data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Community Hub
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between p-4 bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg">
            <div>
              <h3 className="font-semibold">Your Points</h3>
              <p className="text-2xl font-bold text-primary">{userPoints}</p>
            </div>
            <Trophy className="h-8 w-8 text-primary" />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="challenges" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="challenges">Challenges</TabsTrigger>
          <TabsTrigger value="achievements">Achievements</TabsTrigger>
          <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
        </TabsList>
        
        <TabsContent value="challenges" className="space-y-4">
          {challenges.length === 0 ? (
            <Card>
              <CardContent className="text-center py-8">
                <Target className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="font-medium mb-2">No Active Challenges</h3>
                <p className="text-sm text-muted-foreground">
                  Check back soon for new challenges to join!
                </p>
              </CardContent>
            </Card>
          ) : (
            challenges.map(renderChallenge)
          )}
        </TabsContent>
        
        <TabsContent value="achievements" className="space-y-4">
          {achievements.length === 0 ? (
            <Card>
              <CardContent className="text-center py-8">
                <Award className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="font-medium mb-2">No Achievements Yet</h3>
                <p className="text-sm text-muted-foreground">
                  Complete workouts and challenges to unlock achievements!
                </p>
              </CardContent>
            </Card>
          ) : (
            achievements.map(renderAchievement)
          )}
        </TabsContent>
        
        <TabsContent value="leaderboard" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Global Leaderboard
              </CardTitle>
            </CardHeader>
            <CardContent>
              {leaderboard.length === 0 ? (
                <div className="text-center py-8">
                  <Trophy className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="font-medium mb-2">Leaderboard Coming Soon</h3>
                  <p className="text-sm text-muted-foreground">
                    Start earning points to appear on the leaderboard!
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {leaderboard.map((entry, index) => (
                    <div
                      key={entry.id}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                          index === 0 ? 'bg-yellow-500 text-white' :
                          index === 1 ? 'bg-gray-400 text-white' :
                          index === 2 ? 'bg-repz-orange text-white' :
                          'bg-muted-foreground/20'
                        }`}>
                          {entry.rank}
                        </div>
                        <span className="font-medium">{entry.display_name || 'Anonymous'}</span>
                      </div>
                      <Badge variant="outline">{entry.total_points} pts</Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};