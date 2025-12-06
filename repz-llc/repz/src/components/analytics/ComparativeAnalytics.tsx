import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/ui/molecules/Card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/molecules/Tabs"
import { Button } from "@/ui/atoms/Button"
import { Badge } from "@/ui/atoms/Badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/ui/atoms/Avatar"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts"
import { Users, Trophy, TrendingUp, Target, Medal, Crown } from "lucide-react"

const peerComparisonData = [
  { metric: "Workout Frequency", you: 5.2, peers: 4.1, percentile: 78 },
  { metric: "Strength Progress", you: 15, peers: 12, percentile: 85 },
  { metric: "Cardio Endurance", you: 8.5, peers: 7.2, percentile: 72 },
  { metric: "Consistency Score", you: 89, peers: 76, percentile: 91 },
  { metric: "Goal Achievement", you: 67, peers: 58, percentile: 88 }
]

const ageGroupData = [
  { range: "25-30", strength: 85, cardio: 78, flexibility: 65 },
  { range: "30-35", strength: 82, cardio: 75, flexibility: 68 },
  { range: "35-40", strength: 79, cardio: 73, flexibility: 70 },
  { range: "40-45", strength: 76, cardio: 70, flexibility: 72 },
  { range: "You", strength: 92, cardio: 85, flexibility: 75 }
]

const leaderboardData = [
  { rank: 1, name: "Alex Johnson", score: 952, avatar: "/placeholder-avatar.png", trend: "+5" },
  { rank: 2, name: "Sarah Chen", score: 948, avatar: "/placeholder-avatar.png", trend: "+2" },
  { rank: 3, name: "Mike Rodriguez", score: 935, avatar: "/placeholder-avatar.png", trend: "-1" },
  { rank: 4, name: "You", score: 912, avatar: "/placeholder-avatar.png", trend: "+8" },
  { rank: 5, name: "Emma Davis", score: 898, avatar: "/placeholder-avatar.png", trend: "+3" },
  { rank: 6, name: "John Smith", score: 887, avatar: "/placeholder-avatar.png", trend: "-2" },
  { rank: 7, name: "Lisa Wang", score: 876, avatar: "/placeholder-avatar.png", trend: "+1" },
]

const progressComparison = [
  { month: "Oct", you: 75, similarGoals: 68 },
  { month: "Nov", you: 82, similarGoals: 73 },
  { month: "Dec", you: 88, similarGoals: 78 },
  { month: "Jan", you: 94, similarGoals: 82 },
]

export function ComparativeAnalytics() {
  const [comparisonType, setComparisonType] = useState("peers")

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1: return <Crown className="h-4 w-4 text-yellow-500" />
      case 2: return <Medal className="h-4 w-4 text-gray-400" />
      case 3: return <Medal className="h-4 w-4 text-amber-600" />
      default: return <Trophy className="h-4 w-4 text-muted-foreground" />
    }
  }

  const getTrendColor = (trend: string) => {
    if (trend.startsWith("+")) return "text-green-600"
    if (trend.startsWith("-")) return "text-red-600"
    return "text-muted-foreground"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Users className="h-6 w-6 text-primary" />
          <div>
            <h2 className="text-2xl font-bold">Comparative Analytics</h2>
            <p className="text-muted-foreground">See how you stack up against your peers</p>
          </div>
        </div>
        <div className="flex gap-2">
          {[
            { value: "peers", label: "Peers" },
            { value: "age-group", label: "Age Group" },
            { value: "goals", label: "Similar Goals" }
          ].map((type) => (
            <Button
              key={type.value}
              variant={comparisonType === type.value ? "default" : "outline"}
              size="sm"
              onClick={() => setComparisonType(type.value)}
            >
              {type.label}
            </Button>
          ))}
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="achievements">Achievements</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Overall Ranking</CardTitle>
                <Trophy className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">#4</div>
                <div className="flex items-center text-xs text-muted-foreground">
                  <TrendingUp className="h-3 w-3 mr-1 text-green-600" />
                  +2 positions this week
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Percentile</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">82nd</div>
                <div className="text-xs text-muted-foreground">
                  Better than 82% of users
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Fitness Score</CardTitle>
                <Medal className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">912</div>
                <div className="flex items-center text-xs text-muted-foreground">
                  <TrendingUp className="h-3 w-3 mr-1 text-green-600" />
                  +48 points this month
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Performance vs Peers</CardTitle>
              <CardDescription>How you compare across key fitness metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={peerComparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="you" fill="hsl(var(--primary))" name="You" />
                  <Bar dataKey="peers" fill="hsl(var(--muted))" name="Peer Average" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Progress vs Similar Goals</CardTitle>
                <CardDescription>Compare with users who have similar fitness objectives</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={progressComparison}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="you" stroke="hsl(var(--primary))" strokeWidth={3} name="You" />
                    <Line type="monotone" dataKey="similarGoals" stroke="hsl(var(--secondary))" strokeWidth={2} name="Similar Goals" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Strengths & Areas for Improvement</CardTitle>
                <CardDescription>Based on peer comparison analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <h4 className="font-medium text-green-600">Strengths</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Workout Consistency</span>
                      <Badge variant="secondary">Top 10%</Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Strength Progress</span>
                      <Badge variant="secondary">Top 15%</Badge>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium text-repz-orange">Areas to Improve</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Flexibility Training</span>
                      <Badge variant="outline">Below Average</Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Recovery Time</span>
                      <Badge variant="outline">Average</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="leaderboard" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Fitness Leaderboard</CardTitle>
              <CardDescription>Top performers in your peer group this month</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {leaderboardData.map((user) => (
                  <div 
                    key={user.rank} 
                    className={`flex items-center gap-4 p-3 rounded-lg border ${
                      user.name === "You" ? "bg-primary/5 border-primary" : "bg-background"
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-[60px]">
                      {getRankIcon(user.rank)}
                      <span className="font-bold text-lg">#{user.rank}</span>
                    </div>
                    
                    <Avatar className="h-10 w-10">
                      <AvatarImage src={user.avatar} />
                      <AvatarFallback>{user.name.charAt(0)}</AvatarFallback>
                    </Avatar>
                    
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">
                        {user.name}
                        {user.name === "You" && <Badge variant="secondary" className="ml-2">You</Badge>}
                      </p>
                      <p className="text-sm text-muted-foreground">Score: {user.score}</p>
                    </div>
                    
                    <div className={`text-sm font-medium ${getTrendColor(user.trend)}`}>
                      {user.trend}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Age Group Performance</CardTitle>
              <CardDescription>How you compare within your age demographic</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={ageGroupData.slice(-1).concat(ageGroupData.slice(0, -1))}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="range" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} />
                  <Radar name="Strength" dataKey="strength" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.1} />
                  <Radar name="Cardio" dataKey="cardio" stroke="hsl(var(--secondary))" fill="hsl(var(--secondary))" fillOpacity={0.1} />
                  <Radar name="Flexibility" dataKey="flexibility" stroke="hsl(var(--accent))" fill="hsl(var(--accent))" fillOpacity={0.1} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="achievements" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Comparative Achievements</CardTitle>
              <CardDescription>Milestones reached compared to your peer group</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-3">
                  <h4 className="font-medium">Recent Achievements</h4>
                  <div className="space-y-2">
                    <div className="flex items-center gap-3 p-2 bg-green-50 rounded-lg border">
                      <Medal className="h-5 w-5 text-green-600" />
                      <div>
                        <p className="font-medium text-sm">30-Day Streak</p>
                        <p className="text-xs text-muted-foreground">Achieved by only 15% of users</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-2 bg-blue-50 rounded-lg border">
                      <Trophy className="h-5 w-5 text-blue-600" />
                      <div>
                        <p className="font-medium text-sm">Strength Milestone</p>
                        <p className="text-xs text-muted-foreground">Top 20% in your age group</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <h4 className="font-medium">Upcoming Milestones</h4>
                  <div className="space-y-2">
                    <div className="flex items-center gap-3 p-2 bg-repz-orange/10 rounded-lg border">
                      <Target className="h-5 w-5 text-repz-orange" />
                      <div>
                        <p className="font-medium text-sm">Elite Status</p>
                        <p className="text-xs text-muted-foreground">88 points away from top 5%</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-2 bg-purple-50 rounded-lg border">
                      <Crown className="h-5 w-5 text-purple-600" />
                      <div>
                        <p className="font-medium text-sm">Consistency Champion</p>
                        <p className="text-xs text-muted-foreground">7 more days for 60-day streak</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}