import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  BarChart3, 
  Swords, 
  Lightbulb, 
  GraduationCap,
  TrendingUp,
  Clock,
  Users,
  Trophy,
  Activity,
  ArrowRight,
  AlertCircle,
  RefreshCw
} from "lucide-react";
import { Link } from "react-router-dom";
import { useState, useEffect, memo } from "react";
import { trackEvent } from "@/lib/analytics";
import { withErrorBoundary } from "@/components/ErrorBoundary";

const Dashboard = memo(() => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    
    // Track dashboard view
    trackEvent("dashboard_viewed", { timestamp: Date.now() });
    
    return () => clearTimeout(timer);
  }, []);
  const recentActivity = [
    {
      id: 1,
      type: "debate",
      title: "AI Regulation Debate",
      models: ["GPT-4o", "Claude 3.5 Sonnet"],
      timestamp: "2 hours ago",
      result: "GPT-4o won with 92% accuracy",
      icon: Swords,
      status: "completed"
    },
    {
      id: 2,
      type: "benchmark",
      title: "MMLU Benchmark",
      models: ["Gemini 1.5 Pro", "Llama 3.1 70B"],
      timestamp: "4 hours ago", 
      result: "Gemini 1.5 Pro: 87.4% accuracy",
      icon: BarChart3,
      status: "completed"
    },
    {
      id: 3,
      type: "creative",
      title: "Coffee Brand Campaign",
      models: ["Claude 3.5 Sonnet", "GPT-4o"],
      timestamp: "1 day ago",
      result: "Creative score: 88/100",
      icon: Lightbulb,
      status: "completed"
    },
    {
      id: 4,
      type: "explanation",
      title: "Quantum Physics for Kids",
      models: ["GPT-4o", "Claude 3.5 Sonnet"],
      timestamp: "2 days ago",
      result: "Clarity: 91%, Empathy: 85%",
      icon: GraduationCap,
      status: "completed"
    }
  ];

  const stats = [
    {
      label: "Total Evaluations",
      value: "2,847",
      change: "+12.5%",
      icon: Activity,
      status: "trust"
    },
    {
      label: "Active Models",
      value: "8",
      change: "+2",
      icon: Users,
      status: "trust"
    },
    {
      label: "Avg Accuracy",
      value: "87.3%",
      change: "+2.1%",
      icon: Trophy,
      status: "trust"
    },
    {
      label: "This Month",
      value: "234",
      change: "+18.2%",
      icon: TrendingUp,
      status: "trust"
    }
  ];

  const quickActions = [
    {
      title: "Start Debate",
      description: "Pit models against each other",
      icon: Swords,
      href: "/arena",
      color: "bg-accent/10 text-accent hover:bg-accent/20",
      action: "debate_started"
    },
    {
      title: "Run Benchmark",
      description: "Test systematic performance",
      icon: BarChart3,
      href: "/bench",
      color: "bg-primary/10 text-primary hover:bg-primary/20",
      action: "benchmark_started"
    },
    {
      title: "Creative Challenge",
      description: "Test collaborative creativity",
      icon: Lightbulb,
      href: "/arena",
      color: "bg-accent/10 text-accent hover:bg-accent/20",
      action: "creative_challenge_started"
    },
    {
      title: "Explanation Test",
      description: "Evaluate teaching ability",
      icon: GraduationCap,
      href: "/arena",
      color: "bg-primary/10 text-primary hover:bg-primary/20",
      action: "explanation_test_started"
    }
  ];

  const handleQuickAction = (action: string, title: string) => {
    trackEvent("quick_action_clicked", { action, title });
  };

  const handleRefresh = () => {
    setIsLoading(true);
    setError(null);
    trackEvent("dashboard_refreshed");
    setTimeout(() => setIsLoading(false), 800);
  };

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-96 space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <h3 className="text-lg font-semibold">Failed to load dashboard</h3>
        <p className="text-muted-foreground text-center max-w-md">{error}</p>
        <Button onClick={handleRefresh} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Stats Overview */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, index) => (
            <Card key={index} className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-8 w-16" />
                  <Skeleton className="h-4 w-12" />
                </div>
                <Skeleton className="h-12 w-12 rounded-xl" />
              </div>
            </Card>
          ))
        ) : (
          stats.map((stat, index) => (
          <Card 
            key={index} 
            className={`
              group p-6 border-border/50 bg-card/60 backdrop-blur-sm
              transition-all duration-300 hover:shadow-trust hover:-translate-y-1
              ${stat.status === 'trust' ? 'hover:border-trust/50' : ''}
            `}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                <p className="text-2xl font-bold text-foreground font-mono">{stat.value}</p>
                <p className="text-sm text-trust flex items-center gap-1 font-medium">
                  <TrendingUp className="h-3 w-3" />
                  {stat.change}
                </p>
              </div>
              <div className={`
                p-3 rounded-xl transition-all duration-300
                ${stat.status === 'trust' 
                  ? 'bg-trust/10 group-hover:bg-trust/20 group-hover:shadow-trust' 
                  : 'bg-primary/10 group-hover:bg-primary/20'
                }
              `}>
                <stat.icon className={`h-6 w-6 ${
                  stat.status === 'trust' ? 'text-trust' : 'text-primary'
                }`} />
              </div>
            </div>
            </Card>
          ))
        )}
      </div>

      {/* Quick Actions */}
      <Card className="p-6">
        <h3 className="text-lg font-bold text-foreground mb-4">Quick Actions</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {isLoading ? (
            Array.from({ length: 4 }).map((_, index) => (
              <Skeleton key={index} className="h-24 rounded-lg" />
            ))
          ) : (
            quickActions.map((action, index) => (
              <Button
                key={index}
                variant="outline"
                size="lg"
                asChild
                className={`h-auto p-4 flex flex-col items-center gap-2 ${action.color} border-border hover:border-current`}
                onClick={() => handleQuickAction(action.action, action.title)}
              >
                <Link to={action.href}>
                  <action.icon className="h-6 w-6" />
                  <div className="text-center">
                    <div className="font-medium">{action.title}</div>
                    <div className="text-xs opacity-80">{action.description}</div>
                  </div>
                </Link>
              </Button>
            ))
          )}
        </div>
      </Card>

      {/* Recent Activity */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-foreground">Recent Activity</h3>
              <Button variant="ghost" size="sm">
                View All
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </div>

            <div className="space-y-4">
              {isLoading ? (
                Array.from({ length: 4 }).map((_, index) => (
                  <div key={index} className="flex items-start gap-4 p-4">
                    <Skeleton className="h-8 w-8 rounded-lg" />
                    <div className="flex-1 space-y-2">
                      <Skeleton className="h-4 w-3/4" />
                      <Skeleton className="h-3 w-1/2" />
                      <Skeleton className="h-3 w-1/3" />
                      <Skeleton className="h-3 w-1/4" />
                    </div>
                  </div>
                ))
              ) : (
                recentActivity.map((activity) => (
                <div key={activity.id} className="flex items-start gap-4 p-4 rounded-lg hover:bg-muted/50 transition-colors">
                  <div className="p-2 rounded-lg bg-accent/10">
                    <activity.icon className="h-4 w-4 text-accent" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-medium text-foreground truncate">{activity.title}</h4>
                      <Badge variant="outline" className="text-xs">
                        {activity.type}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-1">
                      {activity.models.join(" vs ")}
                    </p>
                    <p className="text-sm text-accent font-medium mb-1">
                      {activity.result}
                    </p>
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {activity.timestamp}
                    </div>
                  </div>
                </div>
                ))
              )}
            </div>
          </Card>
        </div>

        {/* Performance Insights */}
        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-bold text-foreground mb-4">Top Performers</h3>
            <div className="space-y-3">
              {[
                { model: "GPT-4o", score: 92.3, trend: "+2.1%" },
                { model: "Claude 3.5 Sonnet", score: 89.7, trend: "+1.8%" },
                { model: "Gemini 1.5 Pro", score: 87.4, trend: "+0.9%" },
                { model: "Llama 3.1 70B", score: 84.2, trend: "-0.3%" }
              ].map((performer, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-foreground">{performer.model}</div>
                    <div className="text-sm text-muted-foreground">{performer.score}% avg</div>
                  </div>
                  <Badge className={`${
                    performer.trend.startsWith('+') 
                      ? 'bg-accent/10 text-accent' 
                      : 'bg-primary/10 text-primary'
                  }`}>
                    {performer.trend}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-lg font-bold text-foreground mb-4">Quick Stats</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Arena Sessions</span>
                <span className="font-medium text-foreground">156</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Benchmark Runs</span>
                <span className="font-medium text-foreground">78</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Evaluations</span>
                <span className="font-medium text-foreground">2,847</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Success Rate</span>
                <span className="font-medium text-accent">94.2%</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
});

Dashboard.displayName = "Dashboard";

export { Dashboard };
export default withErrorBoundary(Dashboard);