import { memo, useMemo, useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, TrendingDown, Minus, CheckCircle, AlertCircle, Clock, Zap, Trophy } from "lucide-react";
import { RippleEffect, MagneticHover, GlowOnHover, SlideInView, StaggeredChildren } from "@/components/MicroInteractions";

// Mock data with better typing
const mockBenchmarkData = [
  { name: "GPT-4", score: 92, trend: "up", change: "+3.2%" },
  { name: "Claude-3", score: 89, trend: "up", change: "+2.8%" },
  { name: "Gemini Pro", score: 85, trend: "down", change: "-1.5%" },
  { name: "PaLM-2", score: 81, trend: "stable", change: "0.0%" },
  { name: "LLaMA-2", score: 78, trend: "up", change: "+1.9%" },
] as const;

interface EvaluationCardProps {
  title: string;
  metric: string;
  status: 'active' | 'warning' | 'success';
  icon: React.ComponentType<{ className?: string }>;
  trend?: string;
}

const statusConfig = {
  active: {
    icon: Clock,
    className: "bg-primary/20 text-primary border-primary/30",
    dotClass: "bg-primary"
  },
  warning: {
    icon: AlertCircle,
    className: "bg-warning/20 text-warning border-warning/30",
    dotClass: "bg-warning"
  },
  success: {
    icon: CheckCircle,
    className: "bg-success/20 text-success border-success/30", 
    dotClass: "bg-success"
  }
} as const;

const EvaluationCard = memo(({ title, metric, status, icon: Icon, trend }: EvaluationCardProps) => {
  const config = statusConfig[status];
  const StatusIcon = config.icon;
  const [isHovered, setIsHovered] = useState(false);

  return (
    <RippleEffect>
      <GlowOnHover glowIntensity="subtle">
        <Card 
          className="relative p-4 glass-subtle border-border/20 transition-all duration-500 hover:shadow-lg group cursor-pointer"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          {/* Gradient overlay on hover */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-secondary/3 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-lg"></div>
          
          <CardHeader className="pb-2 relative z-10">
            <div className="flex items-center justify-between">
              <CardTitle className="heading-refined text-sm">{title}</CardTitle>
              <div className="glass-minimal p-1.5 rounded-lg group-hover:shadow-md transition-all duration-300">
                <StatusIcon className={`h-4 w-4 transition-all duration-300 ${
                  isHovered ? 'text-primary scale-110' : 'text-muted-foreground'
                }`} aria-hidden="true" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-0 relative z-10">
            <div className="flex items-center gap-2 mb-2">
              <span className="heading-display text-2xl transition-colors duration-300 group-hover:text-primary">
                {metric}
              </span>
              {trend && (
                <span className="body-elegant text-xs text-muted-foreground group-hover:text-primary/70 transition-colors duration-300">
                  {trend}
                </span>
              )}
            </div>
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs transition-all duration-300 ${config.className}`}>
              <div className={`w-2 h-2 rounded-full ${config.dotClass} ${
                isHovered ? 'animate-pulse' : ''
              }`} aria-hidden="true" />
              <span className="font-medium">
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </span>
            </div>
          </CardContent>
        </Card>
      </GlowOnHover>
    </RippleEffect>
  );
});

EvaluationCard.displayName = 'EvaluationCard';

const TrendIcon = memo(({ trend, className }: { trend: string; className?: string }) => {
  const icons = {
    up: TrendingUp,
    down: TrendingDown,
    stable: Minus
  };
  
  const colors = {
    up: "text-success",
    down: "text-destructive", 
    stable: "text-muted-foreground"
  };
  
  const IconComponent = icons[trend as keyof typeof icons] || Minus;
  const colorClass = colors[trend as keyof typeof colors] || "text-muted-foreground";
  
  return <IconComponent className={`h-4 w-4 ${colorClass} ${className}`} aria-hidden="true" />;
});

TrendIcon.displayName = 'TrendIcon';

export const BenchmarkPanel = memo(() => {
  const [isVisible, setIsVisible] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  const evaluationMetrics = useMemo(() => [
    { title: "Confidence", metric: "94.2%", status: "success" as const, icon: CheckCircle, trend: "+2.1%" },
    { title: "Reliability", metric: "87.8%", status: "active" as const, icon: Clock, trend: "+0.8%" },
    { title: "Bias Score", metric: "2.1/10", status: "warning" as const, icon: AlertCircle, trend: "-0.3%" },
  ], []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (panelRef.current) {
      observer.observe(panelRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <SlideInView direction="up" delay={200}>
      <MagneticHover strength={0.1}>
        <div 
          ref={panelRef}
          className="glass-panel w-full max-w-md mx-auto border-border/10 shadow-lg hover:shadow-xl transition-all duration-500 relative overflow-hidden"
          role="region"
          aria-labelledby="benchmark-title"
        >
          {/* Sophisticated Background Effects */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary/3 to-secondary/2 opacity-0 hover:opacity-100 transition-opacity duration-700"></div>
          <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl from-primary/5 to-transparent rounded-full blur-2xl"></div>
          
          <CardHeader className="pb-6 relative z-10">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <div className="glass-subtle p-2 rounded-xl">
                  <Trophy className="h-5 w-5 text-primary" />
                </div>
                <CardTitle id="benchmark-title" className="heading-refined text-lg">
                  Strategic Analytics
                </CardTitle>
              </div>
              <Badge className="glass-minimal border-0 bg-success/20 text-success px-3 py-1.5">
                <div className="w-2 h-2 bg-success rounded-full mr-2 animate-pulse" aria-hidden="true" />
                <span className="font-medium text-xs">LIVE</span>
              </Badge>
            </div>
            <CardDescription className="body-elegant">
              Real-time strategic intelligence and model performance tracking
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-8 relative z-10">
            {/* Enhanced Models List */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-1 h-5 bg-gradient-to-b from-primary to-secondary rounded-full"></div>
                <h3 className="heading-refined text-sm text-foreground">Strategic Rankings</h3>
              </div>
              
              <StaggeredChildren staggerDelay={80}>
                <div className="space-y-3" role="list" aria-label="Model rankings">
                  {mockBenchmarkData.map((model, index) => (
                    <RippleEffect key={model.name}>
                      <div 
                        className="group flex items-center justify-between p-3 glass-subtle hover:shadow-md transition-all duration-300 cursor-pointer relative overflow-hidden"
                        role="listitem"
                      >
                        {/* Rank indicator with glow */}
                        <div className="flex items-center gap-4">
                          <div className={`
                            strategic-rank flex items-center justify-center w-8 h-6 text-xs font-bold transition-all duration-300 group-hover:scale-110
                            ${index === 0 ? 'rank-platinum' : 
                              index === 1 ? 'rank-gold' :
                              index === 2 ? 'rank-silver' :
                              index === 3 ? 'rank-bronze' :
                              'performance-standard'}
                          `}>
                            #{index + 1}
                          </div>
                          <div className="flex flex-col">
                            <span className="heading-refined text-sm group-hover:text-primary transition-colors">
                              {model.name}
                            </span>
                            <span className="body-elegant text-xs text-muted-foreground">
                              Strategic AI Model
                            </span>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3">
                          <Progress 
                            value={model.score} 
                            className="w-20 h-2"
                            aria-label={`${model.name} score: ${model.score}%`}
                          />
                          <div className="text-right">
                            <div className="heading-display text-sm font-bold group-hover:text-primary transition-colors">
                              {model.score}
                            </div>
                            <div className="flex items-center gap-1 justify-end">
                              <TrendIcon trend={model.trend} className="group-hover:scale-110 transition-transform" />
                              <span className={`body-elegant text-xs transition-colors ${
                                model.trend === 'up' ? 'text-success group-hover:text-success/80' : 
                                model.trend === 'down' ? 'text-destructive group-hover:text-destructive/80' : 
                                'text-muted-foreground group-hover:text-muted-foreground/80'
                              }`}>
                                {model.change}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Hover effect overlay */}
                        <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-secondary/3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg"></div>
                      </div>
                    </RippleEffect>
                  ))}
                </div>
              </StaggeredChildren>
            </div>

            {/* Enhanced Quick Metrics */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-1 h-5 bg-gradient-to-b from-secondary to-accent rounded-full"></div>
                <h3 className="heading-refined text-sm text-foreground">Strategic Metrics</h3>
              </div>
              
              <StaggeredChildren staggerDelay={100}>
                <div className="grid grid-cols-1 gap-3" role="list" aria-label="Evaluation metrics">
                  {evaluationMetrics.map((metric, index) => (
                    <div 
                      key={metric.title} 
                      role="listitem"
                      style={{'--stagger-index': index + 3} as React.CSSProperties}
                    >
                      <EvaluationCard {...metric} />
                    </div>
                  ))}
                </div>
              </StaggeredChildren>
            </div>

            {/* Strategic Status Indicator */}
            <div className="flex items-center justify-center pt-4 border-t border-border/10">
              <div className="flex items-center gap-3 glass-minimal px-4 py-2 rounded-full">
                <Zap className="h-4 w-4 text-primary animate-pulse" />
                <span className="body-elegant text-xs text-primary font-medium">
                  Strategic Systems Online
                </span>
                <div className="w-2 h-2 bg-primary rounded-full animate-ping"></div>
              </div>
            </div>
          </CardContent>
        </div>
      </MagneticHover>
    </SlideInView>
  );
});

BenchmarkPanel.displayName = 'BenchmarkPanel';