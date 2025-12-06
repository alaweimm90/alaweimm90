import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronRight } from 'lucide-react';

interface MethodCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  metrics: Array<{
    label: string;
    value: string | number;
    color: string;
    description?: string;
  }>;
  status?: 'active' | 'beta' | 'experimental';
  gradient: string;
  onLearnMore?: () => void;
}

export default function MethodCard({
  title,
  description,
  icon,
  metrics,
  status = 'active',
  gradient,
  onLearnMore
}: MethodCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const statusColors = {
    active: 'border-success/30 text-success',
    beta: 'border-warning/30 text-warning', 
    experimental: 'border-info/30 text-info'
  };

  return (
    <Card 
      className={`
        group relative overflow-hidden border-border/40 
        hover:border-primary/30 transition-all duration-300 
        hover:shadow-glow hover:-translate-y-1
        ${gradient}
      `}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="absolute inset-0 bg-card/80 backdrop-blur-sm" />
      
      <CardHeader className="relative z-10">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-background/50 border border-border/20">
              {icon}
            </div>
            <div>
              <CardTitle className="text-xl font-semibold tracking-tight">
                {title}
              </CardTitle>
              <CardDescription className="text-sm text-muted-foreground mt-1">
                {description}
              </CardDescription>
            </div>
          </div>
          
          {status !== 'active' && (
            <Badge 
              variant="outline" 
              className={`text-xs font-mono ${statusColors[status]} capitalize`}
            >
              {status}
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="relative z-10 space-y-4">
        <div className="grid gap-3">
          {metrics.map((metric, index) => (
            <div 
              key={index}
              className="flex items-center justify-between p-3 rounded-lg bg-background/30 border border-border/20"
            >
              <div className="flex flex-col">
                <span className="text-sm font-medium text-foreground">
                  {metric.label}
                </span>
                {metric.description && (
                  <span className="text-xs text-muted-foreground">
                    {metric.description}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${metric.color}`} />
                <span className={`font-mono text-sm font-semibold ${metric.color.replace('bg-', 'text-')}`}>
                  {metric.value}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className={`
          transition-all duration-300 
          ${isHovered ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}
        `}>
          <Button 
            variant="ghost" 
            onClick={onLearnMore}
            className="w-full justify-between text-primary hover:text-primary-foreground hover:bg-primary/90 transition-all duration-200"
          >
            <span className="font-medium">Learn more</span>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>

      {/* Subtle glow effect */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-20 transition-opacity duration-300 bg-gradient-to-br from-primary/20 via-transparent to-accent/20 pointer-events-none" />
    </Card>
  );
}