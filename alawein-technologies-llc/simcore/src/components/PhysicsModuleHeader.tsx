import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ArrowLeft, Download, Settings, BookOpen, ExternalLink, Atom, Zap, Activity } from 'lucide-react';
import { InlineMath } from '@/components/ui/Math';
import { useDomainTheme } from '@/components/DomainThemeProvider';

interface PhysicsModuleHeaderProps {
  title: string;
  description: string;
  category: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Research';
  equation?: string;
  onExport?: () => void;
  onReset?: () => void;
  isRunning?: boolean;
  currentStep?: number;
  className?: string;
}

const difficultyConfig = {
  'Beginner': { 
    color: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 shadow-emerald-500/20',
    icon: '🌱',
    glow: 'shadow-lg shadow-emerald-500/25'
  },
  'Intermediate': { 
    color: 'bg-blue-500/15 text-blue-400 border-blue-500/30 shadow-blue-500/20',
    icon: '⚡',
    glow: 'shadow-lg shadow-blue-500/25'
  },
  'Advanced': { 
    color: 'bg-amber-500/15 text-amber-400 border-amber-500/30 shadow-amber-500/20',
    icon: '🔬',
    glow: 'shadow-lg shadow-amber-500/25'
  },
  'Research': { 
    color: 'bg-red-500/15 text-red-400 border-red-500/30 shadow-red-500/20',
    icon: '',
    glow: 'shadow-lg shadow-red-500/25'
  }
};

const categoryConfig = {
  'Band Structure': { 
    color: 'bg-primary/15 text-primary border-primary/30',
    pattern: 'quantum-lattice',
    glow: 'shadow-lg shadow-primary/25'
  },
  'Quantum Dynamics': { 
    color: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
    pattern: 'wave-interference',
    glow: 'shadow-lg shadow-purple-500/25'
  },
  'Statistical Physics': { 
    color: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    pattern: 'thermal-noise',
    glow: 'shadow-lg shadow-emerald-500/25'
  },
  'Spin & Magnetism': { 
    color: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
    pattern: 'magnetic-field',
    glow: 'shadow-lg shadow-rose-500/25'
  },
  'Materials & Crystals': { 
    color: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
    pattern: 'crystal-structure',
    glow: 'shadow-lg shadow-cyan-500/25'
  },
  'Field Theory': { 
    color: 'bg-indigo-500/15 text-indigo-400 border-indigo-500/30',
    pattern: 'field-lines',
    glow: 'shadow-lg shadow-indigo-500/25'
  },
  'Machine Learning': { 
    color: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
    pattern: 'neural-network',
    glow: 'shadow-lg shadow-orange-500/25'
  }
};

export const PhysicsModuleHeader: React.FC<PhysicsModuleHeaderProps> = ({
  title,
  description,
  category,
  difficulty,
  equation,
  onExport,
  onReset,
  isRunning,
  currentStep,
  className = ''
}) => {
  const navigate = useNavigate();
  const { currentDomain } = useDomainTheme();
  
  const difficultyStyle = difficultyConfig[difficulty];
  const categoryStyle = categoryConfig[category as keyof typeof categoryConfig] || categoryConfig['Band Structure'];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Elegant Navigation Bar */}
      <div className="flex items-center justify-between backdrop-blur-xl bg-background/80 border border-border/50 rounded-2xl p-4 shadow-xl">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/')}
            className="gap-2 hover:bg-background/50 rounded-xl transition-all duration-300 hover:scale-105"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="font-medium bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
              SimCore Lab
            </span>
          </Button>
          <Separator orientation="vertical" className="h-6 bg-border/50" />
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-primary/10 border border-primary/20">
              <Atom className="w-4 h-4 text-primary" />
            </div>
            <span className="text-sm text-muted-foreground font-medium">Science Engine</span>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {currentStep !== undefined && (
            <Badge variant="outline" className="gap-2 bg-background/50 border-border/50 shadow-sm">
              <Activity className="w-3 h-3" />
              {currentStep.toLocaleString()}
            </Badge>
          )}
          {isRunning !== undefined && (
            <Badge 
              variant={isRunning ? "default" : "secondary"}
              className={`gap-2 ${isRunning ? 'animate-pulse' : ''} shadow-sm`}
            >
              <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-accent' : 'bg-muted'}`} />
              {isRunning ? "Computing" : "Ready"}
            </Badge>
          )}
          <div className="flex items-center gap-2">
            {onReset && (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={onReset}
                className="rounded-xl hover:scale-105 transition-all duration-200 border-border/50 hover:border-border"
              >
                <Settings className="w-4 h-4" />
              </Button>
            )}
            {onExport && (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={onExport}
                className="rounded-xl hover:scale-105 transition-all duration-200 border-border/50 hover:border-border"
              >
                <Download className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Sophisticated Module Header Card */}
      <Card className="border border-border/50 bg-gradient-to-br from-card/80 to-card/40 backdrop-blur-xl shadow-2xl rounded-3xl overflow-hidden relative">
        {/* Subtle Pattern Background */}
        <div className={`absolute inset-0 opacity-5 ${categoryStyle.pattern}`} />
        
        {/* Animated Glow Border */}
        <div className="absolute inset-0 rounded-3xl">
          <div className={`absolute inset-0 rounded-3xl ${categoryStyle.glow} opacity-50 animate-pulse`} />
        </div>
        
        <CardHeader className="pb-6 relative z-10">
          <div className="flex items-start justify-between gap-6">
            <div className="space-y-4 flex-1">
              {/* Refined Badge System */}
              <div className="flex items-center gap-3 flex-wrap">
                <Badge 
                  variant="outline" 
                  className={`${categoryStyle.color} ${categoryStyle.glow} border-2 font-semibold px-4 py-1.5 rounded-full text-sm tracking-wide transition-all duration-300 hover:scale-105`}
                >
                  {category}
                </Badge>
                <Badge 
                  variant="outline" 
                  className={`${difficultyStyle.color} ${difficultyStyle.glow} border-2 font-semibold px-4 py-1.5 rounded-full text-sm tracking-wide transition-all duration-300 hover:scale-105`}
                >
                  {difficultyStyle.icon && <span className="mr-1">{difficultyStyle.icon}</span>}
                  {difficulty}
                </Badge>
              </div>
              
              {/* Elegant Title */}
              <h1 className="text-4xl font-bold tracking-tight leading-tight">
                <span className="bg-gradient-to-br from-foreground via-foreground/90 to-foreground/70 bg-clip-text text-transparent">
                  {title}
                </span>
              </h1>
              
              {/* Enhanced Description */}
              <CardDescription className="text-lg leading-relaxed text-muted-foreground/90 max-w-3xl">
                {description}
              </CardDescription>
            </div>
            
            {/* Elegant Equation Display */}
            {equation && (
              <div className="hidden lg:block">
                <Card className="bg-background/30 border-border/30 backdrop-blur-sm rounded-2xl p-6 min-w-[250px] shadow-xl">
                  <div className="text-xs text-muted-foreground mb-3 font-medium tracking-wide uppercase">
                    Key Equation
                  </div>
                  <div className="text-base flex items-center justify-center p-2 whitespace-nowrap">
                    <InlineMath math={equation} />
                  </div>
                  <div className="mt-3 flex justify-center">
                    <div className={`w-16 h-0.5 rounded-full ${categoryStyle.color.split(' ')[0]} opacity-60`} />
                  </div>
                </Card>
              </div>
            )}
          </div>
        </CardHeader>
        
        {/* Mobile Equation Display */}
        {equation && (
          <CardContent className="pt-0 lg:hidden relative z-10">
            <Card className="bg-background/30 border-border/30 backdrop-blur-sm rounded-2xl p-4 shadow-lg">
              <div className="text-xs text-muted-foreground mb-2 font-medium tracking-wide uppercase">
                Key Equation
              </div>
              <div className="text-sm flex items-center justify-center p-2">
                <InlineMath math={equation} />
              </div>
            </Card>
          </CardContent>
        )}
        
        {/* Subtle Footer Accent */}
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary/30 to-transparent" />
      </Card>
    </div>
  );
};