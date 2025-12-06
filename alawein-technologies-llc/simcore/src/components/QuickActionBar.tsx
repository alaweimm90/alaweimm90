import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { 
  Zap, 
  BookOpen, 
  Settings, 
  Share2, 
  Download, 
  Moon, 
  Sun,
  Bell,
  HelpCircle,
  Gamepad2,
  Calculator,
  Sparkles
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from 'next-themes';
import { useToast } from '@/hooks/use-toast';

interface QuickActionBarProps {
  className?: string;
}

export const QuickActionBar: React.FC<QuickActionBarProps> = ({ 
  className = '' 
}) => {
  const navigate = useNavigate();
  const { theme, setTheme } = useTheme();
  const { toast } = useToast();
  const [isSharing, setIsSharing] = useState(false);

  const quickActions = [
    {
      id: 'documentation',
      icon: BookOpen,
      label: 'Documentation',
      description: 'Browse comprehensive guides',
      action: () => navigate('/documentation'),
      color: 'bg-blue-500/10 text-blue-500 hover:bg-blue-500/20'
    },
    {
      id: 'simulation-dashboard',
      icon: Calculator,
      label: 'Simulations',
      description: 'Access simulation dashboard',
      action: () => navigate('/simulation-dashboard'),
      color: 'bg-green-500/10 text-green-500 hover:bg-green-500/20'
    },
    {
      id: 'performance',
      icon: Zap,
      label: 'Performance',
      description: 'Monitor system performance',
      action: () => navigate('/performance-monitor'),
      color: 'bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20'
    }
  ];

  const utilityActions = [
    {
      id: 'theme-toggle',
      icon: theme === 'dark' ? Sun : Moon,
      label: theme === 'dark' ? 'Light Mode' : 'Dark Mode',
      action: () => {
        const newTheme = theme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
        toast({
          title: `Switched to ${newTheme} mode`,
          description: "Theme preference saved automatically.",
        });
      }
    },
    {
      id: 'share',
      icon: Share2,
      label: 'Share',
      action: async () => {
        setIsSharing(true);
        try {
          if (navigator.share) {
            await navigator.share({
              title: 'SimCore - Interactive Physics Laboratory',
              text: 'Explore cutting-edge scientific simulations in your browser',
              url: window.location.origin
            });
          } else {
            await navigator.clipboard.writeText(window.location.origin);
            toast({
              title: "Link copied!",
              description: "SimCore URL has been copied to your clipboard.",
            });
          }
        } catch (error) {
          console.log('Share cancelled or failed');
        } finally {
          setIsSharing(false);
        }
      }
    },
    {
      id: 'help',
      icon: HelpCircle,
      label: 'Help',
      action: () => {
        toast({
          title: "Need help?",
          description: "Check the Documentation section for comprehensive guides and tutorials.",
          duration: 5000,
        });
        navigate('/documentation');
      }
    }
  ];

  const featuredModules = [
    {
      name: 'Graphene Band Structure',
      route: '/modules/graphene-band-structure',
      badge: 'Popular'
    },
    {
      name: 'Quantum Tunneling',
      route: '/modules/quantum-tunneling',
      badge: 'Interactive'
    },
    {
      name: 'MoS₂ Valley Physics',
      route: '/modules/mos2-valley-physics',
      badge: 'Advanced'
    }
  ];

  return (
    <Card className={`w-full ${className}`}>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Quick Actions */}
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              Quick Actions
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-[--spacing-md]">
              {quickActions.map((action) => (
                <Button
                  key={action.id}
                  variant="nav"
                  className={`p-[--spacing-md] h-auto flex flex-col items-center gap-[--spacing-xs] hover:scale-105 transition-all duration-200 min-h-[--touch-target-comfortable] ${action.color}`}
                  onClick={action.action}
                >
                  <action.icon className="w-6 h-6" />
                  <div className="text-center">
                    <div className="font-medium text-sm">{action.label}</div>
                    <div className="text-xs text-muted-foreground">{action.description}</div>
                  </div>
                </Button>
              ))}
            </div>
          </div>

          {/* Utility Actions */}
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-primary" />
              Utilities
            </h3>
            <div className="flex flex-wrap gap-[--spacing-xs]">
              {utilityActions.map((action) => (
                <Button
                  key={action.id}
                  variant="nav"
                  size="sm"
                  className="flex items-center gap-[--spacing-xs] hover:scale-105 transition-all duration-200 min-h-[--touch-target-min]"
                  onClick={action.action}
                  disabled={action.id === 'share' && isSharing}
                >
                  <action.icon className="w-4 h-4" />
                  {action.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Featured Modules */}
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Gamepad2 className="w-5 h-5 text-primary" />
              Featured Modules
            </h3>
            <div className="flex flex-wrap gap-[--spacing-xs]">
              {featuredModules.map((module) => (
                <Button
                  key={module.route}
                  variant="nav"
                  size="sm"
                  className="flex items-center gap-[--spacing-xs] hover:scale-105 transition-all duration-200 min-h-[--touch-target-min]"
                  onClick={() => navigate(module.route)}
                >
                  {module.name}
                  <Badge variant="secondary" className="ml-1 text-xs">
                    {module.badge}
                  </Badge>
                </Button>
              ))}
            </div>
          </div>

          {/* Status Indicator */}
          <div className="flex items-center justify-between pt-4 border-t">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              All systems operational
            </div>
            <div className="text-xs text-muted-foreground">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};