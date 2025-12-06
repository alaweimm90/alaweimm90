import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { 
  Zap, 
  Settings, 
  HelpCircle, 
  Share2, 
  Download,
  Palette,
  Play,
  ChevronUp,
  MousePointer2,
  Sparkles
} from 'lucide-react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { useToast } from '@/hooks/use-toast';

interface FloatingActionButtonProps {
  className?: string;
}

interface ActionItem {
  id: string;
  icon: React.ElementType;
  label: string;
  action: () => void;
  color?: string;
  shortcut?: string;
}

export const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const prefersReducedMotion = usePrefersReducedMotion();
  const { toast } = useToast();

  // Track mouse movement for magnetic effect
  useEffect(() => {
    if (prefersReducedMotion) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const deltaX = e.clientX - centerX;
        const deltaY = e.clientY - centerY;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        
        // Create magnetic effect within 100px radius
        if (distance < 100) {
          const strength = Math.max(0, 1 - distance / 100);
          setMousePosition({
            x: deltaX * strength * 0.1,
            y: deltaY * strength * 0.1
          });
        } else {
          setMousePosition({ x: 0, y: 0 });
        }
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [prefersReducedMotion]);

  const actions: ActionItem[] = [
    {
      id: 'quickstart',
      icon: Play,
      label: 'Quick Start Tutorial',
      action: () => {
        localStorage.removeItem('simcore-tutorial-completed');
        window.location.reload();
      },
      color: 'text-green-500',
      shortcut: 'T'
    },
    {
      id: 'settings',
      icon: Settings,
      label: 'Preferences',
      action: () => {
        // This would open settings
        toast({
          title: "Settings",
          description: "Use the sidebar settings button for full preferences.",
        });
      },
      color: 'text-blue-500',
      shortcut: 'S'
    },
    {
      id: 'help',
      icon: HelpCircle,
      label: 'Help & Documentation',
      action: () => {
        window.location.href = '/documentation';
      },
      color: 'text-purple-500',
      shortcut: 'H'
    },
    {
      id: 'share',
      icon: Share2,
      label: 'Share Platform',
      action: async () => {
        try {
          if (navigator.share) {
            await navigator.share({
              title: 'SimCore - Interactive Physics Laboratory',
              text: 'Explore cutting-edge scientific simulations',
              url: window.location.origin
            });
          } else {
            await navigator.clipboard.writeText(window.location.origin);
            toast({
              title: "Link Copied!",
              description: "SimCore URL has been copied to clipboard.",
            });
          }
        } catch (error) {
          console.log('Share cancelled');
        }
      },
      color: 'text-orange-500',
      shortcut: 'Shift+S'
    }
  ];

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        const action = actions.find(a => {
          if (a.shortcut?.includes('Shift') && e.shiftKey) {
            return a.shortcut.split('+')[1].toLowerCase() === e.key.toLowerCase();
          }
          return a.shortcut?.toLowerCase() === e.key.toLowerCase();
        });
        
        if (action) {
          e.preventDefault();
          action.action();
        }
      }
    };

    window.addEventListener('keydown', handleKeyboard);
    return () => window.removeEventListener('keydown', handleKeyboard);
  }, [actions]);

  const handleMainButtonClick = () => {
    setIsExpanded(!isExpanded);
  };

  const getButtonStyle = () => {
    if (prefersReducedMotion) {
      return {};
    }

    return {
      transform: `translate(${mousePosition.x}px, ${mousePosition.y}px) ${isHovered ? 'scale(1.1)' : 'scale(1)'}`,
      transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    };
  };

  const getActionStyle = (index: number) => {
    if (prefersReducedMotion || !isExpanded) {
      return {
        opacity: isExpanded ? 1 : 0,
        transform: isExpanded ? 'scale(1)' : 'scale(0)',
        transition: 'all 0.2s ease-out',
        transitionDelay: isExpanded ? `${index * 50}ms` : '0ms'
      };
    }

    const angle = (index * 70) - 35; // Spread actions in an arc
    const radius = 80;
    const x = Math.cos((angle * Math.PI) / 180) * radius;
    const y = Math.sin((angle * Math.PI) / 180) * radius;

    return {
      opacity: isExpanded ? 1 : 0,
      transform: isExpanded 
        ? `translate(${x}px, ${y}px) scale(1)` 
        : 'translate(0px, 0px) scale(0)',
      transition: 'all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      transitionDelay: isExpanded ? `${index * 100}ms` : '0ms'
    };
  };

  return (
    <TooltipProvider>
      <div 
        ref={containerRef}
        className={`fixed bottom-6 right-6 z-50 ${className}`}
      >
        {/* Action Buttons */}
        {actions.map((action, index) => (
          <div
            key={action.id}
            className="absolute bottom-0 right-0"
            style={getActionStyle(index)}
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  variant="outline"
                  className={`w-10 h-10 rounded-full shadow-lg backdrop-blur-sm bg-background/80 border-border/50 hover:shadow-xl transition-all duration-200 ${action.color}`}
                  onClick={action.action}
                >
                  <action.icon className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left" className="flex items-center gap-2">
                <span>{action.label}</span>
                {action.shortcut && (
                  <kbd className="px-1.5 py-0.5 text-xs bg-muted rounded">
                    {action.shortcut}
                  </kbd>
                )}
              </TooltipContent>
            </Tooltip>
          </div>
        ))}

        {/* Main FAB */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="lg"
              className="w-14 h-14 rounded-full shadow-xl bg-gradient-to-r from-primary to-accent hover:shadow-2xl relative overflow-hidden group"
              style={getButtonStyle()}
              onClick={handleMainButtonClick}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              {/* Animated background */}
              {!prefersReducedMotion && (
                <div className="absolute inset-0 bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] animate-gradient-x opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              )}
              
              {/* Icon with rotation */}
              <div className="relative z-10">
                {isExpanded ? (
                  <ChevronUp 
                    className={`w-6 h-6 text-primary-foreground transition-transform duration-300 ${
                      !prefersReducedMotion ? 'rotate-180' : ''
                    }`} 
                  />
                ) : (
                  <Sparkles 
                    className={`w-6 h-6 text-primary-foreground transition-all duration-300 ${
                      !prefersReducedMotion && isHovered ? 'rotate-12 scale-110' : ''
                    }`} 
                  />
                )}
              </div>

              {/* Ripple effect */}
              {!prefersReducedMotion && isHovered && (
                <div className="absolute inset-0 rounded-full bg-[hsl(var(--semantic-text-primary))]/20 animate-ping" />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="left">
            <span>{isExpanded ? 'Close Quick Actions' : 'Quick Actions Menu'}</span>
          </TooltipContent>
        </Tooltip>

        {/* Backdrop blur overlay when expanded */}
        {isExpanded && !prefersReducedMotion && (
          <div 
            className="fixed inset-0 bg-background/10 backdrop-blur-[1px] -z-10 transition-opacity duration-300"
            onClick={() => setIsExpanded(false)}
          />
        )}
      </div>
    </TooltipProvider>
  );
};