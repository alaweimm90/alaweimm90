import { memo, useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Settings, Eye, EyeOff, Zap, Trophy, Target, Star } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNotifications } from '@/components/FloatingNotifications';
import { useKeyboard } from '@/components/KeyboardShortcuts';

interface ShowcaseStep {
  id: string;
  title: string;
  description: string;
  component: string;
  duration: number;
  action: () => void;
  category: 'visual' | 'interactive' | 'gaming' | 'technical';
}

interface ShowcaseDemoProps {
  className?: string;
}

const ShowcaseDemoComponent = ({ className = '' }: ShowcaseDemoProps) => {
  const [isActive, setIsActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [autoMode, setAutoMode] = useState(true);
  const [progress, setProgress] = useState(0);
  const { addNotification } = useNotifications();
  const { addShortcut, removeShortcut } = useKeyboard();
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const stepTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const showcaseSteps: ShowcaseStep[] = [
    {
      id: 'welcome',
      title: 'Strategic Command Center',
      description: 'Welcome to the LLM Works interactive demonstration',
      component: 'Hero Section',
      duration: 3000,
      category: 'visual',
      action: () => {
        addNotification({
          type: 'info',
          title: 'Showcase Demo Started',
          message: 'Demonstrating Strategic Command Center capabilities',
          duration: 3000
        });
      }
    },
    {
      id: 'battle-animation',
      title: 'AI Battle Simulation',
      description: 'Advanced robot combat animations with realistic battle phases',
      component: 'AIBattleAnimation',
      duration: 4000,
      category: 'gaming',
      action: () => {
        addNotification({
          type: 'battle',
          title: 'Battle Initiated!',
          message: 'Claude-4 vs GPT-5 - Strategic Combat Simulation',
          icon: Target,
          duration: 4000
        });
      }
    },
    {
      id: 'xp-system',
      title: 'XP Progression System',
      description: 'Gamified progression with achievements, levels, and rewards',
      component: 'XPProgressionSystem',
      duration: 3500,
      category: 'gaming',
      action: () => {
        addNotification({
          type: 'xp',
          title: '+500 XP Gained!',
          message: 'Level 12 → Level 13 - Strategic Commander Rank',
          icon: Star,
          duration: 3500
        });
      }
    },
    {
      id: 'victory-celebration',
      title: 'Victory Celebration',
      description: 'Spectacular particle effects and celebration animations',
      component: 'VictoryCelebration',
      duration: 4000,
      category: 'gaming',
      action: () => {
        addNotification({
          type: 'achievement',
          title: 'Victory Achieved!',
          message: 'Mission Complete - Outstanding Performance!',
          icon: Trophy,
          duration: 4000
        });
      }
    },
    {
      id: 'leaderboard',
      title: 'Gaming Leaderboard',
      description: '6-tier ranking system with ELO ratings and metallic badges',
      component: 'GamingLeaderboard',
      duration: 3000,
      category: 'gaming',
      action: () => {
        addNotification({
          type: 'success',
          title: 'Leaderboard Updated',
          message: 'Platinum tier achieved - Top 1% ranking',
          duration: 3000
        });
      }
    },
    {
      id: 'live-dashboard',
      title: 'Live Intelligence Dashboard',
      description: 'Real-time monitoring with tactical intelligence feeds',
      component: 'LiveIntelligenceDashboard',
      duration: 3500,
      category: 'technical',
      action: () => {
        addNotification({
          type: 'info',
          title: 'Intel Systems Online',
          message: 'All monitoring dashboards operational',
          duration: 3000
        });
      }
    },
    {
      id: 'command-panel',
      title: 'Strategic Command Panel',
      description: 'Multi-tab control interface with mission parameters',
      component: 'StrategicCommandPanel',
      duration: 3000,
      category: 'technical',
      action: () => {
        addNotification({
          type: 'success',
          title: 'Command Center Active',
          message: 'Strategic operations interface ready',
          duration: 3000
        });
      }
    },
    {
      id: 'dynamic-background',
      title: 'Dynamic Background Effects',
      description: 'Animated floating elements with mouse-responsive interactions',
      component: 'DynamicBackground',
      duration: 3000,
      category: 'visual',
      action: () => {
        addNotification({
          type: 'info',
          title: 'Ambient Systems Active',
          message: 'Dynamic visual effects synchronized',
          duration: 3000
        });
      }
    },
    {
      id: 'magnetic-elements',
      title: 'Magnetic Hover Effects',
      description: 'Advanced micro-interactions with magnetic attraction physics',
      component: 'MagneticElements',
      duration: 3000,
      category: 'interactive',
      action: () => {
        addNotification({
          type: 'info',
          title: 'Interactive Layer Online',
          message: 'Magnetic field interactions enabled',
          duration: 3000
        });
      }
    },
    {
      id: 'keyboard-shortcuts',
      title: 'Keyboard Shortcuts',
      description: 'Comprehensive hotkey system for power users',
      component: 'KeyboardShortcuts',
      duration: 2500,
      category: 'interactive',
      action: () => {
        addNotification({
          type: 'info',
          title: 'Hotkey System Ready',
          message: 'Press ? to view all keyboard shortcuts',
          duration: 4000
        });
      }
    },
    {
      id: 'complete',
      title: 'Showcase Complete',
      description: 'Strategic Command Center demonstration finished',
      component: 'Summary',
      duration: 2000,
      category: 'visual',
      action: () => {
        addNotification({
          type: 'achievement',
          title: 'Demo Complete!',
          message: 'All Strategic Command Center features demonstrated',
          icon: Trophy,
          duration: 5000
        });
      }
    }
  ];

  // Add keyboard shortcuts for demo control
  useEffect(() => {
    if (isActive) {
      addShortcut({
        key: 'space',
        description: 'Play/Pause showcase demo',
        action: () => setIsPaused(prev => !prev),
        category: 'actions'
      });

      addShortcut({
        key: 'r',
        description: 'Restart showcase demo',
        action: restartDemo,
        category: 'actions'
      });

      addShortcut({
        key: 'Escape',
        description: 'Exit showcase demo',
        action: stopDemo,
        category: 'actions'
      });
    }

    return () => {
      removeShortcut('space');
      removeShortcut('r');
      removeShortcut('Escape');
    };
  }, [isActive, addShortcut, removeShortcut]);

  // Auto-progression logic
  useEffect(() => {
    if (!isActive || isPaused || !autoMode) return;

    const currentStepData = showcaseSteps[currentStep];
    if (!currentStepData) return;

    // Execute step action
    currentStepData.action();

    // Progress bar animation
    let progressStart = 0;
    const progressInterval = 50;
    const progressIncrement = (progressInterval / currentStepData.duration) * 100;

    intervalRef.current = setInterval(() => {
      progressStart += progressIncrement;
      setProgress(Math.min(progressStart, 100));
    }, progressInterval);

    // Move to next step
    stepTimeoutRef.current = setTimeout(() => {
      if (currentStep < showcaseSteps.length - 1) {
        setCurrentStep(prev => prev + 1);
        setProgress(0);
      } else {
        stopDemo();
      }
    }, currentStepData.duration);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (stepTimeoutRef.current) clearTimeout(stepTimeoutRef.current);
    };
  }, [isActive, currentStep, isPaused, autoMode]);

  const startDemo = () => {
    setIsActive(true);
    setCurrentStep(0);
    setProgress(0);
    setIsPaused(false);
  };

  const stopDemo = () => {
    setIsActive(false);
    setCurrentStep(0);
    setProgress(0);
    setIsPaused(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (stepTimeoutRef.current) clearTimeout(stepTimeoutRef.current);
  };

  const restartDemo = () => {
    stopDemo();
    setTimeout(startDemo, 100);
  };

  const togglePause = () => {
    setIsPaused(prev => !prev);
  };

  const nextStep = () => {
    if (currentStep < showcaseSteps.length - 1) {
      setCurrentStep(prev => prev + 1);
      setProgress(0);
    }
  };

  const previousStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
      setProgress(0);
    }
  };

  const getCurrentStep = () => showcaseSteps[currentStep];

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'visual': return Eye;
      case 'interactive': return Zap;
      case 'gaming': return Trophy;
      case 'technical': return Settings;
      default: return Star;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'visual': return 'text-blue-400';
      case 'interactive': return 'text-purple-400';
      case 'gaming': return 'text-orange-400';
      case 'technical': return 'text-green-400';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Demo Control Panel */}
      <div className="glass-panel p-6 border border-primary/20">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="glass-subtle p-2 rounded-xl">
              <Play className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="heading-refined text-lg">Interactive Showcase Demo</h3>
              <p className="text-xs text-muted-foreground">
                Experience all Strategic Command Center features
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Auto Mode Toggle */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAutoMode(prev => !prev)}
              className={`glass-minimal border-primary/30 ${autoMode ? 'bg-primary/10' : ''}`}
            >
              {autoMode ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              <span className="hidden sm:inline ml-2">
                {autoMode ? 'Auto' : 'Manual'}
              </span>
            </Button>

            {/* Control Buttons */}
            {!isActive ? (
              <Button onClick={startDemo} className="bg-primary hover:bg-primary/90">
                <Play className="h-4 w-4 mr-2" />
                Start Demo
              </Button>
            ) : (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={togglePause}
                  className="glass-minimal border-primary/30"
                >
                  {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={restartDemo}
                  className="glass-minimal border-primary/30"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={stopDemo}
                  className="glass-minimal border-red-500/30 text-red-400 hover:bg-red-500/10"
                >
                  Stop
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Current Step Display */}
        {isActive && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {(() => {
                  const currentStepData = getCurrentStep();
                  const IconComponent = getCategoryIcon(currentStepData.category);
                  return (
                    <>
                      <div className="glass-subtle p-2 rounded-lg">
                        <IconComponent className={`h-4 w-4 ${getCategoryColor(currentStepData.category)}`} />
                      </div>
                      <div>
                        <h4 className="heading-refined text-sm font-semibold">
                          {currentStepData.title}
                        </h4>
                        <p className="text-xs text-muted-foreground">
                          {currentStepData.description}
                        </p>
                      </div>
                    </>
                  );
                })()}
              </div>

              <div className="text-xs text-muted-foreground">
                Step {currentStep + 1} of {showcaseSteps.length}
              </div>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="glass-minimal h-2 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-75 ease-linear"
                  style={{ width: `${progress}%` }}
                />
              </div>

              {/* Step Navigation */}
              <div className="flex justify-between items-center">
                <div className="flex gap-1">
                  {showcaseSteps.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => setCurrentStep(index)}
                      className={`w-2 h-2 rounded-full transition-colors ${
                        index === currentStep
                          ? 'bg-primary'
                          : index < currentStep
                          ? 'bg-primary/50'
                          : 'bg-muted/30'
                      }`}
                      disabled={autoMode}
                    />
                  ))}
                </div>

                {!autoMode && (
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={previousStep}
                      disabled={currentStep === 0}
                      className="glass-minimal border-primary/30"
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={nextStep}
                      disabled={currentStep === showcaseSteps.length - 1}
                      className="glass-minimal border-primary/30"
                    >
                      Next
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Feature Summary */}
        {!isActive && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            {['visual', 'interactive', 'gaming', 'technical'].map((category) => {
              const count = showcaseSteps.filter(step => step.category === category).length;
              const IconComponent = getCategoryIcon(category);
              
              return (
                <div key={category} className="glass-minimal p-3 rounded-lg text-center">
                  <IconComponent className={`h-5 w-5 mx-auto mb-2 ${getCategoryColor(category)}`} />
                  <div className="text-sm font-medium capitalize">{category}</div>
                  <div className="text-xs text-muted-foreground">{count} features</div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Keyboard Shortcuts Hint */}
      {isActive && (
        <div className="glass-minimal p-3 rounded-lg border border-primary/10">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Settings className="h-3 w-3" />
            <span>Keyboard shortcuts:</span>
            <kbd className="px-1.5 py-0.5 bg-muted/30 border border-muted/50 rounded text-xs">Space</kbd>
            <span>to pause,</span>
            <kbd className="px-1.5 py-0.5 bg-muted/30 border border-muted/50 rounded text-xs">R</kbd>
            <span>to restart,</span>
            <kbd className="px-1.5 py-0.5 bg-muted/30 border border-muted/50 rounded text-xs">Esc</kbd>
            <span>to exit</span>
          </div>
        </div>
      )}
    </div>
  );
};

export const ShowcaseDemo = memo(ShowcaseDemoComponent);