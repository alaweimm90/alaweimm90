import { memo, useState, useEffect, useRef, ReactNode } from 'react';
import { X, ChevronLeft, ChevronRight, Play, SkipForward, MapPin, Compass, Target, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useKeyboard } from '@/components/KeyboardShortcuts';

interface TourStep {
  id: string;
  title: string;
  description: string;
  target: string; // CSS selector for the target element
  position: 'top' | 'bottom' | 'left' | 'right' | 'center';
  category: 'basic' | 'advanced' | 'gaming' | 'professional';
  action?: () => void;
  waitForElement?: boolean; // Wait for element to appear
}

interface FeatureTourProps {
  className?: string;
  autoStart?: boolean;
  tourType?: 'quick' | 'comprehensive' | 'gaming' | 'professional';
}

const FeatureTourComponent = ({ 
  className = '', 
  autoStart = false, 
  tourType = 'comprehensive' 
}: FeatureTourProps) => {
  const [isActive, setIsActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [tourSteps, setTourSteps] = useState<TourStep[]>([]);
  const [highlightPosition, setHighlightPosition] = useState({ x: 0, y: 0, width: 0, height: 0 });
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const { addShortcut, removeShortcut } = useKeyboard();
  const observerRef = useRef<MutationObserver | null>(null);

  const allTourSteps: TourStep[] = [
    // Basic Navigation
    {
      id: 'welcome',
      title: 'Welcome to Strategic Command Center',
      description: 'This tour will guide you through all the powerful features of the LLM Works platform.',
      target: 'main',
      position: 'center',
      category: 'basic'
    },
    {
      id: 'navigation',
      title: 'Main Navigation',
      description: 'Access different areas: Arena for battles, Bench for benchmarks, and strategic dashboards.',
      target: 'nav',
      position: 'bottom',
      category: 'basic'
    },
    {
      id: 'showcase-demo',
      title: 'Interactive Showcase Demo',
      description: 'Experience all features in an automated demonstration. Perfect for portfolio presentations!',
      target: '[aria-labelledby="showcase-section"]',
      position: 'top',
      category: 'professional'
    },
    {
      id: 'hero-battle',
      title: 'AI Battle Animation',
      description: 'Watch advanced robot combat simulations with realistic battle phases and tactical maneuvers.',
      target: '[data-tour="battle-animation"]',
      position: 'bottom',
      category: 'gaming',
      waitForElement: true
    },
    {
      id: 'live-dashboard',
      title: 'Live Intelligence Dashboard',
      description: 'Monitor real-time strategic evaluations, system performance, and threat intelligence.',
      target: '[aria-labelledby="dashboard-section"]',
      position: 'top',
      category: 'professional'
    },
    {
      id: 'command-panel',
      title: 'Strategic Command Panel',
      description: 'Advanced control interface with module management, session monitoring, and terminal access.',
      target: '[aria-labelledby="command-section"]',
      position: 'top',
      category: 'advanced'
    },
    {
      id: 'xp-progression',
      title: 'XP Progression System',
      description: 'Level up through commander ranks, unlock achievements, and earn XP through evaluations.',
      target: '[aria-labelledby="progression-section"]',
      position: 'top',
      category: 'gaming'
    },
    {
      id: 'leaderboards',
      title: 'Elite Rankings',
      description: 'Compete in tiered rankings from Bronze to Legendary status with ELO ratings.',
      target: '[aria-labelledby="rankings-section"]',
      position: 'top',
      category: 'gaming'
    },
    {
      id: 'technical-specs',
      title: 'Technical Documentation',
      description: 'Comprehensive specs including architecture, APIs, security framework, and performance metrics.',
      target: '[aria-labelledby="specs-section"]',
      position: 'top',
      category: 'professional'
    },
    {
      id: 'floating-notifications',
      title: 'Smart Notifications',
      description: 'Real-time achievements, battle results, and system updates with particle effects.',
      target: '.fixed.top-20.right-4',
      position: 'left',
      category: 'advanced',
      waitForElement: true
    },
    {
      id: 'keyboard-shortcuts',
      title: 'Keyboard Shortcuts',
      description: 'Press ? to view all hotkeys. Navigate efficiently with powerful shortcuts for every feature.',
      target: '.fixed.bottom-4.right-4',
      position: 'left',
      category: 'advanced'
    },
    {
      id: 'dynamic-background',
      title: 'Dynamic Visual Effects',
      description: 'Interactive background elements respond to mouse movement with sophisticated animations.',
      target: 'body',
      position: 'center',
      category: 'advanced'
    },
    {
      id: 'complete',
      title: 'Tour Complete!',
      description: 'You\'ve explored all the Strategic Command Center features. Ready to begin your AI evaluation missions?',
      target: 'main',
      position: 'center',
      category: 'basic'
    }
  ];

  const filterStepsForTourType = (type: string): TourStep[] => {
    switch (type) {
      case 'quick':
        return allTourSteps.filter(step => step.category === 'basic');
      case 'gaming':
        return allTourSteps.filter(step => ['basic', 'gaming'].includes(step.category));
      case 'professional':
        return allTourSteps.filter(step => ['basic', 'professional', 'advanced'].includes(step.category));
      case 'comprehensive':
      default:
        return allTourSteps;
    }
  };

  useEffect(() => {
    setTourSteps(filterStepsForTourType(tourType));
  }, [tourType]);

  useEffect(() => {
    if (autoStart && !isActive && !isCompleted) {
      setTimeout(() => startTour(), 2000);
    }
  }, [autoStart, isActive, isCompleted]);

  // Add keyboard shortcuts for tour control
  useEffect(() => {
    if (isActive) {
      addShortcut({
        key: 'ArrowRight',
        description: 'Next tour step',
        action: nextStep,
        category: 'ui'
      });

      addShortcut({
        key: 'ArrowLeft',
        description: 'Previous tour step',
        action: previousStep,
        category: 'ui'
      });

      addShortcut({
        key: 'Escape',
        description: 'Exit feature tour',
        action: endTour,
        category: 'ui'
      });
    }

    return () => {
      removeShortcut('ArrowRight');
      removeShortcut('ArrowLeft');
      removeShortcut('Escape');
    };
  }, [isActive, addShortcut, removeShortcut]);

  const updateHighlight = (targetSelector: string) => {
    const target = document.querySelector(targetSelector);
    if (!target) {
      console.warn(`Tour target not found: ${targetSelector}`);
      return;
    }

    const rect = target.getBoundingClientRect();
    const scrollX = window.pageXOffset;
    const scrollY = window.pageYOffset;

    setHighlightPosition({
      x: rect.left + scrollX,
      y: rect.top + scrollY,
      width: rect.width,
      height: rect.height
    });

    // Calculate tooltip position
    const step = tourSteps[currentStep];
    let tooltipX = rect.left + scrollX;
    let tooltipY = rect.top + scrollY;

    switch (step.position) {
      case 'top':
        tooltipX = rect.left + scrollX + rect.width / 2;
        tooltipY = rect.top + scrollY - 20;
        break;
      case 'bottom':
        tooltipX = rect.left + scrollX + rect.width / 2;
        tooltipY = rect.bottom + scrollY + 20;
        break;
      case 'left':
        tooltipX = rect.left + scrollX - 20;
        tooltipY = rect.top + scrollY + rect.height / 2;
        break;
      case 'right':
        tooltipX = rect.right + scrollX + 20;
        tooltipY = rect.top + scrollY + rect.height / 2;
        break;
      case 'center':
        tooltipX = window.innerWidth / 2;
        tooltipY = window.innerHeight / 2;
        break;
    }

    setTooltipPosition({ x: tooltipX, y: tooltipY });

    // Scroll element into view
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const waitForElement = (selector: string): Promise<Element> => {
    return new Promise((resolve) => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
        return;
      }

      observerRef.current = new MutationObserver((mutations, observer) => {
        const element = document.querySelector(selector);
        if (element) {
          observer.disconnect();
          resolve(element);
        }
      });

      observerRef.current.observe(document.body, {
        childList: true,
        subtree: true
      });
    });
  };

  const executeCurrentStep = async () => {
    const step = tourSteps[currentStep];
    if (!step) return;

    // Execute step action if any
    if (step.action) {
      step.action();
    }

    // Wait for element if needed
    if (step.waitForElement) {
      await waitForElement(step.target);
      // Small delay to ensure element is rendered
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    updateHighlight(step.target);
  };

  useEffect(() => {
    if (isActive && tourSteps.length > 0) {
      executeCurrentStep();
    }
  }, [isActive, currentStep, tourSteps]);

  const startTour = () => {
    setIsActive(true);
    setCurrentStep(0);
    setIsCompleted(false);
  };

  const endTour = () => {
    setIsActive(false);
    setIsCompleted(true);
    if (observerRef.current) {
      observerRef.current.disconnect();
    }
  };

  const nextStep = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      endTour();
    }
  };

  const previousStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const skipTour = () => {
    endTour();
  };

  const getCurrentStep = () => tourSteps[currentStep];

  if (!isActive) {
    return (
      <div className={`space-y-4 ${className}`}>
        {/* Tour Control Panel */}
        <div className="glass-panel p-6 border border-primary/20">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="glass-subtle p-2 rounded-xl">
                <Compass className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h3 className="heading-refined text-lg">Feature Tour Guide</h3>
                <p className="text-xs text-muted-foreground">
                  Discover all Strategic Command Center capabilities
                </p>
              </div>
            </div>

            <Button onClick={startTour} className="bg-primary hover:bg-primary/90">
              <Play className="h-4 w-4 mr-2" />
              Start Tour
            </Button>
          </div>

          {/* Tour Type Selection */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { type: 'quick', label: 'Quick Tour', description: '3 min overview', icon: Target },
              { type: 'gaming', label: 'Gaming Features', description: 'XP, battles, rankings', icon: Target },
              { type: 'professional', label: 'Professional', description: 'Enterprise features', icon: Target },
              { type: 'comprehensive', label: 'Complete Tour', description: 'All features', icon: Compass }
            ].map(({ type, label, description, icon: Icon }) => (
              <button
                key={type}
                onClick={() => setTourSteps(filterStepsForTourType(type))}
                className={`glass-minimal p-4 rounded-lg text-left hover:bg-muted/10 transition-colors ${
                  tourSteps === filterStepsForTourType(type) ? 'border border-primary/30' : ''
                }`}
              >
                <Icon className="h-5 w-5 text-primary mb-2" />
                <div className="text-sm font-medium">{label}</div>
                <div className="text-xs text-muted-foreground">{description}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const currentStepData = getCurrentStep();
  if (!currentStepData) return null;

  return (
    <>
      {/* Dark Overlay */}
      <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 transition-opacity duration-300" />

      {/* Highlight Spotlight */}
      <div
        className="fixed z-50 pointer-events-none transition-all duration-500 ease-out"
        style={{
          left: highlightPosition.x - 8,
          top: highlightPosition.y - 8,
          width: highlightPosition.width + 16,
          height: highlightPosition.height + 16,
          boxShadow: `
            0 0 0 4px hsl(var(--primary) / 0.8),
            0 0 0 8px hsl(var(--primary) / 0.4),
            0 0 40px hsl(var(--primary) / 0.6),
            0 0 0 9999px rgba(0, 0, 0, 0.7)
          `,
          borderRadius: '8px'
        }}
      />

      {/* Tour Tooltip */}
      <div
        className="fixed z-50 transition-all duration-500 ease-out"
        style={{
          left: tooltipPosition.x,
          top: tooltipPosition.y,
          transform: currentStepData.position === 'center' 
            ? 'translate(-50%, -50%)' 
            : currentStepData.position === 'top' 
            ? 'translate(-50%, -100%)' 
            : currentStepData.position === 'bottom'
            ? 'translate(-50%, 0)'
            : currentStepData.position === 'left'
            ? 'translate(-100%, -50%)'
            : 'translate(0, -50%)'
        }}
      >
        <div className="glass-panel max-w-sm border border-primary/30 shadow-xl">
          <div className="p-6">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-2">
                <MapPin className="h-4 w-4 text-primary" />
                <div className="text-xs text-muted-foreground">
                  Step {currentStep + 1} of {tourSteps.length}
                </div>
              </div>
              <button
                onClick={endTour}
                className="glass-minimal p-1 rounded-lg hover:bg-muted/20 transition-colors"
              >
                <X className="h-4 w-4 text-muted-foreground" />
              </button>
            </div>

            {/* Content */}
            <div className="mb-6">
              <h3 className="heading-refined text-lg mb-2">{currentStepData.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {currentStepData.description}
              </p>
            </div>

            {/* Progress Bar */}
            <div className="mb-4">
              <div className="glass-minimal h-2 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-300"
                  style={{ width: `${((currentStep + 1) / tourSteps.length) * 100}%` }}
                />
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center justify-between">
              <Button
                variant="outline"
                size="sm"
                onClick={previousStep}
                disabled={currentStep === 0}
                className="glass-minimal border-primary/30"
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>

              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={skipTour}
                  className="glass-minimal border-muted/30 text-muted-foreground"
                >
                  <SkipForward className="h-4 w-4 mr-1" />
                  Skip
                </Button>

                <Button
                  size="sm"
                  onClick={nextStep}
                  className="bg-primary hover:bg-primary/90"
                >
                  {currentStep === tourSteps.length - 1 ? (
                    <>
                      <CheckCircle className="h-4 w-4 mr-1" />
                      Finish
                    </>
                  ) : (
                    <>
                      Next
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Tooltip Arrow */}
        {currentStepData.position !== 'center' && (
          <div
            className={`absolute w-3 h-3 bg-background border transform rotate-45 ${
              currentStepData.position === 'top' 
                ? 'top-full left-1/2 -translate-x-1/2 -translate-y-1/2 border-t-0 border-l-0' 
                : currentStepData.position === 'bottom'
                ? 'bottom-full left-1/2 -translate-x-1/2 translate-y-1/2 border-b-0 border-r-0'
                : currentStepData.position === 'left'
                ? 'left-full top-1/2 -translate-x-1/2 -translate-y-1/2 border-l-0 border-b-0'
                : 'right-full top-1/2 translate-x-1/2 -translate-y-1/2 border-r-0 border-t-0'
            }`}
            style={{ borderColor: 'hsl(var(--primary) / 0.3)' }}
          />
        )}
      </div>

      {/* Keyboard Shortcuts Hint */}
      <div className="fixed bottom-4 left-4 glass-minimal px-4 py-2 rounded-lg z-50">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Use</span>
          <kbd className="px-1.5 py-0.5 bg-muted/30 border border-muted/50 rounded text-xs">←</kbd>
          <kbd className="px-1.5 py-0.5 bg-muted/30 border border-muted/50 rounded text-xs">→</kbd>
          <span>to navigate,</span>
          <kbd className="px-1.5 py-0.5 bg-muted/30 border border-muted/50 rounded text-xs">Esc</kbd>
          <span>to exit</span>
        </div>
      </div>
    </>
  );
};

export const FeatureTour = memo(FeatureTourComponent);