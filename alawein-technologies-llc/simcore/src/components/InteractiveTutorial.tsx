import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  PlayCircle, 
  PauseCircle, 
  SkipForward, 
  SkipBack, 
  X, 
  Check,
  ArrowRight,
  Lightbulb,
  Target,
  BookOpen
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useNavigate } from 'react-router-dom';

interface TutorialStep {
  id: string;
  title: string;
  description: string;
  target?: string;
  action?: () => void;
  completion?: () => boolean;
}

interface InteractiveTutorialProps {
  tutorialId: string;
  steps: TutorialStep[];
  onComplete?: () => void;
  onSkip?: () => void;
  className?: string;
}

export const InteractiveTutorial: React.FC<InteractiveTutorialProps> = ({
  tutorialId,
  steps,
  onComplete,
  onSkip,
  className = ''
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [isVisible, setIsVisible] = useState(true);
  const { toast } = useToast();

  const progress = ((currentStep + 1) / steps.length) * 100;
  const currentTutorialStep = steps[currentStep];

  useEffect(() => {
    // Auto-advance when step is completed
    if (isPlaying && currentTutorialStep?.completion && currentTutorialStep.completion()) {
      const timer = setTimeout(() => {
        handleNext();
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [isPlaying, currentStep, currentTutorialStep]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCompletedSteps(prev => new Set(prev).add(currentStep));
      setCurrentStep(prev => prev + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleComplete = () => {
    setCompletedSteps(prev => new Set(prev).add(currentStep));
    setIsPlaying(false);
    toast({
      title: "Tutorial Complete!",
      description: "Great job! You've completed the tutorial.",
    });
    onComplete?.();
  };

  const handleSkip = () => {
    setIsVisible(false);
    onSkip?.();
    toast({
      title: "Tutorial Skipped",
      description: "You can always restart the tutorial from the help menu.",
    });
  };

  const executeStepAction = () => {
    if (currentTutorialStep?.action) {
      currentTutorialStep.action();
      toast({
        title: "Step Action Executed",
        description: currentTutorialStep.description,
      });
    }
  };

  if (!isVisible) return null;

  return (
    <Card className={`fixed bottom-6 right-6 w-96 z-50 shadow-lg border-primary/20 ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-primary" />
            Interactive Tutorial
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleSkip}
            className="h-6 w-6 p-0"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Step {currentStep + 1} of {steps.length}</span>
            <Badge variant="outline" className="text-xs">
              {Math.round(progress)}% Complete
            </Badge>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Current Step Content */}
        <div className="space-y-3">
          <h3 className="font-semibold flex items-center gap-2">
            <Target className="w-4 h-4 text-primary" />
            {currentTutorialStep?.title}
          </h3>
          
          <p className="text-sm text-muted-foreground leading-relaxed">
            {currentTutorialStep?.description}
          </p>

          {currentTutorialStep?.action && (
            <Button
              variant="outline"
              size="sm"
              onClick={executeStepAction}
              className="w-full"
            >
              <PlayCircle className="w-4 h-4 mr-2" />
              Try It Now
            </Button>
          )}
        </div>

        {/* Navigation Controls */}
        <div className="flex items-center justify-between pt-3 border-t">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePrevious}
              disabled={currentStep === 0}
            >
              <SkipBack className="w-4 h-4" />
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
              className={isPlaying ? 'text-orange-500' : 'text-green-500'}
            >
              {isPlaying ? <PauseCircle className="w-4 h-4" /> : <PlayCircle className="w-4 h-4" />}
            </Button>
          </div>

          <div className="flex items-center gap-2">
            {completedSteps.has(currentStep) && (
              <Check className="w-4 h-4 text-green-500" />
            )}
            
            {currentStep === steps.length - 1 ? (
              <Button
                size="sm"
                onClick={handleComplete}
                className="flex items-center gap-1"
              >
                Complete
                <Check className="w-4 h-4" />
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={handleNext}
                className="flex items-center gap-1"
              >
                Next
                <SkipForward className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Step Indicators */}
        <div className="flex justify-center gap-1 pt-2">
          {steps.map((_, index) => (
            <div
              key={index}
              className={`w-2 h-2 rounded-full transition-colors ${
                index === currentStep
                  ? 'bg-primary'
                  : completedSteps.has(index)
                  ? 'bg-green-500'
                  : 'bg-muted'
              }`}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

// Predefined tutorial configurations
export const platformTutorialSteps: TutorialStep[] = [
  {
    id: 'welcome',
    title: 'Welcome to SimCore',
    description: 'This interactive tutorial will guide you through the key features of our interactive scientific computing platform.'
  },
  {
    id: 'navigation',
    title: 'Navigation Basics',
    description: 'Use the sidebar to navigate between different scientific modules. Each module focuses on a specific area of computational science.'
  },
  {
    id: 'modules',
    title: 'Scientific Modules',
    description: 'Explore our collection of interactive physics simulations. Start with beginner modules and progress to advanced research tools.'
  },
  {
    id: 'documentation',
    title: 'Documentation Hub',
    description: 'Access comprehensive guides, theoretical backgrounds, and mathematical derivations in the Documentation section.',
    action: () => window.location.hash = '#documentation'
  },
  {
    id: 'features',
    title: 'Advanced Features',
    description: 'Discover ML-enhanced simulations, WebGPU acceleration, and collaborative tools for research and education.'
  }
];

export const moduleSpecificTutorial = {
  'graphene': [
    {
      id: 'graphene-intro',
      title: 'Graphene Physics',
      description: 'Learn about the electronic band structure of graphene and its unique properties.'
    },
    {
      id: 'graphene-controls',
      title: 'Parameter Controls',
      description: 'Adjust the hopping parameters and see how they affect the band structure in real-time.'
    },
    {
      id: 'graphene-theory',
      title: 'Theoretical Background',
      description: 'Explore the tight-binding model and the derivation of graphene\'s dispersion relation.'
    }
  ],
  'quantum-tunneling': [
    {
      id: 'tunneling-intro',
      title: 'Quantum Tunneling',
      description: 'Understand how quantum particles can pass through potential barriers.'
    },
    {
      id: 'tunneling-wave',
      title: 'Wave Function',
      description: 'Observe how the quantum wave function behaves near and through the barrier.'
    },
    {
      id: 'tunneling-probability',
      title: 'Tunneling Probability',
      description: 'See how the tunneling probability depends on barrier height and width.'
    }
  ]
};