import React from 'react';
import { CheckCircle, Circle, ChevronRight, Clock, AlertCircle } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

interface EnhancedProgressIndicatorProps {
  currentStep: number;
  totalSteps: number;
  completedSteps: number[];
  tierColors?: {
    primary: string;
    background: string;
    text: string;
    accent: string;
    border: string;
    badge: string;
  };
  onStepClick?: (step: number) => void;
  showEstimatedTime?: boolean;
  timePerStep?: number; // minutes
  startTime?: Date;
}

export const EnhancedProgressIndicator: React.FC<EnhancedProgressIndicatorProps> = ({
  currentStep,
  totalSteps,
  completedSteps,
  tierColors,
  onStepClick,
  showEstimatedTime = true,
  timePerStep = 3,
  startTime
}) => {
  const steps = [
    { name: 'Account', description: 'Basic information', estimate: 2 },
    { name: 'Personal', description: 'Your details', estimate: 4 },
    { name: 'Health', description: 'Medical history', estimate: 5 },
    { name: 'Training', description: 'Fitness goals', estimate: 4 },
    { name: 'Nutrition', description: 'Diet preferences', estimate: 3 },
    { name: 'Tier Goals', description: 'Specific objectives', estimate: 3 },
    { name: 'Payment', description: 'Complete setup', estimate: 2 }
  ];

  const getStepStatus = (stepNumber: number) => {
    if (completedSteps.includes(stepNumber)) return 'completed';
    if (stepNumber === currentStep) return 'current';
    if (stepNumber < currentStep) return 'available';
    return 'pending';
  };

  const calculateProgress = () => {
    const totalCompleted = completedSteps.length;
    const currentProgress = currentStep > totalCompleted ? 0.5 : 0; // Partial progress for current step
    return ((totalCompleted + currentProgress) / totalSteps) * 100;
  };

  const calculateEstimatedTime = () => {
    const remainingSteps = steps.slice(currentStep - 1);
    const totalMinutes = remainingSteps.reduce((acc, step) => acc + step.estimate, 0);
    return totalMinutes;
  };

  const formatElapsedTime = () => {
    if (!startTime) return null;
    const elapsed = Math.floor((Date.now() - startTime.getTime()) / 60000);
    return elapsed;
  };

  const progressValue = calculateProgress();
  const estimatedTimeRemaining = calculateEstimatedTime();
  const elapsedTime = formatElapsedTime();

  return (
    <div className="enhanced-progress-indicator">
      {/* Header with Progress Bar and Time */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <h3 className="text-lg font-semibold">Onboarding Progress</h3>
            <Badge 
              variant="outline" 
              className="px-2 py-1"
              style={{ borderColor: tierColors?.primary }}
            >
              {currentStep} of {totalSteps}
            </Badge>
          </div>
          
          {showEstimatedTime && (
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              {elapsedTime !== null && (
                <div className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  <span>{elapsedTime}m elapsed</span>
                </div>
              )}
              <div className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                <span>~{estimatedTimeRemaining}m remaining</span>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <Progress 
            value={progressValue} 
            className="h-3 w-full"
            style={{
              '--progress-background': tierColors?.primary || '#ea580c'
            } as React.CSSProperties}
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{Math.round(progressValue)}% Complete</span>
            <span>{totalSteps - completedSteps.length} steps remaining</span>
          </div>
        </div>
      </div>

      {/* Desktop Step Navigation */}
      <div className="hidden lg:block">
        <div className="grid grid-cols-7 gap-2 mb-6">
          {steps.map((step, index) => {
            const stepNumber = index + 1;
            const status = getStepStatus(stepNumber);
            const isClickable = onStepClick && (status === 'completed' || status === 'current');
            
            return (
              <Tooltip key={stepNumber}>
                <TooltipTrigger asChild>
                  <div
                    className={`
                      relative p-3 rounded-lg border-2 transition-all duration-200 cursor-pointer
                      ${status === 'completed' 
                        ? 'border-green-500 bg-green-50' 
                        : status === 'current'
                        ? 'border-primary bg-primary/5'
                        : status === 'available'
                        ? 'border-blue-400 bg-blue-50'
                        : 'border-gray-200 bg-gray-50'
                      }
                      ${isClickable ? 'hover:shadow-md hover:scale-105' : 'cursor-default'}
                    `}
                    style={status === 'current' ? {
                      borderColor: tierColors?.primary,
                      backgroundColor: `${tierColors?.primary}10`
                    } : {}}
                    onClick={() => isClickable && onStepClick?.(stepNumber)}
                  >
                    {/* Status Icon */}
                    <div className="flex items-center justify-center mb-2">
                      {status === 'completed' ? (
                        <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                          <CheckCircle className="w-5 h-5 text-white" />
                        </div>
                      ) : status === 'current' ? (
                        <div 
                          className="w-8 h-8 rounded-full flex items-center justify-center text-white"
                          style={{ backgroundColor: tierColors?.primary }}
                        >
                          <span className="text-sm font-bold">{stepNumber}</span>
                        </div>
                      ) : status === 'available' ? (
                        <div className="w-8 h-8 rounded-full bg-blue-400 flex items-center justify-center">
                          <Circle className="w-5 h-5 text-white" />
                        </div>
                      ) : (
                        <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                          <span className="text-sm text-gray-500">{stepNumber}</span>
                        </div>
                      )}
                    </div>

                    {/* Step Info */}
                    <div className="text-center">
                      <div className="text-xs font-medium mb-1">{step.name}</div>
                      <div className="text-xs text-muted-foreground">{step.description}</div>
                      <div className="text-xs text-muted-foreground mt-1">~{step.estimate}min</div>
                    </div>

                    {/* Current Step Indicator */}
                    {status === 'current' && (
                      <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2">
                        <div 
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: tierColors?.primary }}
                        />
                      </div>
                    )}
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="text-center">
                    <div className="font-medium">{step.name}</div>
                    <div className="text-xs text-muted-foreground">{step.description}</div>
                    <div className="text-xs">Estimated: {step.estimate} minutes</div>
                    {status === 'completed' && (
                      <div className="text-xs text-green-600 mt-1">✓ Completed</div>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </div>

      {/* Mobile Step Navigation */}
      <div className="lg:hidden">
        <div className="flex items-center justify-between mb-4">
          <Button
            variant="ghost"
            size="sm"
            disabled={currentStep === 1}
            onClick={() => onStepClick?.(currentStep - 1)}
            className="flex items-center gap-1"
          >
            <ChevronRight className="h-4 w-4 rotate-180" />
            Previous
          </Button>

          <div className="flex items-center gap-2">
            {steps.slice(Math.max(0, currentStep - 2), currentStep + 1).map((step, index) => {
              const stepNumber = currentStep - 1 + index;
              const status = getStepStatus(stepNumber);
              
              return (
                <div
                  key={stepNumber}
                  className={`
                    w-3 h-3 rounded-full transition-all duration-200
                    ${status === 'completed' 
                      ? 'bg-green-500' 
                      : status === 'current'
                      ? 'bg-primary ring-2 ring-primary/30'
                      : 'bg-gray-300'
                    }
                  `}
                  style={status === 'current' ? {
                    backgroundColor: tierColors?.primary,
                    '--tw-ring-color': `${tierColors?.primary}30`
                  } as React.CSSProperties : {}}
                />
              );
            })}
          </div>

          <Button
            variant="ghost"
            size="sm"
            disabled={currentStep === totalSteps}
            onClick={() => onStepClick?.(currentStep + 1)}
            className="flex items-center gap-1"
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {/* Current Step Info */}
        <div className="text-center p-4 bg-muted/50 rounded-lg">
          <div className="font-medium">{steps[currentStep - 1]?.name}</div>
          <div className="text-sm text-muted-foreground">{steps[currentStep - 1]?.description}</div>
          <div className="text-xs text-muted-foreground mt-1">
            Estimated time: ~{steps[currentStep - 1]?.estimate} minutes
          </div>
        </div>
      </div>

      {/* Smart Suggestions */}
      {currentStep > 1 && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <div className="font-medium text-blue-900">Pro Tip</div>
              <div className="text-blue-700">
                {currentStep === 2 && "Take your time with personal details - accurate info helps us create better plans."}
                {currentStep === 3 && "Be thorough with health information - it ensures your safety and program effectiveness."}
                {currentStep === 4 && "Clear training goals help us design the perfect workout plan for you."}
                {currentStep === 5 && "Honest nutrition habits help us create realistic meal recommendations."}
                {currentStep === 6 && "Tier-specific goals unlock advanced features tailored to your ambitions."}
                {currentStep === 7 && "You're almost done! Secure payment unlocks your personalized coaching journey."}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};