import { useMemo, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, Clock, AlertCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProgressStep {
  id: string;
  label: string;
  description?: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress?: number;
}

interface ProgressIndicatorProps {
  steps: ProgressStep[];
  currentStep?: string;
  className?: string;
}

export function ProgressIndicator({ steps, currentStep, className }: ProgressIndicatorProps) {
  const currentStepIndex = useMemo(() => {
    return currentStep ? steps.findIndex(step => step.id === currentStep) : -1;
  }, [steps, currentStep]);

  const overallProgress = useMemo(() => {
    const completedSteps = steps.filter(step => step.status === 'completed').length;
    const currentStepProgress = currentStep ? 
      (steps.find(step => step.id === currentStep)?.progress || 0) / 100 : 0;
    
    return ((completedSteps + currentStepProgress) / steps.length) * 100;
  }, [steps, currentStep]);

  const getStepIcon = (step: ProgressStep, index: number) => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-success" />;
      case 'processing':
        return <Loader2 className="h-4 w-4 text-primary animate-spin" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      case 'pending':
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStepVariant = (step: ProgressStep) => {
    switch (step.status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'default';
      case 'error':
        return 'destructive';
      case 'pending':
      default:
        return 'secondary';
    }
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Overall Progress */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <h4 className="font-medium">Overall Progress</h4>
              <span className="text-sm text-muted-foreground">
                {Math.round(overallProgress)}%
              </span>
            </div>
            <Progress value={overallProgress} className="h-2" />
          </div>

          {/* Step Details */}
          <div className="space-y-4">
            {steps.map((step, index) => (
              <div
                key={step.id}
                className={cn(
                  "flex items-start gap-3 p-3 rounded-lg transition-colors",
                  step.status === 'processing' && "bg-primary/5 border border-primary/20",
                  step.status === 'completed' && "bg-success/5",
                  step.status === 'error' && "bg-destructive/5"
                )}
              >
                <div className="flex-shrink-0 mt-0.5">
                  {getStepIcon(step, index)}
                </div>
                
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{step.label}</span>
                    <Badge variant={getStepVariant(step) as 'default' | 'destructive' | 'outline' | 'secondary'} className="text-xs">
                      {step.status}
                    </Badge>
                  </div>
                  
                  {step.description && (
                    <p className="text-sm text-muted-foreground">
                      {step.description}
                    </p>
                  )}
                  
                  {step.status === 'processing' && step.progress !== undefined && (
                    <div className="space-y-1">
                      <Progress value={step.progress} className="h-1" />
                      <span className="text-xs text-muted-foreground">
                        {step.progress}% complete
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Hook for managing progress state
export function useProgressTracker(initialSteps: ProgressStep[]) {
  const [steps, setSteps] = useState<ProgressStep[]>(initialSteps);
  const [currentStep, setCurrentStep] = useState<string | undefined>();

  const updateStep = (stepId: string, updates: Partial<ProgressStep>) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, ...updates } : step
    ));
  };

  const startStep = (stepId: string) => {
    setCurrentStep(stepId);
    updateStep(stepId, { status: 'processing', progress: 0 });
  };

  const completeStep = (stepId: string) => {
    updateStep(stepId, { status: 'completed', progress: 100 });
    
    // Auto-advance to next step if available
    const currentIndex = steps.findIndex(step => step.id === stepId);
    const nextStep = steps[currentIndex + 1];
    if (nextStep && nextStep.status === 'pending') {
      setCurrentStep(nextStep.id);
    } else {
      setCurrentStep(undefined);
    }
  };

  const errorStep = (stepId: string, error?: string) => {
    updateStep(stepId, { 
      status: 'error', 
      description: error || steps.find(s => s.id === stepId)?.description 
    });
    setCurrentStep(undefined);
  };

  const updateProgress = (stepId: string, progress: number) => {
    updateStep(stepId, { progress: Math.min(100, Math.max(0, progress)) });
  };

  const reset = () => {
    setSteps(initialSteps);
    setCurrentStep(undefined);
  };

  return {
    steps,
    currentStep,
    updateStep,
    startStep,
    completeStep,
    errorStep,
    updateProgress,
    reset
  };
}