import React, { useEffect, useState, useCallback } from 'react';
import { useTutorial } from '@/contexts/TutorialContext';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { X, ArrowLeft, ArrowRight, Play } from 'lucide-react';
import { useAccessibilityContext } from '@/components/AccessibilityProvider';

interface Position {
  top: number;
  left: number;
  width: number;
  height: number;
}

export const TutorialOverlay: React.FC = () => {
  const { isActive, currentStep, totalSteps, currentTutorial, nextStep, previousStep, skipTutorial, completeTutorial } = useTutorial();
  const { announce } = useAccessibilityContext();
  const [highlightPosition, setHighlightPosition] = useState<Position | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ top: number; left: number } | null>(null);

  const updatePositions = useCallback(() => {
    if (!isActive || !currentTutorial) return;

    const currentStepData = currentTutorial[currentStep];
    if (!currentStepData) return;

    const targetElement = document.querySelector(currentStepData.target);
    if (!targetElement) return;

    const rect = targetElement.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

    // Set highlight position
    setHighlightPosition({
      top: rect.top + scrollTop,
      left: rect.left + scrollLeft,
      width: rect.width,
      height: rect.height
    });

    // Calculate tooltip position based on preferred position
    let tooltipTop = rect.top + scrollTop;
    let tooltipLeft = rect.left + scrollLeft;

    const tooltipPadding = 16;
    const tooltipWidth = 320; // Approximate width
    const tooltipHeight = 200; // Approximate height

    switch (currentStepData.position) {
      case 'top':
        tooltipTop = rect.top + scrollTop - tooltipHeight - tooltipPadding;
        tooltipLeft = rect.left + scrollLeft + rect.width / 2 - tooltipWidth / 2;
        break;
      case 'bottom':
        tooltipTop = rect.bottom + scrollTop + tooltipPadding;
        tooltipLeft = rect.left + scrollLeft + rect.width / 2 - tooltipWidth / 2;
        break;
      case 'left':
        tooltipTop = rect.top + scrollTop + rect.height / 2 - tooltipHeight / 2;
        tooltipLeft = rect.left + scrollLeft - tooltipWidth - tooltipPadding;
        break;
      case 'right':
      default:
        tooltipTop = rect.top + scrollTop + rect.height / 2 - tooltipHeight / 2;
        tooltipLeft = rect.right + scrollLeft + tooltipPadding;
        break;
    }

    // Ensure tooltip stays within viewport
    const maxLeft = window.innerWidth - tooltipWidth - tooltipPadding;
    const maxTop = window.innerHeight - tooltipHeight - tooltipPadding;

    tooltipLeft = Math.max(tooltipPadding, Math.min(maxLeft, tooltipLeft));
    tooltipTop = Math.max(tooltipPadding, Math.min(maxTop, tooltipTop));

    setTooltipPosition({ top: tooltipTop, left: tooltipLeft });
  }, [isActive, currentTutorial, currentStep]);

  // Update positions when tutorial changes or window resizes
  useEffect(() => {
    if (isActive) {
      updatePositions();
      window.addEventListener('resize', updatePositions);
      window.addEventListener('scroll', updatePositions);

      return () => {
        window.removeEventListener('resize', updatePositions);
        window.removeEventListener('scroll', updatePositions);
      };
    }
  }, [isActive, updatePositions]);

  // Announce step changes for screen readers
  useEffect(() => {
    if (isActive && currentTutorial) {
      const stepData = currentTutorial[currentStep];
      if (stepData) {
        announce(`Tutorial step ${currentStep + 1} of ${totalSteps}: ${stepData.title}. ${stepData.content}`);
      }
    }
  }, [isActive, currentStep, currentTutorial, totalSteps, announce]);

  // Scroll to highlighted element
  useEffect(() => {
    if (isActive && currentTutorial) {
      const stepData = currentTutorial[currentStep];
      if (stepData) {
        const targetElement = document.querySelector(stepData.target);
        if (targetElement) {
          targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }
  }, [isActive, currentTutorial, currentStep]);

  if (!isActive || !currentTutorial || !tooltipPosition) {
    return null;
  }

  const currentStepData = currentTutorial[currentStep];
  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === totalSteps - 1;

  return (
    <>
      {/* Dark overlay */}
      <div 
        className="fixed inset-0 bg-black/50 z-40 transition-opacity duration-300"
        aria-hidden="true"
      />
      
      {/* Highlight box */}
      {highlightPosition && (
        <div
          className="fixed z-50 border-2 border-primary rounded-lg shadow-glow-primary animate-pulse"
          style={{
            top: highlightPosition.top - 4,
            left: highlightPosition.left - 4,
            width: highlightPosition.width + 8,
            height: highlightPosition.height + 8,
          }}
          aria-hidden="true"
        />
      )}
      
      {/* Tutorial tooltip */}
      <Card
        className="fixed z-50 w-80 max-w-[90vw] p-6 bg-card border-primary/20 shadow-xl"
        style={{
          top: tooltipPosition.top,
          left: tooltipPosition.left,
        }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="tutorial-title"
        aria-describedby="tutorial-content"
      >
        {/* Close button */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 h-6 w-6"
          onClick={skipTutorial}
          aria-label="Close tutorial"
        >
          <X className="h-4 w-4" />
        </Button>
        
        {/* Step indicators */}
        <div className="flex justify-center mb-4">
          <div className="flex space-x-2">
            {Array.from({ length: totalSteps }, (_, i) => (
              <div
                key={i}
                className={`w-2 h-2 rounded-full transition-colors duration-200 ${
                  i === currentStep 
                    ? 'bg-primary' 
                    : i < currentStep 
                    ? 'bg-primary/60' 
                    : 'bg-muted'
                }`}
                aria-hidden="true"
              />
            ))}
          </div>
        </div>
        
        {/* Content */}
        <div className="space-y-4">
          <div>
            <h3 id="tutorial-title" className="text-lg font-semibold mb-2">
              {currentStepData.title}
            </h3>
            <p id="tutorial-content" className="text-muted-foreground text-sm leading-relaxed">
              {currentStepData.content}
            </p>
          </div>
          
          {/* Progress indicator */}
          <div className="text-xs text-muted-foreground text-center">
            Step {currentStep + 1} of {totalSteps}
          </div>
          
          {/* Action buttons */}
          <div className="flex justify-between items-center pt-2">
            <div className="flex space-x-2">
              {!isFirstStep && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={previousStep}
                  className="gap-2"
                >
                  <ArrowLeft className="h-3 w-3" />
                  Back
                </Button>
              )}
            </div>
            
            <div className="flex space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={skipTutorial}
                className="text-muted-foreground hover:text-foreground"
              >
                Skip Tour
              </Button>
              
              {isLastStep ? (
                <Button
                  size="sm"
                  onClick={completeTutorial}
                  className="gap-2"
                >
                  <Play className="h-3 w-3" />
                  Start Exploring
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={nextStep}
                  className="gap-2"
                >
                  {currentStepData.action || 'Next'}
                  <ArrowRight className="h-3 w-3" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </Card>
    </>
  );
};