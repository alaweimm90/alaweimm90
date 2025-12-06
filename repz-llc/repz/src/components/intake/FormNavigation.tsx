import React from 'react';
import { Button } from "@/ui/atoms/Button";
import { ArrowLeft, ArrowRight } from 'lucide-react';

interface FormNavigationProps {
  currentStep: number;
  totalSteps: number;
  onPrevious: () => void;
  onNext: () => void;
  onSubmit: () => void;
  isSubmitting: boolean;
  canProceed: boolean;
  tierColors?: {
    primary: string;
    background: string;
    text: string;
    accent: string;
    border: string;
    badge: string;
  };
}

export const FormNavigation: React.FC<FormNavigationProps> = ({
  currentStep,
  totalSteps,
  onPrevious,
  onNext,
  onSubmit,
  isSubmitting,
  canProceed,
  tierColors
}) => {
  const isFirstStep = currentStep === 1;
  const isLastStep = currentStep === totalSteps;

  return (
    <div className="form-navigation flex items-center justify-between py-6 border-t border-gray-200 bg-white">
      {/* Previous Button */}
      <Button
        type="button"
        variant="outline"
        onClick={onPrevious}
        disabled={isFirstStep || isSubmitting}
        className="flex items-center gap-2"
      >
        <ArrowLeft className="w-4 h-4" />
        Previous
      </Button>

      {/* Step Info */}
      <div className="flex items-center gap-4 text-sm text-gray-600">
        <span className="hidden sm:inline">Step {currentStep} of {totalSteps}</span>
        
        {!canProceed && currentStep > 1 && (
          <span className="text-repz-orange text-xs">
            Please complete required fields
          </span>
        )}
      </div>

      {/* Next/Submit Button */}
      {isLastStep ? (
        <Button
          type="button"
          onClick={onSubmit}
          disabled={!canProceed || isSubmitting}
          className="flex items-center gap-2 text-white"
          style={{
            backgroundColor: tierColors?.primary || 'hsl(var(--repz-orange))',
            borderColor: tierColors?.primary || 'hsl(var(--repz-orange))'
          }}
        >
          {isSubmitting ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              Processing...
            </>
          ) : (
            <>
              Complete Registration
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </Button>
      ) : (
        <Button
          type="button"
          onClick={onNext}
          disabled={!canProceed || isSubmitting}
          className="flex items-center gap-2 text-white"
          style={{
            backgroundColor: tierColors?.primary || 'hsl(var(--repz-orange))',
            borderColor: tierColors?.primary || 'hsl(var(--repz-orange))'
          }}
        >
          Next
          <ArrowRight className="w-4 h-4" />
        </Button>
      )}
    </div>
  );
};