import React from 'react';
import { CheckCircle } from 'lucide-react';

interface ProgressIndicatorProps {
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
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  currentStep,
  totalSteps,
  completedSteps,
  tierColors
}) => {
  const steps = [
    'Account',
    'Personal', 
    'Health',
    'Training',
    'Nutrition',
    'Tier Goals',
    'Payment'
  ];

  const getStepStatus = (stepNumber: number) => {
    if (completedSteps.includes(stepNumber)) return 'completed';
    if (stepNumber === currentStep) return 'current';
    return 'pending';
  };

  return (
    <div className="progress-indicator mb-4 mt-6">
      {/* Desktop Progress Bar - Increased width for better visibility */}
      <div className="hidden sm:flex items-start justify-between mb-6 gap-1">
        {steps.map((step, index) => {
          const stepNumber = index + 1;
          const status = getStepStatus(stepNumber);
          
          return (
            <div key={stepNumber} className="flex items-center min-w-0">
              <div className="flex items-center min-w-0">
                {/* Step Circle */}
                <div
                  className={`
                    flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-200 flex-shrink-0
                    ${status === 'completed' 
                      ? 'text-white' 
                      : status === 'current'
                      ? 'text-white'
                      : 'bg-gray-100 border-gray-300 text-gray-400'
                    }
                  `}
                  style={status !== 'pending' ? {
                    backgroundColor: tierColors?.primary || 'hsl(var(--repz-orange))',
                    borderColor: tierColors?.primary || 'hsl(var(--repz-orange))'
                  } : {}}
                >
                  {status === 'completed' ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <span className="text-sm font-semibold">{stepNumber}</span>
                  )}
                </div>
                {/* Step Label - Increased width */}
                <div className="ml-3 min-w-0 w-20">
                  <div
                    className={`text-sm font-medium truncate ${
                      status === 'current' ? '' : 'text-gray-600'
                    }`}
                    style={status === 'current' ? { color: tierColors?.primary || 'hsl(var(--repz-orange))' } : {}}
                    title={step}
                  >
                    {step}
                  </div>
                  <div className="text-xs text-gray-400 whitespace-nowrap">
                    {stepNumber}/{totalSteps}
                  </div>
                </div>
              </div>
              {/* Connector Line */}
              {stepNumber < totalSteps && (
                <div
                  className={`
                    w-8 h-0.5 mx-2 transition-all duration-200 flex-shrink-0
                    ${completedSteps.includes(stepNumber) 
                      ? '' 
                      : 'bg-gray-300'
                    }
                  `}
                  style={completedSteps.includes(stepNumber) ? {
                    backgroundColor: tierColors?.primary || 'hsl(var(--repz-orange))'
                  } : {}}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Mobile Progress Bar */}
      <div className="sm:hidden mb-6">
        <div className="flex items-center justify-between text-sm text-gray-600 mb-3">
          <span>Step {currentStep} of {totalSteps}</span>
          <span>{Math.round((currentStep / totalSteps) * 100)}% Complete</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
          <div
            className="h-2 rounded-full transition-all duration-300"
            style={{ 
              width: `${(currentStep / totalSteps) * 100}%`,
              backgroundColor: tierColors?.primary || 'hsl(var(--repz-orange))'
            }}
          />
        </div>
        
        {/* Mobile Step Labels */}
        <div className="flex justify-center">
          <span className="text-sm text-gray-600 font-medium">
            {steps[currentStep - 1]}
          </span>
        </div>
      </div>
    </div>
  );
};