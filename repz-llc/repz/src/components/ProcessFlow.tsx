import React, { useEffect, useRef, useState } from 'react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';
import { CheckCircle, FileText, Zap, Target, Trophy } from 'lucide-react';

interface ProcessStep {
  id: number;
  title: string;
  duration: string;
  description: string;
  icon: React.ElementType;
  color: string;
}

const processSteps: ProcessStep[] = [
  {
    id: 1,
    title: "Selection",
    duration: "5 minutes",
    description: "Complete form, choose tier",
    icon: FileText,
    color: "repz-orange"
  },
  {
    id: 2,
    title: "Plan",
    duration: "2-3 days", 
    description: "Custom program development",
    icon: Target,
    color: "repz-orange"
  },
  {
    id: 3,
    title: "Program",
    duration: "1-2 days",
    description: "Receive program",
    icon: Zap,
    color: "repz-orange"
  },
  {
    id: 4,
    title: "Results",
    duration: "4-12 weeks",
    description: "Ongoing coaching and results",
    icon: Trophy,
    color: "repz-orange-glow"
  }
];

export const ProcessFlow: React.FC = () => {
  const { ref, isVisible } = useScrollAnimation({ threshold: 0.3 });
  const [activeStep, setActiveStep] = useState(0);
  const [lineProgress, setLineProgress] = useState(0);

  useEffect(() => {
    if (isVisible) {
      const timer = setInterval(() => {
        setActiveStep(prev => {
          if (prev < processSteps.length) {
            setLineProgress((prev + 1) / processSteps.length * 100);
            return prev + 1;
          }
          return prev;
        });
      }, 800);

      return () => clearInterval(timer);
    }
  }, [isVisible]);

  return (
    <div ref={ref as React.RefObject<HTMLDivElement>} className="w-full max-w-6xl mx-auto px-4">
      {/* Header */}
      <div className="text-center mb-16">
        <h2 className="text-3xl sm:text-4xl font-montserrat font-bold mb-4 text-white">
          How Our Monthly Coaching Works
        </h2>
        <p className="text-xl text-gray-300 mb-2">
          From consultation to transformation - here's our proven monthly coaching process
        </p>
        <p className="text-sm text-gray-400">
          This process applies to monthly coaching subscriptions only (Core, Adaptive, Performance, Concierge)
        </p>
      </div>

      {/* Process Flow */}
      <div className="relative">
        {/* Connection Line */}
        <div className="absolute top-16 left-0 right-0 h-1 bg-gray-600/30 rounded-full overflow-hidden">
          <div 
            className="h-full bg-repz-orange rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${lineProgress}%` }}
          />
        </div>

        {/* Process Steps */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative">
          {processSteps.map((step, index) => {
            const Icon = step.icon;
            const isActive = index < activeStep;
            const isCurrentStep = index === activeStep - 1;
            
            return (
              <div 
                key={step.id}
                className="flex flex-col items-center text-center group cursor-pointer"
                onMouseEnter={() => {
                  if (isVisible && activeStep >= processSteps.length) {
                    setLineProgress((index + 1) / processSteps.length * 100);
                  }
                }}
                onMouseLeave={() => {
                  if (isVisible && activeStep >= processSteps.length) {
                    setLineProgress(100);
                  }
                }}
              >
                {/* Step Circle */}
                <div className={`
                  relative w-16 h-16 rounded-full flex items-center justify-center mb-4 z-10
                  ${isActive 
                    ? step.color === 'repz-orange' 
                      ? 'bg-repz-orange/70 shadow-md shadow-repz-orange/20' 
                      : step.color === 'repz-orange'
                      ? 'bg-repz-orange shadow-lg shadow-repz-orange/30'
                      : step.color === 'repz-orange'
                      ? 'bg-repz-orange shadow-lg shadow-repz-orange/40'
                      : 'bg-repz-orange shadow-2xl shadow-repz-orange/60 animate-pulse'
                    : 'bg-gray-700 border-2 border-gray-600'
                  }
                  ${isCurrentStep ? 'animate-gentle-pulse scale-110' : ''}
                  ${step.id === 4 ? 'transition-all [transition-duration:8000ms] animate-gentle-pulse' : 'transition-all duration-500'}
                  group-hover:scale-110 group-hover:shadow-xl
                  ${step.color === 'repz-orange-glow' && isActive ? 'ring-4 ring-repz-orange/30 ring-offset-2 ring-offset-gray-800' : ''}
                `}>
                  {isActive && (
                    <CheckCircle className="absolute -top-2 -right-2 w-6 h-6 text-green-400 bg-gray-800 rounded-full animate-scale-in" />
                  )}
                  
                  <Icon className={`
                    w-8 h-8 transition-all duration-300
                    ${isActive ? 'text-white' : 'text-gray-400'}
                  `} />
                  
                  {/* Step Number */}
                  <div className={`
                    absolute -bottom-2 -right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                    ${isActive 
                      ? 'bg-white text-repz-orange' 
                      : 'bg-gray-600 text-gray-300'
                    }
                  `}>
                    {step.id}
                  </div>
                </div>

                {/* Step Content */}
                <div className={`transition-all duration-500 ${isActive ? 'opacity-100 transform translate-y-0' : 'opacity-60 transform translate-y-2'}`}>
                  <h3 className="text-xl font-semibold mb-2 text-white group-hover:text-repz-orange transition-colors">
                    {step.title}
                  </h3>
                  <p className={`text-sm font-medium mb-3 transition-colors ${
                    isActive 
                      ? step.color === 'repz-orange' 
                        ? 'text-repz-orange/80' 
                        : step.color === 'repz-orange'
                        ? 'text-repz-orange'
                        : step.color === 'repz-orange'
                        ? 'text-repz-orange'
                        : 'text-repz-orange font-bold'
                      : 'text-gray-400'
                  }`}>
                    {step.duration}
                  </p>
                  <p className="text-gray-300 text-sm leading-relaxed group-hover:text-gray-200 transition-colors">
                    {step.description}
                  </p>
                </div>

                {/* Progress Indicator */}
                <div className="mt-4 w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                  <div className={`
                    h-full transition-all duration-1000 ease-out rounded-full
                    ${isActive 
                      ? 'bg-repz-orange w-full' 
                      : 'bg-gray-600 w-0'
                    }
                  `} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Floating Particles Along Path */}
        {isVisible && (
          <>
            <div className="absolute top-16 left-1/4 w-2 h-2 bg-repz-orange rounded-full animate-gentle-drift opacity-60" />
            <div className="absolute top-16 left-2/4 w-1.5 h-1.5 bg-repz-orange rounded-full animate-gentle-float opacity-40" style={{animationDelay: '1s'}} />
            <div className="absolute top-16 left-3/4 w-1 h-1 bg-repz-orange rounded-full animate-gentle-pulse opacity-50" style={{animationDelay: '2s'}} />
          </>
        )}
      </div>
    </div>
  );
};