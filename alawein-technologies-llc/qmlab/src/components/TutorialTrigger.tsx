import React from 'react';
import { Button } from '@/components/ui/button';
import { HelpCircle, Play } from 'lucide-react';
import { useTutorial } from '@/contexts/TutorialContext';
import { TutorialStep } from '@/contexts/TutorialContext';

export interface TutorialTriggerProps {
  variant?: 'icon' | 'button';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  customSteps?: TutorialStep[];
}

export const TutorialTrigger: React.FC<TutorialTriggerProps> = ({ 
  variant = 'button', 
  size = 'md', 
  className = '',
  customSteps
}) => {
  const { startTutorial, replayTutorial, hasSeenTutorial } = useTutorial();

  const defaultSteps: TutorialStep[] = [
    {
      id: 'circuit-builder',
      title: 'Build Quantum Circuits',
      content: 'Design quantum circuits by selecting gates and adding them to qubits. Each gate transforms the quantum state in unique ways.',
      target: '[data-tutorial="circuit-builder"]',
      position: 'right',
      action: 'Try Adding a Gate'
    },
    {
      id: 'bloch-sphere',
      title: 'Visualize Quantum States',
      content: 'Watch how quantum states evolve on the Bloch sphere. This 3D visualization shows the quantum state of your qubits in real-time.',
      target: '[data-tutorial="bloch-sphere"]',
      position: 'left',
      action: 'Explore the Visualization'
    }
  ];

  const handleStartTutorial = () => {
    const steps = customSteps || defaultSteps;
    if (hasSeenTutorial) {
      replayTutorial();
    } else {
      startTutorial(steps);
    }
  };

  if (variant === 'icon') {
    return (
      <Button
        variant="ghost"
        size="icon"
        onClick={handleStartTutorial}
        className={`relative ${className}`}
        aria-label={hasSeenTutorial ? "Replay tutorial" : "Start tutorial"}
        title={hasSeenTutorial ? "Replay tutorial" : "Start tutorial"}
      >
        <HelpCircle className="h-5 w-5" />
        {!hasSeenTutorial && (
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full animate-pulse" 
               aria-hidden="true" />
        )}
      </Button>
    );
  }

  return (
    <Button
      variant="outline"
      size={size === 'md' ? 'md' : size}
      onClick={handleStartTutorial}
      className={`gap-2 ${className}`}
    >
      <Play className="h-4 w-4" />
      {hasSeenTutorial ? 'Replay Tour' : 'Take a Tour'}
      {!hasSeenTutorial && (
        <span className="ml-1 px-2 py-0.5 text-xs bg-primary text-primary-foreground rounded-full">
          New
        </span>
      )}
    </Button>
  );
};