import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { trackQuantumEvents } from '@/lib/analytics';

export interface TutorialStep {
  id: string;
  title: string;
  content: string;
  target: string; // CSS selector for the element to highlight
  position?: 'top' | 'bottom' | 'left' | 'right';
  action?: string; // Optional call-to-action text
}

export interface TutorialContextType {
  isActive: boolean;
  currentStep: number;
  totalSteps: number;
  currentTutorial: TutorialStep[] | null;
  startTutorial: (steps: TutorialStep[]) => void;
  nextStep: () => void;
  previousStep: () => void;
  skipTutorial: () => void;
  completeTutorial: () => void;
  replayTutorial: () => void;
  hasSeenTutorial: boolean;
  setHasSeenTutorial: (seen: boolean) => void;
  showTourPreference: boolean;
  setShowTourPreference: (show: boolean) => void;
}

const TutorialContext = createContext<TutorialContextType | undefined>(undefined);

const TUTORIAL_STORAGE_KEY = 'qmlab-tutorial-seen';
const TUTORIAL_PREFERENCE_KEY = 'qmlab-show-tour';

export const TutorialProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isActive, setIsActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentTutorial, setCurrentTutorial] = useState<TutorialStep[] | null>(null);
  const [hasSeenTutorial, setHasSeenTutorialState] = useState(() => {
    try {
      return localStorage.getItem(TUTORIAL_STORAGE_KEY) === 'true';
    } catch {
      return false;
    }
  });

  const [showTourPreference, setShowTourPreferenceState] = useState(() => {
    try {
      const stored = localStorage.getItem(TUTORIAL_PREFERENCE_KEY);
      // Default to false (don't show tour automatically)
      return stored === null ? false : stored === 'true';
    } catch {
      return false;
    }
  });

  const setHasSeenTutorial = useCallback((seen: boolean) => {
    setHasSeenTutorialState(seen);
    try {
      if (seen) {
        localStorage.setItem(TUTORIAL_STORAGE_KEY, 'true');
      } else {
        localStorage.removeItem(TUTORIAL_STORAGE_KEY);
      }
    } catch {
      // Handle localStorage errors gracefully
    }
  }, []);

  const setShowTourPreference = useCallback((show: boolean) => {
    setShowTourPreferenceState(show);
    try {
      localStorage.setItem(TUTORIAL_PREFERENCE_KEY, show ? 'true' : 'false');
    } catch {
      // Handle localStorage errors gracefully
    }
  }, []);

  const startTutorial = useCallback((steps: TutorialStep[]) => {
    setCurrentTutorial(steps);
    setCurrentStep(0);
    setIsActive(true);
    
    // Track tutorial start
    trackQuantumEvents.tutorialStart(steps.length);
  }, []);

  const nextStep = useCallback(() => {
    if (!currentTutorial) return;
    
    const newStep = currentStep + 1;
    
    if (newStep >= currentTutorial.length) {
      completeTutorial();
    } else {
      setCurrentStep(newStep);
      trackQuantumEvents.tutorialStep(newStep + 1, currentTutorial.length);
    }
  }, [currentStep, currentTutorial]);

  const previousStep = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const skipTutorial = useCallback(() => {
    setIsActive(false);
    setCurrentTutorial(null);
    setCurrentStep(0);
    setHasSeenTutorial(true);
    
    // Track tutorial skip
    trackQuantumEvents.tutorialSkip(currentStep + 1, currentTutorial?.length || 0);
  }, [currentStep, currentTutorial, setHasSeenTutorial]);

  const completeTutorial = useCallback(() => {
    setIsActive(false);
    setCurrentTutorial(null);
    setCurrentStep(0);
    setHasSeenTutorial(true);
    
    // Track tutorial completion
    trackQuantumEvents.tutorialComplete(currentTutorial?.length || 0);
  }, [currentTutorial, setHasSeenTutorial]);

  const replayTutorial = useCallback(() => {
    if (currentTutorial) {
      setCurrentStep(0);
      setIsActive(true);
      setHasSeenTutorial(false);
      
      // Track tutorial replay
      trackQuantumEvents.tutorialReplay();
    }
  }, [currentTutorial, setHasSeenTutorial]);

  // Auto-start tutorial ONLY if user preference allows it
  useEffect(() => {
    // Only auto-start if:
    // 1. User has explicitly enabled tour preference
    // 2. They haven't seen the tutorial yet
    // 3. Tutorial is not currently active
    if (showTourPreference && !hasSeenTutorial && !isActive) {
      const timer = setTimeout(() => {
        // Define the default tutorial steps
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
        
        startTutorial(defaultSteps);
      }, 2000); // 2 second delay to let the page load

      return () => clearTimeout(timer);
    }
  }, [showTourPreference, hasSeenTutorial, isActive, startTutorial]);

  const value: TutorialContextType = {
    isActive,
    currentStep,
    totalSteps: currentTutorial?.length || 0,
    currentTutorial,
    startTutorial,
    nextStep,
    previousStep,
    skipTutorial,
    completeTutorial,
    replayTutorial,
    hasSeenTutorial,
    setHasSeenTutorial,
    showTourPreference,
    setShowTourPreference
  };

  return (
    <TutorialContext.Provider value={value}>
      {children}
    </TutorialContext.Provider>
  );
};

export const useTutorial = (): TutorialContextType => {
  const context = useContext(TutorialContext);
  if (context === undefined) {
    throw new Error('useTutorial must be used within a TutorialProvider');
  }
  return context;
};