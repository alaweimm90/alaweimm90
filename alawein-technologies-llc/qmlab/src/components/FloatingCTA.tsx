import React, { useState, useEffect } from 'react';
import { EnhancedButton } from '@/components/ui/enhanced-button';
import { Play, BookOpen, Search } from 'lucide-react';
import { useTutorial } from '@/contexts/TutorialContext';
import { trackQuantumEvents } from '@/lib/analytics';

interface FloatingCTAProps {
  className?: string;
}

export const FloatingCTA: React.FC<FloatingCTAProps> = ({ className = '' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const { startTutorial, hasSeenTutorial } = useTutorial();

  // Show floating CTA when user scrolls past hero section
  useEffect(() => {
    const handleScroll = () => {
      const heroSection = document.querySelector('section');
      if (heroSection) {
        const heroBottom = heroSection.offsetTop + heroSection.offsetHeight;
        const scrollPosition = window.scrollY + window.innerHeight;
        setIsVisible(scrollPosition > heroBottom + 200);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleStartTutorial = () => {
    const defaultSteps = [
      {
        id: 'circuit-builder',
        title: 'Build Quantum Circuits',
        content: 'Design quantum circuits by selecting gates and adding them to qubits.',
        target: '[data-tutorial="circuit-builder"]',
        position: 'right' as const,
        action: 'Try Adding a Gate'
      },
      {
        id: 'bloch-sphere',
        title: 'Visualize Quantum States',
        content: 'Watch how quantum states evolve on the Bloch sphere.',
        target: '[data-tutorial="bloch-sphere"]',
        position: 'left' as const,
        action: 'Explore the Visualization'
      }
    ];
    
    startTutorial(defaultSteps);
    trackQuantumEvents.tutorialStart(2);
  };

  const triggerSearch = () => {
    const searchEvent = new KeyboardEvent('keydown', {
      key: 'k',
      metaKey: true,
      bubbles: true
    });
    document.dispatchEvent(searchEvent);
    trackQuantumEvents.searchOpen('floating-cta');
  };

  if (!isVisible) return null;

  return (
    <div className={`fixed bottom-6 right-6 z-40 flex flex-col gap-3 ${className}`}>
      {/* Main floating CTA */}
      <EnhancedButton
        variant="cta-floating"
        size="floating"
        onClick={handleStartTutorial}
        className="shadow-2xl hover:shadow-3xl animate-float"
        aria-label={hasSeenTutorial ? "Replay quantum tutorial" : "Start quantum tutorial"}
        title={hasSeenTutorial ? "Replay Tutorial" : "Start Learning!"}
      >
        <Play className="w-6 h-6" />
        {!hasSeenTutorial && (
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
        )}
      </EnhancedButton>

      {/* Secondary floating actions */}
      <div className="flex flex-col gap-2 opacity-80 hover:opacity-100 transition-opacity">
        <EnhancedButton
          variant="cta-secondary"
          size="icon"
          onClick={triggerSearch}
          className="h-10 w-10 shadow-lg"
          aria-label="Open search"
        >
          <Search className="w-4 h-4" />
        </EnhancedButton>

        <EnhancedButton
          variant="cta-learn"
          size="icon"
          onClick={() => {
            const learningSection = document.getElementById('learning-resources');
            if (learningSection) {
              learningSection.scrollIntoView({ behavior: 'smooth' });
            }
          }}
          className="h-10 w-10 shadow-lg"
          aria-label="Go to learning resources"
        >
          <BookOpen className="w-4 h-4" />
        </EnhancedButton>
      </div>
    </div>
  );
};