import React, { useEffect, useRef, useState } from 'react';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';
import { Pause, Play, SkipForward, Volume2, VolumeX } from 'lucide-react';

// Focus management for complex interfaces
export const FocusManager = ({ children }: { children: React.ReactNode }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab' && containerRef.current) {
        const focusableElements = containerRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div ref={containerRef} className="focus-within:outline-none">
      {children}
    </div>
  );
};

// Screen reader announcements for dynamic content
export const LiveRegion = ({ 
  message, 
  priority = 'polite' 
}: { 
  message: string;
  priority?: 'polite' | 'assertive';
}) => {
  return (
    <div
      role="status"
      aria-live={priority}
      aria-atomic="true"
      className="sr-only"
    >
      {message}
    </div>
  );
};

// Accessible simulation controls
export const AccessibleSimulationControls = ({
  isPlaying,
  onPlay,
  onPause,
  onReset,
  onMute,
  isMuted = false,
  className
}: {
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onMute?: () => void;
  isMuted?: boolean;
  className?: string;
}) => {
  const { isMobile } = useResponsiveEnhanced();
  const [announceText, setAnnounceText] = useState('');

  const handlePlay = () => {
    onPlay();
    setAnnounceText('Simulation started');
  };

  const handlePause = () => {
    onPause();
    setAnnounceText('Simulation paused');
  };

  const handleReset = () => {
    onReset();
    setAnnounceText('Simulation reset');
  };

  const handleMute = () => {
    onMute?.();
    setAnnounceText(isMuted ? 'Audio enabled' : 'Audio muted');
  };

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <LiveRegion message={announceText} />
      
      <div 
        role="group" 
        aria-label="Simulation controls"
        className="flex items-center gap-1 sm:gap-2"
      >
        <button
          onClick={isPlaying ? handlePause : handlePlay}
          className={cn(
            'flex items-center justify-center rounded-md',
            'bg-primary text-primary-foreground',
            'hover:bg-primary/90 active:bg-primary/80',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
            'transition-all duration-200',
            isMobile ? 'w-12 h-12' : 'w-10 h-10 sm:w-12 sm:h-12'
          )}
          aria-label={isPlaying ? 'Pause simulation' : 'Play simulation'}
          aria-pressed={isPlaying}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4 sm:w-5 sm:h-5" />
          ) : (
            <Play className="w-4 h-4 sm:w-5 sm:h-5" />
          )}
        </button>

        <button
          onClick={handleReset}
          className={cn(
            'flex items-center justify-center rounded-md',
            'bg-muted text-muted-foreground',
            'hover:bg-muted/80 hover:text-foreground',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
            'transition-all duration-200',
            isMobile ? 'w-12 h-12' : 'w-10 h-10 sm:w-12 sm:h-12'
          )}
          aria-label="Reset simulation"
        >
          <SkipForward className="w-4 h-4 sm:w-5 sm:h-5" />
        </button>

        {onMute && (
          <button
            onClick={handleMute}
            className={cn(
              'flex items-center justify-center rounded-md',
              'bg-muted text-muted-foreground',
              'hover:bg-muted/80 hover:text-foreground',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
              'transition-all duration-200',
              isMobile ? 'w-12 h-12' : 'w-10 h-10 sm:w-12 sm:h-12'
            )}
            aria-label={isMuted ? 'Unmute audio' : 'Mute audio'}
            aria-pressed={isMuted}
          >
            {isMuted ? (
              <VolumeX className="w-4 h-4 sm:w-5 sm:h-5" />
            ) : (
              <Volume2 className="w-4 h-4 sm:w-5 sm:h-5" />
            )}
          </button>
        )}
      </div>
    </div>
  );
};

// Skip to content link
export const SkipToContent = () => {
  return (
    <a
      href="#main-content"
      className={cn(
        'absolute -top-40 left-6 z-50 px-4 py-2',
        'bg-primary text-primary-foreground rounded-md',
        'focus:top-6 transition-all duration-200',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-foreground'
      )}
    >
      Skip to main content
    </a>
  );
};

// Accessible data visualization descriptions
export const ChartDescription = ({ 
  title, 
  summary, 
  data 
}: { 
  title: string;
  summary: string;
  data?: Array<{ label: string; value: number }>;
}) => {
  return (
    <div className="sr-only">
      <h3>{title}</h3>
      <p>{summary}</p>
      {data && (
        <dl>
          {data.map((item, index) => (
            <React.Fragment key={index}>
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
            </React.Fragment>
          ))}
        </dl>
      )}
    </div>
  );
};