import { memo } from 'react';
import { Loader2, Swords, Brain, Sparkles, Zap } from 'lucide-react';

interface LoadingStateProps {
  type: 'debate-init' | 'ai-loading' | 'environment-setup' | 'citation-check' | 'generic';
  message?: string;
  progress?: number;
}

const LoadingStateComponent = ({ type, message, progress }: LoadingStateProps) => {
  const getLoadingContent = () => {
    switch (type) {
      case 'debate-init':
        return {
          icon: Swords,
          title: 'Initializing Neural Arena',
          description: message || 'Preparing combatants for intellectual battle...',
          animation: 'animate-pulse'
        };
      case 'ai-loading':
        return {
          icon: Brain,
          title: 'Loading AI Personalities',
          description: message || 'Calibrating neural signatures and energy fields...',
          animation: 'animate-spin'
        };
      case 'environment-setup':
        return {
          icon: Sparkles,
          title: 'Setting Up Environment',
          description: message || 'Generating contextual effects and particle systems...',
          animation: 'animate-bounce'
        };
      case 'citation-check':
        return {
          icon: Zap,
          title: 'Verifying Citations',
          description: message || 'Checking sources and fact-checking claims...',
          animation: 'animate-ping'
        };
      default:
        return {
          icon: Loader2,
          title: 'Loading',
          description: message || 'Preparing your experience...',
          animation: 'animate-spin'
        };
    }
  };

  const content = getLoadingContent();
  const IconComponent = content.icon;

  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] space-y-6">
      {/* Animated Icon */}
      <div className="relative">
        <div className="absolute inset-0 bg-primary/20 rounded-full blur-3xl animate-pulse" />
        <div className={`relative glass-panel p-8 rounded-full ${content.animation}`}>
          <IconComponent className="h-16 w-16 text-primary" />
        </div>
        
        {/* Orbital particles */}
        <div className="absolute inset-0">
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 bg-primary rounded-full"
              style={{
                animation: `orbit ${3 + i}s linear infinite`,
                animationDelay: `${i * 0.5}s`,
                top: '50%',
                left: '50%',
                transform: `translate(-50%, -50%) rotate(${i * 120}deg) translateX(60px)`
              }}
            />
          ))}
        </div>
      </div>

      {/* Loading Text */}
      <div className="text-center space-y-2">
        <h3 className="heading-refined text-xl">{content.title}</h3>
        <p className="text-muted-foreground text-sm max-w-md">
          {content.description}
        </p>
      </div>

      {/* Progress Bar */}
      {progress !== undefined && (
        <div className="w-64">
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Custom CSS for orbit animation */}
      <style jsx>{`
        @keyframes orbit {
          from {
            transform: translate(-50%, -50%) rotate(0deg) translateX(60px) rotate(0deg);
          }
          to {
            transform: translate(-50%, -50%) rotate(360deg) translateX(60px) rotate(-360deg);
          }
        }
      `}</style>
    </div>
  );
};

export const LoadingState = memo(LoadingStateComponent);

interface ErrorStateProps {
  type: 'connection' | 'debate-failed' | 'citation-error' | 'generic';
  message?: string;
  onRetry?: () => void;
}

const ErrorStateComponent = ({ type, message, onRetry }: ErrorStateProps) => {
  const getErrorContent = () => {
    switch (type) {
      case 'connection':
        return {
          title: 'Connection Lost',
          description: message || 'Unable to connect to the Neural Arena. Please check your connection.',
          icon: '⚡',
          color: 'text-yellow-500'
        };
      case 'debate-failed':
        return {
          title: 'Debate Initialization Failed',
          description: message || 'The combatants could not be initialized. Please try again.',
          icon: '⚔️',
          color: 'text-red-500'
        };
      case 'citation-error':
        return {
          title: 'Citation Verification Error',
          description: message || 'Unable to verify sources. Debate will continue without fact-checking.',
          icon: '📚',
          color: 'text-orange-500'
        };
      default:
        return {
          title: 'Something Went Wrong',
          description: message || 'An unexpected error occurred. Please try again.',
          icon: '⚠️',
          color: 'text-red-500'
        };
    }
  };

  const content = getErrorContent();

  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] space-y-6">
      {/* Error Icon */}
      <div className="relative">
        <div className={`text-6xl ${content.color} animate-pulse`}>
          {content.icon}
        </div>
      </div>

      {/* Error Message */}
      <div className="text-center space-y-2 max-w-md">
        <h3 className="heading-refined text-xl">{content.title}</h3>
        <p className="text-muted-foreground text-sm">
          {content.description}
        </p>
      </div>

      {/* Retry Button */}
      {onRetry && (
        <button
          onClick={onRetry}
          className="glass-panel px-6 py-3 rounded-lg hover:bg-primary/10 transition-all duration-300 flex items-center gap-2"
        >
          <Loader2 className="h-4 w-4" />
          <span>Try Again</span>
        </button>
      )}
    </div>
  );
};

export const ErrorState = memo(ErrorStateComponent);

interface EmptyStateProps {
  type: 'no-debates' | 'no-results' | 'no-citations' | 'generic';
  message?: string;
  actionLabel?: string;
  onAction?: () => void;
}

const EmptyStateComponent = ({ type, message, actionLabel, onAction }: EmptyStateProps) => {
  const getEmptyContent = () => {
    switch (type) {
      case 'no-debates':
        return {
          title: 'No Active Debates',
          description: message || 'Select a scenario or configure combatants to begin.',
          icon: Swords,
          defaultAction: 'Start New Debate'
        };
      case 'no-results':
        return {
          title: 'No Results Yet',
          description: message || 'Complete a debate to see results and statistics.',
          icon: Brain,
          defaultAction: 'View Demo'
        };
      case 'no-citations':
        return {
          title: 'No Citations Found',
          description: message || 'Arguments will be evaluated without external sources.',
          icon: Sparkles,
          defaultAction: 'Continue Anyway'
        };
      default:
        return {
          title: 'Nothing Here Yet',
          description: message || 'Get started to see content.',
          icon: Zap,
          defaultAction: 'Get Started'
        };
    }
  };

  const content = getEmptyContent();
  const IconComponent = content.icon;

  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] space-y-6">
      {/* Empty State Icon */}
      <div className="relative">
        <div className="glass-minimal p-8 rounded-full">
          <IconComponent className="h-16 w-16 text-muted-foreground opacity-50" />
        </div>
      </div>

      {/* Empty State Message */}
      <div className="text-center space-y-2 max-w-md">
        <h3 className="heading-refined text-xl text-muted-foreground">
          {content.title}
        </h3>
        <p className="text-muted-foreground text-sm opacity-80">
          {content.description}
        </p>
      </div>

      {/* Action Button */}
      {onAction && (
        <button
          onClick={onAction}
          className="glass-panel px-6 py-3 rounded-lg hover:bg-primary/10 transition-all duration-300 border border-primary/20"
        >
          {actionLabel || content.defaultAction}
        </button>
      )}
    </div>
  );
};

export const EmptyState = memo(EmptyStateComponent);