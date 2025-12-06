import { useEffect, useState } from 'react';

interface PageTransitionProps {
  children: React.ReactNode;
}

/**
 * PageTransition component displays a loading screen during page transitions
 *
 * Shows an animated loading overlay with spinning logo and brand text for 800ms
 * before revealing page content. Provides a premium loading experience during route changes.
 *
 * @component
 * @param {PageTransitionProps} props - Component props
 * @param {React.ReactNode} props.children - Page content to display after transition
 *
 * @example
 * <PageTransition>
 *   <YourPageContent />
 * </PageTransition>
 *
 * @remarks
 * - Fixed 800ms loading duration
 * - Animated spinning logo with pulse effect
 * - Fade-in animation for brand text
 * - Uses localStorage key 'pageLoaded' to skip on subsequent loads
 * - Z-index 50 to overlay content
 */
const PageTransition = ({ children }: PageTransitionProps) => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-background z-50 flex items-center justify-center">
        <div className="flex flex-col items-center space-y-6">
          {/* Animated Logo */}
          <div className="relative">
            <div className="w-16 h-16 border-4 border-accent/20 rounded-full animate-pulse" />
            <div className="absolute inset-0 w-16 h-16 border-4 border-accent border-t-transparent rounded-full animate-spin" />
          </div>

          {/* Brand Text */}
          <div className="text-center animate-fade-in-up">
            <h1 className="text-2xl font-headline font-bold mb-2">Live It Iconic</h1>
            <p className="text-muted-foreground text-sm">Engineered for performance</p>
          </div>
        </div>
      </div>
    );
  }

  return <div className="animate-fade-in">{children}</div>;
};

export default PageTransition;
