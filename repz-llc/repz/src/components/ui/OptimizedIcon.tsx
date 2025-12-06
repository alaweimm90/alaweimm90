import React, { memo, Suspense, lazy } from 'react';
import { LucideProps } from 'lucide-react';

// Dynamically import icons to reduce initial bundle size
const iconCache = new Map<string, React.ComponentType<LucideProps>>();

const createLazyIcon = (iconName: string) => {
  if (iconCache.has(iconName)) {
    return iconCache.get(iconName)!;
  }

  const LazyIcon = lazy(() =>
    import('lucide-react').then((module) => ({
      default: (module as Record<string, React.ComponentType<LucideProps>>)[iconName]
    }))
  );

  iconCache.set(iconName, LazyIcon as React.ComponentType<LucideProps>);
  return LazyIcon as React.ComponentType<LucideProps>;
};

interface OptimizedIconProps extends LucideProps {
  name: string;
}

const OptimizedIconComponent: React.FC<OptimizedIconProps> = ({ name, ...props }) => {
  const IconComponent = createLazyIcon(name);

  return (
    <Suspense fallback={<div className="w-4 h-4 bg-gray-200 rounded animate-pulse" />}>
      <IconComponent {...props} />
    </Suspense>
  );
};

// Memoize to prevent unnecessary re-renders
export const OptimizedIcon = memo(OptimizedIconComponent);

// Pre-defined commonly used icons for better performance
export const CommonIcons = {
  Loading: memo(() => (
    <div className="w-4 h-4 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
  )),
  
  ArrowRight: memo(({ className = "w-4 h-4", ...props }: LucideProps) => (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 5l7 7-7 7"
      />
    </svg>
  )),
  
  Check: memo(({ className = "w-4 h-4", ...props }: LucideProps) => (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M5 13l4 4L19 7"
      />
    </svg>
  )),
  
  X: memo(({ className = "w-4 h-4", ...props }: LucideProps) => (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M6 18L18 6M6 6l12 12"
      />
    </svg>
  )),
  
  ChevronDown: memo(({ className = "w-4 h-4", ...props }: LucideProps) => (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M19 9l-7 7-7-7"
      />
    </svg>
  )),
  
  User: memo(({ className = "w-4 h-4", ...props }: LucideProps) => (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
      />
    </svg>
  ))
};

// Set display names for debugging
CommonIcons.Loading.displayName = 'LoadingIcon';
CommonIcons.ArrowRight.displayName = 'ArrowRightIcon';
CommonIcons.Check.displayName = 'CheckIcon';
CommonIcons.X.displayName = 'XIcon';
CommonIcons.ChevronDown.displayName = 'ChevronDownIcon';
CommonIcons.User.displayName = 'UserIcon';

OptimizedIcon.displayName = 'OptimizedIcon';