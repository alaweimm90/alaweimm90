import { ReactNode, useState, useEffect } from 'react';
import {
  Skeleton,
  HeroSkeleton,
  CollectionGridSkeleton,
  FeaturedCollectionsSkeleton,
} from '@/components/ui/skeleton';

interface LoadingWrapperProps {
  children: ReactNode;
  isLoading?: boolean;
  skeletonType?: 'hero' | 'collection' | 'featured' | 'default';
  delay?: number;
}

const LoadingWrapper = ({
  children,
  isLoading = false,
  skeletonType = 'default',
  delay = 0,
}: LoadingWrapperProps) => {
  const [showSkeleton, setShowSkeleton] = useState(false);

  useEffect(() => {
    if (isLoading) {
      const timer = setTimeout(() => setShowSkeleton(true), delay);
      return () => clearTimeout(timer);
    } else {
      setShowSkeleton(false);
    }
  }, [isLoading, delay]);

  if (!isLoading || !showSkeleton) {
    return <>{children}</>;
  }

  const renderSkeleton = () => {
    switch (skeletonType) {
      case 'hero':
        return <HeroSkeleton />;
      case 'collection':
        return <CollectionGridSkeleton />;
      case 'featured':
        return <FeaturedCollectionsSkeleton />;
      default:
        return (
          <div className="min-h-screen bg-lii-bg flex items-center justify-center">
            <div className="text-center space-y-4">
              <Skeleton className="h-12 w-64 mx-auto" />
              <Skeleton className="h-6 w-96 mx-auto" />
              <div className="flex justify-center gap-4">
                <Skeleton className="h-10 w-32" />
                <Skeleton className="h-10 w-32" />
              </div>
            </div>
          </div>
        );
    }
  };

  return renderSkeleton();
};

export default LoadingWrapper;
