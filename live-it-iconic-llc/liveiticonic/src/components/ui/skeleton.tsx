import { cn } from '@/lib/utils';

function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('animate-pulse rounded-md bg-lii-charcoal/20', className)} {...props} />
  );
}

// Product Card Skeleton
function ProductCardSkeleton() {
  return (
    <div className="bg-lii-ink border border-lii-gold/10 rounded-2xl overflow-hidden">
      <Skeleton className="aspect-[4/5] w-full" />
      <div className="p-6 space-y-4">
        <Skeleton className="h-6 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
        <div className="flex justify-between items-center">
          <Skeleton className="h-6 w-16" />
          <Skeleton className="h-10 w-24 rounded-lg" />
        </div>
      </div>
    </div>
  );
}

// Hero Section Skeleton
function HeroSkeleton() {
  return (
    <div className="relative min-h-screen flex items-center">
      <div className="container mx-auto px-6">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <Skeleton className="h-16 w-4/5 mx-auto" />
          <Skeleton className="h-6 w-3/5 mx-auto" />
          <Skeleton className="h-6 w-2/5 mx-auto" />
          <div className="flex justify-center gap-4 pt-8">
            <Skeleton className="h-12 w-32 rounded-lg" />
            <Skeleton className="h-12 w-32 rounded-lg" />
          </div>
        </div>
      </div>
      <Skeleton className="absolute inset-0 -z-10" />
    </div>
  );
}

// Collection Grid Skeleton
function CollectionGridSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
      {Array.from({ length: count }).map((_, i) => (
        <ProductCardSkeleton key={i} />
      ))}
    </div>
  );
}

// Featured Collections Skeleton
function FeaturedCollectionsSkeleton() {
  return (
    <div className="space-y-12">
      <div className="text-center space-y-4">
        <Skeleton className="h-12 w-80 mx-auto" />
        <Skeleton className="h-6 w-96 mx-auto" />
      </div>
      <CollectionGridSkeleton count={3} />
    </div>
  );
}

// Navigation Menu Skeleton
function NavigationSkeleton() {
  return (
    <div className="flex items-center space-x-8">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-6 w-20" />
      ))}
    </div>
  );
}

// Testimonial Skeleton
function TestimonialSkeleton() {
  return (
    <div className="bg-lii-ink border border-lii-gold/10 rounded-2xl p-8">
      <div className="flex items-start gap-4">
        <Skeleton className="w-12 h-12 rounded-full flex-shrink-0" />
        <div className="flex-1 space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-4 w-3/5" />
          <div className="flex items-center gap-2 pt-2">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-4 w-16" />
          </div>
        </div>
      </div>
    </div>
  );
}

// Blog Post Skeleton
function BlogPostSkeleton() {
  return (
    <div className="bg-lii-ink border border-lii-gold/10 rounded-2xl overflow-hidden">
      <Skeleton className="aspect-[16/9] w-full" />
      <div className="p-6 space-y-4">
        <Skeleton className="h-6 w-20" />
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
        <div className="flex items-center justify-between pt-4">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-8 w-24 rounded-lg" />
        </div>
      </div>
    </div>
  );
}

export {
  Skeleton,
  ProductCardSkeleton,
  HeroSkeleton,
  CollectionGridSkeleton,
  FeaturedCollectionsSkeleton,
  NavigationSkeleton,
  TestimonialSkeleton,
  BlogPostSkeleton,
};
