import { useState, useEffect } from 'react';
import { ElegantFlamingo, MajesticPelican, CaribbeanFrigateBird, TropicalTanager } from './logo';
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from './ui/carousel';

interface BirdDesign {
  name: string;
  component: React.ComponentType<{ size?: number; color?: string; className?: string }>;
  characteristics: string[];
  month: number; // Featured month (1-12)
}

const birdDesigns: BirdDesign[] = [
  {
    name: 'Elegant Flamingo',
    component: ElegantFlamingo,
    characteristics: ['Grace', 'Refinement', 'Iconic Silhouette'],
    month: 1, // January-March
  },
  {
    name: 'Majestic Pelican',
    component: MajesticPelican,
    characteristics: ['Strength', 'Endurance', 'Bold Presence'],
    month: 4, // April-June
  },
  {
    name: 'Caribbean Frigate Bird',
    component: CaribbeanFrigateBird,
    characteristics: ['Freedom', 'Power', 'Dynamic Energy'],
    month: 7, // July-September
  },
  {
    name: 'Tropical Tanager',
    component: TropicalTanager,
    characteristics: ['Vibrancy', 'Exclusivity', 'Tropical Spirit'],
    month: 10, // October-December
  },
];

interface BirdLogoShowcaseProps {
  variant?: 'full' | 'compact' | 'carousel';
  className?: string;
}

export const BirdLogoShowcase: React.FC<BirdLogoShowcaseProps> = ({
  variant = 'full',
  className = '',
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [featuredBird, setFeaturedBird] = useState<BirdDesign | null>(null);

  // Determine featured bird based on current month
  useEffect(() => {
    const currentMonth = new Date().getMonth() + 1; // 1-12
    const featured = birdDesigns.find(bird => {
      if (bird.month <= 3) return currentMonth <= 3;
      if (bird.month <= 6) return currentMonth > 3 && currentMonth <= 6;
      if (bird.month <= 9) return currentMonth > 6 && currentMonth <= 9;
      return currentMonth > 9;
    });
    setFeaturedBird(featured || birdDesigns[0]);
  }, []);

  // Compact variant for navigation
  if (variant === 'compact') {
    const Bird = featuredBird?.component || ElegantFlamingo;
    console.log('BirdLogoShowcase compact rendering:', { featuredBird: featuredBird?.name, Bird });
    return (
      <div
        className={`group relative ${className}`}
        onMouseEnter={() => setHoveredIndex(0)}
        onMouseLeave={() => setHoveredIndex(null)}
      >
        <div className="transition-all duration-500 ease-out-quart group-hover:scale-110 group-hover:rotate-6 group-hover:drop-shadow-[0_0_20px_rgba(193,160,96,0.7)]">
          <Bird size={48} className="text-lii-gold animate-fade-in-up" />
        </div>

        {/* Hover tooltip */}
        {hoveredIndex === 0 && (
          <div className="absolute top-full left-1/2 -translate-x-1/2 mt-4 z-50 animate-fade-in-up pointer-events-none">
            <div className="bg-lii-ink/95 backdrop-blur-xl border border-lii-gold/30 rounded-xl px-6 py-4 shadow-[0_8px_32px_rgba(212,175,55,0.4)] min-w-[220px]">
              <p className="text-lii-gold font-display text-lg mb-2 tracking-wide">
                {featuredBird?.name}
              </p>
              <p className="text-lii-ash text-sm mb-3 font-ui">Featured this season</p>
              <div className="flex flex-wrap gap-2">
                {featuredBird?.characteristics.map((char, idx) => (
                  <span
                    key={idx}
                    className="text-xs px-3 py-1 bg-lii-gold/10 text-lii-gold rounded-full border border-lii-gold/20 font-ui"
                  >
                    {char}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Carousel variant
  if (variant === 'carousel') {
    return (
      <div className={`w-full max-w-md mx-auto ${className}`}>
        <h3 className="text-2xl font-display text-lii-gold text-center mb-8 tracking-wide">
          Featured Birds
        </h3>
        <Carousel
          opts={{
            align: 'center',
            loop: true,
          }}
          className="w-full"
        >
          <CarouselContent>
            {birdDesigns.map((bird, index) => {
              const Bird = bird.component;
              return (
                <CarouselItem key={index}>
                  <div className="p-8">
                    <div className="group flex flex-col items-center gap-6 bg-lii-ink/50 backdrop-blur-xl border border-lii-gold/20 rounded-2xl p-8 transition-all duration-500 hover:scale-105 hover:border-lii-gold/40 hover:shadow-[0_8px_48px_rgba(212,175,55,0.3)]">
                      <div className="transition-all duration-500 group-hover:scale-110 group-hover:rotate-3 group-hover:drop-shadow-[0_0_24px_rgba(193,160,96,0.6)]">
                        <Bird size={120} className="text-lii-gold animate-scale-in" />
                      </div>
                      <div className="text-center">
                        <h4 className="text-2xl font-display text-lii-gold mb-3 tracking-wide transition-all duration-300 group-hover:scale-110">
                          {bird.name}
                        </h4>
                        <div className="flex flex-wrap justify-center gap-2">
                          {bird.characteristics.map((char, idx) => (
                            <span
                              key={idx}
                              className="text-sm px-4 py-2 bg-lii-gold/10 text-lii-gold rounded-full border border-lii-gold/30 font-ui tracking-wide transition-all duration-300 hover:bg-lii-gold/20 hover:border-lii-gold/50"
                            >
                              {char}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </CarouselItem>
              );
            })}
          </CarouselContent>
          <CarouselPrevious className="text-lii-gold border-lii-gold/30 hover:bg-lii-gold/10" />
          <CarouselNext className="text-lii-gold border-lii-gold/30 hover:bg-lii-gold/10" />
        </Carousel>
      </div>
    );
  }

  // Full showcase variant
  return (
    <div className={`w-full ${className}`}>
      <div className="text-center mb-12 animate-fade-in-up">
        <h2 className="text-4xl md:text-5xl font-display text-lii-gold mb-4 tracking-wide">
          Caribbean Bird Collection
        </h2>
        <p className="text-lii-ash text-lg font-ui max-w-2xl mx-auto">
          Four iconic designs representing the spirit of Live It Iconic. Each bird embodies luxury,
          performance, and Caribbean elegance.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
        {birdDesigns.map((bird, index) => {
          const Bird = bird.component;
          const isFeatured = featuredBird?.name === bird.name;

          return (
            <div
              key={index}
              className="group relative"
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
              style={{
                animationDelay: `${index * 150}ms`,
              }}
            >
              <div
                className={`relative bg-lii-ink/50 backdrop-blur-xl border rounded-2xl p-8 transition-all duration-500 ease-out-quart hover:scale-105 hover:rotate-1 animate-fade-in-up ${
                  isFeatured
                    ? 'border-lii-gold/60 shadow-[0_8px_48px_rgba(212,175,55,0.4)] animate-pulse'
                    : 'border-lii-gold/20 hover:border-lii-gold/40 hover:shadow-[0_8px_48px_rgba(212,175,55,0.3)]'
                }`}
              >
                {/* Featured badge */}
                {isFeatured && (
                  <div className="absolute -top-3 -right-3 bg-lii-gold text-lii-bg px-4 py-1 rounded-full text-xs font-ui font-semibold tracking-wider shadow-[0_4px_16px_rgba(212,175,55,0.5)] animate-bounce">
                    FEATURED
                  </div>
                )}

                {/* Bird logo */}
                <div className="flex justify-center mb-6 transition-all duration-500 ease-out-quart group-hover:scale-110 group-hover:rotate-3 group-hover:drop-shadow-[0_0_28px_rgba(193,160,96,0.7)]">
                  <Bird size={96} className="text-lii-gold" />
                </div>

                {/* Bird info */}
                <div className="text-center">
                  <h3 className="text-xl font-display text-lii-gold mb-3 tracking-wide transition-all duration-300 group-hover:scale-110">
                    {bird.name}
                  </h3>

                  {/* Characteristics - shown on hover */}
                  <div
                    className={`flex flex-col gap-2 transition-all duration-500 ease-out-quart ${
                      hoveredIndex === index
                        ? 'opacity-100 max-h-40 translate-y-0'
                        : 'opacity-0 max-h-0 -translate-y-4 pointer-events-none'
                    }`}
                  >
                    {bird.characteristics.map((char, idx) => (
                      <span
                        key={idx}
                        style={{ animationDelay: `${idx * 50}ms` }}
                        className="text-sm px-4 py-2 bg-lii-gold/10 text-lii-gold rounded-full border border-lii-gold/30 font-ui tracking-wide animate-fade-in-up hover:bg-lii-gold/20 hover:border-lii-gold/50 hover:scale-105 transition-all duration-300"
                      >
                        {char}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Featured bird callout */}
      {featuredBird && (
        <div className="mt-16 text-center animate-fade-in-up" style={{ animationDelay: '600ms' }}>
          <p className="text-lii-ash text-base font-ui">
            <span className="text-lii-gold font-semibold">{featuredBird.name}</span> is featured
            this season
          </p>
        </div>
      )}
    </div>
  );
};

export default BirdLogoShowcase;
