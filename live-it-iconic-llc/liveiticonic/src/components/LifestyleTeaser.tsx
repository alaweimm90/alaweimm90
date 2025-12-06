import coastalRoadImage from '@/assets/collection-banner-coastal-road.jpg';
import lifestyleCobblestoneSupercarImage from '@/assets/lifestyle-cobblestone-supercar.jpg';
import supercarsValleyImage from '@/assets/supercars-mountain-valley.jpg';
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';
import { useEffect, useRef } from 'react';

/**
 * LifestyleTeaser component displays an image gallery with staggered animation
 *
 * Shows three high-quality lifestyle/automotive images in a grid layout with IntersectionObserver-triggered
 * stagger animation. Includes a call-to-action button to explore the lifestyle section.
 *
 * @component
 *
 * @example
 * <LifestyleTeaser />
 *
 * @remarks
 * - Uses IntersectionObserver to trigger animations on scroll
 * - Images load from assets folder
 * - Stagger animation applied at 0.2 threshold visibility
 * - Responsive grid layout adapts to screen size
 * - Includes overlay text and call-to-action button
 */
const LifestyleTeaser = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const galleryRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-stagger');
          }
        });
      },
      { threshold: 0.2 }
    );

    if (galleryRef.current) {
      observer.observe(galleryRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} className="py-16 md:py-20 bg-lii-bg relative overflow-hidden">
      {/* Subtle orb */}
      <div className="absolute inset-0 opacity-10">
        <div
          className="absolute top-1/2 left-1/4 w-64 h-64 bg-lii-gold rounded-full blur-[100px] animate-pulse"
          style={{ animationDuration: '14s' }}
        ></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Glass Header Container */}
        <div className="text-center mb-12 max-w-4xl mx-auto">
          <div className="backdrop-blur-xl bg-lii-ink/30 border border-lii-gold/20 rounded-2xl p-10 md:p-12 shadow-[0_8px_32px_rgba(193,160,96,0.1)]">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-display tracking-tight mb-4">
              <span className="bg-gradient-to-r from-lii-cloud via-lii-gold to-lii-cloud bg-clip-text text-transparent">
                The Philosophy
              </span>
            </h2>
            <div className="w-16 h-px bg-gradient-to-r from-transparent via-lii-gold to-transparent mx-auto mb-4"></div>
            <p className="text-lg sm:text-xl font-ui text-lii-ash max-w-2xl mx-auto leading-relaxed">
              Beyond performance apparel. Embracing a mindset where excellence is lived every day.
            </p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-16 items-center max-w-7xl mx-auto mb-20">
          {/* Content Side */}
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl sm:text-3xl font-display font-light text-engine-rev mb-6">
                Live at the Apex
              </h3>
              <div className="space-y-6 text-lg font-ui font-light text-foreground/80 leading-relaxed">
                <p>
                  From dawn training sessions to evening galas. From track days to boardroom
                  presentations. Live It Iconic moves seamlessly between worlds of performance and
                  sophistication.
                </p>
                <p>
                  This isn't just what you wear. It's how you approach every challenge, every
                  opportunity, every moment that matters.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <Button
                size="lg"
                className="luxury-button text-lii-black text-base px-8 py-3 font-ui font-medium tracking-wide group"
              >
                Explore Collection
                <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-300" />
              </Button>

              <div className="text-sm font-ui text-foreground/50">Discover your aesthetic</div>
            </div>
          </div>

          {/* Image Showcase */}
          <div ref={galleryRef} className="relative">
            {/* Main Featured Image */}
            <div className="relative group mb-6">
              <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-lii-gold/10 via-transparent to-lii-champagne/10 p-1">
                <img
                  src={supercarsValleyImage}
                  alt="Three luxury supercars parked on mountain valley overlook with dramatic peaks and scenic views"
                  width="800"
                  height="350"
                  className="w-full h-[350px] object-cover rounded-xl group-hover:scale-[1.02] transition-all duration-1000 ease-out"
                  style={{
                    filter:
                      'grayscale(0.2) contrast(1.25) brightness(0.88) saturate(0.85) sepia(0.12) hue-rotate(15deg)',
                    transition: 'filter 0.7s ease',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.filter =
                      'grayscale(0.15) contrast(1.28) brightness(0.92) saturate(0.9) sepia(0.15) hue-rotate(18deg)';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.filter =
                      'grayscale(0.2) contrast(1.25) brightness(0.88) saturate(0.85) sepia(0.12) hue-rotate(15deg)';
                  }}
                />

                {/* Subtle Overlay */}
                <div className="absolute inset-1 bg-gradient-to-t from-lii-black/40 via-transparent to-transparent rounded-xl"></div>
              </div>

              {/* Elegant Accents */}
              <div className="absolute -bottom-3 -right-3 w-6 h-6 border border-lii-gold/60 rotate-45 group-hover:rotate-90 transition-transform duration-1000"></div>
            </div>

            {/* Secondary Images Grid */}
            <div className="grid grid-cols-2 gap-4">
              <div className="relative group">
                <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-lii-champagne/10 via-transparent to-lii-gold/10 p-1">
                  <img
                    src={coastalRoadImage}
                    alt="Red luxury supercar driving along coastal cliff road with ocean waves crashing below"
                    width="400"
                    height="128"
                    className="w-full h-32 object-cover rounded-lg group-hover:scale-105 transition-all duration-700"
                    style={{
                      filter:
                        'grayscale(0.25) contrast(1.28) brightness(0.86) saturate(0.8) sepia(0.15) hue-rotate(18deg)',
                      transition: 'filter 0.7s ease',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.filter =
                        'grayscale(0.18) contrast(1.3) brightness(0.9) saturate(0.88) sepia(0.18) hue-rotate(20deg)';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.filter =
                        'grayscale(0.25) contrast(1.28) brightness(0.86) saturate(0.8) sepia(0.15) hue-rotate(18deg)';
                    }}
                  />
                </div>
              </div>

              <div className="relative group">
                <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-lii-gold/10 via-transparent to-lii-champagne/10 p-1">
                  <img
                    src={lifestyleCobblestoneSupercarImage}
                    alt="Classic supercar parked on historic cobblestone street with European architecture in background"
                    width="400"
                    height="128"
                    className="w-full h-32 object-cover rounded-lg group-hover:scale-105 transition-all duration-700"
                    style={{
                      filter:
                        'grayscale(0.25) contrast(1.26) brightness(0.87) saturate(0.82) sepia(0.14) hue-rotate(17deg)',
                      transition: 'filter 0.7s ease',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.filter =
                        'grayscale(0.2) contrast(1.28) brightness(0.9) saturate(0.88) sepia(0.16) hue-rotate(19deg)';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.filter =
                        'grayscale(0.25) contrast(1.26) brightness(0.87) saturate(0.82) sepia(0.14) hue-rotate(17deg)';
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Call to Action Section with luxury quote styling */}
        <div className="text-center">
          <div className="max-w-2xl mx-auto relative px-8">
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-px h-16 bg-gradient-to-b from-transparent via-lii-gold/50 to-transparent"></div>
            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-px h-16 bg-gradient-to-b from-transparent via-lii-gold/50 to-transparent"></div>
            <p className="text-lg font-ui font-light text-lii-gold/80 italic mb-8 relative">
              <span className="text-lii-gold/40 text-3xl absolute -left-4 -top-2">"</span>
              Excellence isn't a destination. It's a way of traveling.
              <span className="text-lii-gold/40 text-3xl absolute -right-4 -bottom-4">"</span>
            </p>
            <div className="w-16 h-px bg-gradient-to-r from-transparent via-lii-gold/60 to-transparent mx-auto shadow-[0_0_8px_rgba(193,160,96,0.4)]"></div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LifestyleTeaser;
