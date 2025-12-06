import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import heroImage from '@/assets/hero-athlete-supercar.jpg';

/**
 * Hero component displays the full-height landing page hero section with animations
 *
 * Features a background image with gradient overlay, animated gold glow orbs, floating
 * geometric accents, and a centered call-to-action with staggered fade-in animations.
 * Includes shop and explore lifestyle buttons with premium visual effects.
 *
 * @component
 *
 * @example
 * <Hero />
 *
 * @remarks
 * - Full viewport height section with absolute positioning for background
 * - Includes multiple CSS animations: heroZoom, luxuryPulse, subtleFloat, sparkle, etc.
 * - Background image uses grayscale and contrast filters for luxury effect
 * - All decorative elements use aria-hidden for accessibility
 * - Staggered animation delays for sequential text entrance
 */
const Hero = () => {
  return (
    <section className="relative h-screen w-full overflow-hidden flex items-center justify-center">
      {/* Hero Image - Softer Filter */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat transition-transform duration-[20000ms] ease-out"
        style={{
          backgroundImage: `url(${heroImage})`,
          filter:
            'grayscale(0.15) contrast(1.15) brightness(0.9) saturate(0.9) sepia(0.08) hue-rotate(10deg)',
          transform: 'scale(1.05)',
          animation: 'heroZoom 20s ease-out forwards',
        }}
        role="img"
        aria-label="Athlete showcasing Live It Iconic apparel with luxury supercar in coastal mountain setting"
      >
        <div className="absolute inset-0 bg-gradient-to-b from-lii-bg/85 via-lii-bg/65 to-lii-bg/90"></div>
      </div>

      {/* Animated Luxury Glow Orbs */}
      <div className="absolute inset-0 opacity-25 pointer-events-none" aria-hidden="true">
        <div
          className="absolute top-1/4 left-1/3 w-[500px] h-[500px] bg-lii-gold rounded-full blur-[120px]"
          style={{
            animation: 'luxuryPulse 8s ease-in-out infinite',
          }}
        ></div>
        <div
          className="absolute bottom-1/3 right-1/3 w-[400px] h-[400px] bg-lii-gold/60 rounded-full blur-[100px]"
          style={{
            animation: 'luxuryPulse 10s ease-in-out infinite 3s',
          }}
        ></div>
      </div>

      {/* Floating Geometric Accents */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none opacity-15"
        aria-hidden="true"
      >
        <div
          className="absolute top-1/4 right-1/5 w-16 h-16 border border-lii-gold transform rotate-45"
          style={{
            animation: 'subtleFloat 8s ease-in-out infinite',
          }}
        ></div>
        <div
          className="absolute bottom-1/4 left-1/6 w-20 h-20 border border-lii-gold transform rotate-45"
          style={{
            animation: 'subtleFloat 10s ease-in-out infinite 4s',
          }}
        ></div>
        <div
          className="absolute top-1/2 left-1/4 w-2 h-2 bg-lii-gold rounded-full"
          style={{
            animation: 'sparkle 6s ease-in-out infinite',
          }}
        ></div>
        <div
          className="absolute top-1/3 right-1/3 w-2 h-2 bg-lii-gold rounded-full"
          style={{
            animation: 'sparkle 7s ease-in-out infinite 2s',
          }}
        ></div>
      </div>

      {/* Elegant Top Border Animation */}
      <div className="absolute top-0 left-0 right-0 h-px overflow-hidden z-20" aria-hidden="true">
        <div
          className="w-full h-full bg-gradient-to-r from-transparent via-lii-gold to-transparent"
          style={{
            animation: 'borderShimmer 8s ease-in-out infinite',
          }}
        ></div>
      </div>

      {/* Main Content - Centered with Staggered Animations */}
      <div className="relative z-10 w-full px-4 sm:px-6">
        <div className="max-w-5xl mx-auto text-center">
          {/* Decorative Top Accent */}
          <div
            className="w-32 h-px bg-gradient-to-r from-transparent via-lii-gold to-transparent mx-auto mb-16"
            style={{
              animation: 'fadeInScale 1s cubic-bezier(0.16, 1, 0.3, 1) 0.2s forwards',
              opacity: 0,
              boxShadow: '0 0 20px rgba(193, 160, 96, 0.5)',
            }}
          ></div>

          <h1
            className="text-7xl sm:text-8xl md:text-9xl font-display tracking-tight mb-10 leading-none relative"
            style={{
              animation: 'fadeInUp 1.4s cubic-bezier(0.16, 1, 0.3, 1) 0.4s forwards',
              opacity: 0,
            }}
          >
            <span className="relative inline-block">
              <span className="relative z-10 bg-gradient-to-br from-lii-cloud via-lii-cloud to-lii-ash bg-clip-text text-transparent animate-shimmerText">
                Live It Iconic
              </span>
              <div
                className="absolute inset-0 blur-3xl bg-lii-gold/30 -z-10"
                style={{
                  animation: 'luxuryPulse 4s ease-in-out infinite',
                }}
              ></div>
            </span>
          </h1>

          {/* Decorative Middle Accent with Quote Marks */}
          <div
            className="flex items-center justify-center gap-4 mb-10"
            style={{
              animation: 'fadeInScale 1s cubic-bezier(0.16, 1, 0.3, 1) 0.8s forwards',
              opacity: 0,
            }}
          >
            <div className="w-12 h-px bg-gradient-to-r from-transparent to-lii-gold/60"></div>
            <div className="flex gap-2">
              <div
                className="w-1.5 h-1.5 rounded-full bg-lii-gold"
                style={{
                  animation: 'pulse 3s ease-in-out infinite',
                }}
              ></div>
              <div
                className="w-1.5 h-1.5 rounded-full bg-lii-gold/70"
                style={{
                  animation: 'pulse 3s ease-in-out infinite 0.5s',
                }}
              ></div>
              <div
                className="w-1.5 h-1.5 rounded-full bg-lii-gold/50"
                style={{
                  animation: 'pulse 3s ease-in-out infinite 1s',
                }}
              ></div>
            </div>
            <div className="w-12 h-px bg-gradient-to-l from-transparent to-lii-gold/60"></div>
          </div>

          <p
            className="text-2xl sm:text-3xl md:text-4xl font-ui text-lii-cloud mb-6 max-w-3xl mx-auto leading-relaxed relative"
            style={{
              animation: 'fadeInUp 1.4s cubic-bezier(0.16, 1, 0.3, 1) 1s forwards',
              opacity: 0,
            }}
          >
            <span className="relative inline-block hover:text-lii-gold transition-colors duration-500">
              Designed for motion; refined for nights out
            </span>
          </p>

          <p
            className="text-lg sm:text-xl md:text-2xl font-ui text-lii-cloud mb-14 max-w-3xl mx-auto leading-relaxed relative"
            style={{
              animation: 'fadeInUp 1.4s cubic-bezier(0.16, 1, 0.3, 1) 1.1s forwards',
              opacity: 0,
            }}
          >
            Precision-cut apparel inspired by automotive discipline
          </p>

          <div
            className="flex flex-col sm:flex-row justify-center items-center gap-5 mt-8"
            style={{
              animation: 'fadeInUp 1.4s cubic-bezier(0.16, 1, 0.3, 1) 1.3s forwards',
              opacity: 0,
            }}
          >
            <Button
              variant="primary"
              size="lg"
              asChild
              className="group relative overflow-hidden text-base px-10 h-14"
            >
              <Link to="/shop">
                <span className="relative z-10">Shop the Drop</span>
                <div className="absolute inset-0 bg-lii-gold-press transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left -z-10"></div>
              </Link>
            </Button>
            <Button variant="secondary" size="lg" asChild className="group text-base px-10 h-14">
              <Link to="/lifestyle">
                <span className="relative">Explore lifestyle</span>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};
export default Hero;
