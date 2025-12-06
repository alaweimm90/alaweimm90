import athleteRunning from '@/assets/athlete-running.png';
import { useEffect, useRef } from 'react';

const AthleteShowcase = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const imageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-fade-in-up');
          }
        });
      },
      { threshold: 0.1 }
    );

    if (imageRef.current) {
      observer.observe(imageRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="py-32 bg-gradient-to-br from-background via-lii-ink/30 to-background relative overflow-hidden"
    >
      {/* Background Effects */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-gradient-to-br from-lii-gold to-lii-copper rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/4 w-48 h-48 bg-gradient-to-br from-lii-copper to-lii-crimson rounded-full blur-2xl animate-pulse animate-delay-1000"></div>
      </div>

      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 items-center max-w-7xl mx-auto">
          {/* Left - Athlete Image */}
          <div ref={imageRef} className="relative opacity-0">
            <div className="relative group">
              {/* Main Image Container */}
              <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-lii-gold/20 via-lii-copper/15 to-lii-crimson/10 p-8">
                <img
                  src={athleteRunning}
                  alt="Professional athlete running on track with focused expression and dynamic motion blur"
                  width="800"
                  height="600"
                  className="w-full h-auto object-cover rounded-xl supercar-filter group-hover:scale-105 transition-all duration-1000 ease-out"
                />

                {/* Overlay Effects */}
                <div className="absolute inset-0 bg-gradient-to-t from-lii-gold/30 via-transparent to-lii-copper/20 opacity-0 group-hover:opacity-100 transition-all duration-700 rounded-2xl"></div>

                {/* Floating Elements */}
                <div className="absolute -top-4 -right-4 w-16 h-16 border-2 border-lii-gold/40 rotate-45 group-hover:rotate-90 transition-transform duration-1000"></div>
                <div className="absolute -bottom-6 -left-6 w-12 h-12 bg-gradient-to-br from-lii-copper to-lii-crimson rounded-full opacity-60 group-hover:scale-150 transition-all duration-800"></div>
              </div>

              {/* Glow Effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-lii-gold/20 to-lii-copper/20 rounded-2xl blur-xl scale-110 opacity-0 group-hover:opacity-70 transition-all duration-700 -z-10"></div>
            </div>
          </div>

          {/* Right - Content */}
          <div className="space-y-8">
            <div>
              <h2 className="text-4xl md:text-6xl font-display font-light tracking-tight mb-8 leading-tight">
                <span className="text-gradient">Built by athletes.</span>
                <br />
                <span className="text-foreground">Worn by legends.</span>
              </h2>
            </div>

            <div className="space-y-6 text-lg font-ui font-light text-foreground/80 leading-relaxed">
              <p>
                Training begins at 4:30 AM. Competition weight: 185 lbs. Fabric tested in -10°F to
                105°F conditions.
              </p>

              <p>
                Cotton-elastane blend provides 15% stretch recovery. Moisture-wicking finish reduces
                drying time by 40%.
              </p>
            </div>

            {/* Performance Stats */}
            <div className="grid grid-cols-3 gap-8 pt-8 border-t border-lii-gold/20">
              <div className="text-center group">
                <div className="text-3xl font-display font-light text-lii-gold mb-2 group-hover:text-lii-champagne transition-colors duration-500">
                  4°C
                </div>
                <div className="text-sm font-ui font-medium text-foreground/60 tracking-wide uppercase">
                  Min Operating
                </div>
              </div>
              <div className="text-center group">
                <div className="text-3xl font-display font-light text-lii-champagne mb-2 group-hover:text-lii-bronze transition-colors duration-500">
                  48hr
                </div>
                <div className="text-sm font-ui font-medium text-foreground/60 tracking-wide uppercase">
                  Dry Time
                </div>
              </div>
              <div className="text-center group">
                <div className="text-3xl font-display font-light text-lii-bronze mb-2 group-hover:text-lii-gold transition-colors duration-500">
                  200+
                </div>
                <div className="text-sm font-ui font-medium text-foreground/60 tracking-wide uppercase">
                  Wash Cycles
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AthleteShowcase;
