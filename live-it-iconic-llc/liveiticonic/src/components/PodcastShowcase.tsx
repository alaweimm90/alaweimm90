import { Play, Youtube } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { CanvasBackground } from './CanvasBackground';
import { PerspectiveCard } from './PerspectiveCard';
import { PODCAST_PERSPECTIVES } from '@/constants/podcastPerspectives';

const PodcastShowcase = () => {

  return (
    <section className="py-16 md:py-20 bg-lii-bg relative overflow-hidden">
      <CanvasBackground />

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-6xl mx-auto">
          {/* Glass Header Container */}
          <div className="text-center mb-10 max-w-4xl mx-auto">
            <div className="backdrop-blur-xl bg-lii-bg/40 border border-lii-gold/20 rounded-2xl p-6 md:p-8 shadow-[0_8px_32px_rgba(193,160,96,0.12)]">
              <div className="inline-flex items-center gap-2 mb-3">
                <Youtube className="w-6 h-6 text-lii-gold" />
                <span className="text-xs font-ui tracking-widest uppercase text-lii-gold">
                  Live Weekly
                </span>
              </div>

              <h2 className="text-3xl sm:text-4xl md:text-5xl font-display tracking-tight mb-3">
                <span className="bg-gradient-to-r from-lii-cloud via-lii-gold to-lii-cloud bg-clip-text text-transparent">
                  The Iconic Perspectives
                </span>
              </h2>

              <div className="w-16 h-px bg-gradient-to-r from-transparent via-lii-gold to-transparent mx-auto mb-3"></div>

              <p className="text-base sm:text-lg font-ui text-lii-ash max-w-2xl mx-auto leading-relaxed">
                Weekly conversations exploring excellence from three distinct viewpoints.
              </p>
            </div>
          </div>

          {/* Video Showcase with elegant frame */}
          <div className="mb-16 relative">
            {/* Animated glow effect */}
            <div className="absolute -inset-1 bg-gradient-to-r from-lii-gold/20 via-lii-gold/10 to-lii-gold/20 rounded-2xl blur-xl"></div>

            <div className="relative rounded-2xl overflow-hidden border border-lii-gold/30 bg-gradient-to-br from-lii-charcoal/30 to-lii-ink/50 backdrop-blur-sm shadow-[0_0_40px_rgba(193,160,96,0.15)]">
              <div className="aspect-video relative bg-gradient-to-br from-lii-ink/80 to-lii-charcoal/60">
                {/* Animated scan line effect */}
                <div className="absolute inset-0 overflow-hidden">
                  <div className="absolute w-full h-px bg-gradient-to-r from-transparent via-lii-gold/30 to-transparent animate-[scan_3s_ease-in-out_infinite]"></div>
                </div>

                {/* Placeholder for YouTube embed */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="w-20 h-20 rounded-full bg-lii-gold/10 border-2 border-lii-gold flex items-center justify-center mb-6 group cursor-pointer hover:bg-lii-gold/20 hover:scale-110 transition-all duration-300 shadow-[0_0_30px_rgba(193,160,96,0.3)]">
                    <Play className="w-10 h-10 text-lii-gold ml-1" />
                  </div>
                  <p className="text-lii-ash font-ui text-sm tracking-wider">Latest Episode</p>
                </div>

                {/* Subtle corner accents */}
                <div className="absolute top-4 left-4 w-12 h-12 border-l-2 border-t-2 border-lii-gold/30"></div>
                <div className="absolute top-4 right-4 w-12 h-12 border-r-2 border-t-2 border-lii-gold/30"></div>
                <div className="absolute bottom-4 left-4 w-12 h-12 border-l-2 border-b-2 border-lii-gold/30"></div>
                <div className="absolute bottom-4 right-4 w-12 h-12 border-r-2 border-b-2 border-lii-gold/30"></div>
              </div>

              {/* Episode Info Bar */}
              <div className="px-8 py-6 border-t border-lii-gold/10">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <div>
                    <h3 className="text-lg font-display font-light text-lii-cloud mb-1">
                      Redefining Success
                    </h3>
                    <p className="text-sm text-lii-ash font-ui font-light">
                      Episode 12 · 45 min · Three perspectives on what it means to truly succeed
                    </p>
                  </div>
                  <Button variant="primary" className="flex items-center gap-2 font-ui font-medium">
                    <Youtube className="w-4 h-4" />
                    Subscribe
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Three Perspectives */}
          <div className="grid md:grid-cols-3 gap-8">
            {PODCAST_PERSPECTIVES.map((perspective, index) => (
              <PerspectiveCard key={index} perspective={perspective} />
            ))}
          </div>

          {/* CTA Section */}
          <div className="text-center mt-16">
            <div className="inline-block">
              <p className="text-lii-ash font-ui font-light mb-6">
                New episodes every week. Real conversations. Zero fluff.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button
                  variant="primary"
                  size="lg"
                  className="flex items-center gap-2 font-ui font-medium"
                >
                  <Youtube className="w-5 h-5" />
                  Watch on YouTube
                </Button>
                <Button variant="secondary" size="lg" className="font-ui font-medium">
                  View All Episodes
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PodcastShowcase;
