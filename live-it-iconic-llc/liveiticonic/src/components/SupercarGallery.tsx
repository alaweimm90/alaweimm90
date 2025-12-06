import coastalRoadImage from '@/assets/collection-banner-coastal-road.jpg';
import lifestyleImage from '@/assets/lifestyle-cobblestone-supercar.jpg';
import redSupercarImage from '@/assets/red-supercar-coastal-road.jpg';
import supercarsImage from '@/assets/supercars-mountain-valley.jpg';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Play } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
interface GalleryItem {
  id: string;
  image: string;
  title: string;
  subtitle: string;
  description: string;
  alt: string;
  isVideo?: boolean;
}
const galleryItems: GalleryItem[] = [
  {
    id: '1',
    image: supercarsImage,
    title: 'Mountain Precision',
    subtitle: 'Alpine Engineering',
    description:
      'Where peak performance meets breathtaking landscapes. Every curve designed for excellence.',
    alt: 'Three modern supercars parked on mountain overlook with dramatic valley and peak views in background',
  },
  {
    id: '2',
    image: coastalRoadImage,
    title: 'Coastal Velocity',
    subtitle: 'Ocean Drive Collection',
    description: 'Inspired by endless horizons and the freedom of the open road.',
    alt: 'Red supercar driving along coastal road with ocean waves and cliffs visible in background',
  },
  {
    id: '3',
    image: redSupercarImage,
    title: 'Urban Legend',
    subtitle: 'City Performance',
    description: 'Dominating cityscapes with sophisticated power and timeless design.',
    alt: 'Black supercar parked in urban setting with modern city skyline and architecture in background',
  },
  {
    id: '4',
    image: lifestyleImage,
    title: 'Heritage Lines',
    subtitle: 'Classic Meets Modern',
    description: 'Timeless elegance reimagined for the contemporary athlete.',
    alt: 'Classic supercar parked on cobblestone street with historic European architecture and buildings',
  },
];
const SupercarGallery = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);
  const sectionRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!isAutoPlaying) return;
    const interval = setInterval(() => {
      setCurrentIndex(prev => (prev + 1) % galleryItems.length);
    }, 6000);
    return () => clearInterval(interval);
  }, [isAutoPlaying]);
  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-fade-in-up');
          }
        });
      },
      {
        threshold: 0.1,
      }
    );
    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }
    return () => observer.disconnect();
  }, []);
  const goToPrevious = () => {
    setCurrentIndex(prev => (prev - 1 + galleryItems.length) % galleryItems.length);
    setIsAutoPlaying(false);
  };
  const goToNext = () => {
    setCurrentIndex(prev => (prev + 1) % galleryItems.length);
    setIsAutoPlaying(false);
  };
  const currentItem = galleryItems[currentIndex];
  return (
    <section
      ref={sectionRef}
      className="py-16 md:py-20 bg-lii-ink relative overflow-hidden opacity-0"
    >
      {/* Subtle orb */}
      <div className="absolute inset-0 opacity-10">
        <div
          className="absolute top-1/2 right-1/4 w-64 h-64 bg-lii-gold rounded-full blur-[100px] animate-pulse"
          style={{
            animationDuration: '12s',
          }}
        ></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <h2 className="sr-only">Automotive inspiration gallery</h2>

        {/* Live region for screen readers */}
        <div aria-live="polite" aria-atomic="true" className="sr-only">
          Slide {currentIndex + 1} of {galleryItems.length}: {currentItem.title}
        </div>

        {/* Main Gallery */}
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-5 gap-8 items-center">
            {/* Left Content */}
            <div className="lg:col-span-2 space-y-8">
              <div>
                <h3 className="text-2xl sm:text-3xl font-display font-light text-engine-rev mb-4">
                  {currentItem.title}
                </h3>
                <h4 className="text-lg text-lii-gold/80 font-ui font-medium mb-6">
                  {currentItem.subtitle}
                </h4>
                <p className="text-foreground/80 font-ui leading-relaxed">
                  {currentItem.description}
                </p>
              </div>

              {/* Navigation */}
              <div className="flex items-center gap-4">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={goToPrevious}
                  aria-label="Previous slide"
                  className="w-12 h-12 bg-lii-ink/60 backdrop-blur-sm border-lii-gold/20 text-lii-gold hover:border-lii-gold/40 hover:bg-lii-gold/10"
                >
                  <ChevronLeft className="w-5 h-5" />
                </Button>

                <div className="flex gap-2">
                  {galleryItems.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => {
                        setCurrentIndex(index);
                        setIsAutoPlaying(false);
                      }}
                      aria-label={`Go to slide ${index + 1} of ${galleryItems.length}`}
                      aria-current={index === currentIndex ? 'true' : 'false'}
                      className={`w-2 h-8 rounded-full transition-all duration-300 ${
                        index === currentIndex
                          ? 'bg-lii-gold'
                          : 'bg-lii-gold/30 hover:bg-lii-gold/50'
                      }`}
                    />
                  ))}
                </div>

                <Button
                  variant="outline"
                  size="icon"
                  onClick={goToNext}
                  aria-label="Next slide"
                  className="w-12 h-12 bg-lii-ink/60 backdrop-blur-sm border-lii-gold/20 text-lii-gold hover:border-lii-gold/40 hover:bg-lii-gold/10"
                >
                  <ChevronRight className="w-5 h-5" />
                </Button>
              </div>

              {/* CTA Button */}
              <div className="pt-4">
                <Button
                  size="lg"
                  className="luxury-button text-lii-black text-base px-8 py-3 font-ui font-medium tracking-wide"
                >
                  Explore Collection
                </Button>
              </div>
            </div>

            {/* Right Image */}
            <div className="lg:col-span-3">
              <div className="relative group">
                <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-lii-gold/10 via-transparent to-lii-champagne/10 p-1">
                  <img
                    src={currentItem.image}
                    alt={currentItem.alt}
                    width="800"
                    height="500"
                    className="w-full h-[400px] sm:h-[500px] object-cover rounded-xl group-hover:scale-[1.02] transition-all duration-1000 ease-out"
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

                  {/* Video Play Overlay */}
                  {currentItem.isVideo && (
                    <div className="absolute inset-1 flex items-center justify-center bg-lii-black/20 rounded-xl opacity-0 group-hover:opacity-100 transition-all duration-500">
                      <div className="w-16 h-16 bg-lii-gold/90 rounded-full flex items-center justify-center backdrop-blur-sm">
                        <Play className="w-6 h-6 text-lii-black ml-1" />
                      </div>
                    </div>
                  )}

                  {/* Elegant Gradient Overlay */}
                  <div className="absolute inset-1 bg-gradient-to-t from-lii-black/30 via-transparent to-transparent rounded-xl"></div>
                </div>

                {/* Accent Elements */}
                <div className="absolute -bottom-4 -left-4 w-8 h-8 border border-lii-gold/40 rotate-45 group-hover:rotate-90 transition-transform duration-1000"></div>
                <div className="absolute -top-4 -right-4 w-6 h-6 bg-lii-champagne/20 rounded-full group-hover:scale-125 transition-transform duration-1000"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Auto-play Status */}
        <div className="flex justify-center mt-12">
          <button
            onClick={() => setIsAutoPlaying(!isAutoPlaying)}
            className={`text-sm font-ui transition-colors duration-300 ${
              isAutoPlaying ? 'text-lii-gold' : 'text-foreground/40 hover:text-foreground/60'
            }`}
          >
            {isAutoPlaying ? 'Auto-playing gallery' : 'Gallery paused'}
          </button>
        </div>
      </div>
    </section>
  );
};
export default SupercarGallery;
