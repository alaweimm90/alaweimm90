import lifestyleCollectionImage from '@/assets/lifestyle-athlete-gullwing-car.jpg';
import performanceCollectionImage from '@/assets/red-supercar-coastal-road.jpg';
import streetCollectionImage from '@/assets/red-supercar-cobblestone-street.jpg';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

/**
 * FeaturedCollections component showcases three main product collections with background images
 *
 * Displays three featured collection cards (Street, Performance, Essentials) with high-quality
 * background images, collection titles, subtitles, and descriptions. Each card includes a
 * call-to-action button to shop the collection. Grid layout adapts responsively to screen size.
 *
 * @component
 *
 * @example
 * <FeaturedCollections />
 *
 * @remarks
 * - Collections are hardcoded with fixed image and text data
 * - Uses background-image CSS for card backgrounds
 * - Responsive grid layout (stacks on mobile)
 * - Each collection has a shop button with navigation
 * - Accessibility: Includes alt text for all background images
 */

const collections = [
  {
    title: 'Street',
    subtitle: 'Night',
    description: 'Urban essentials for the late hours',
    image: streetCollectionImage,
    alt: 'Red supercar with racing stripes navigating narrow European cobblestone streets with authentic urban atmosphere',
  },
  {
    title: 'Performance',
    subtitle: 'Training',
    description: 'Technical gear for peak moments',
    image: performanceCollectionImage,
    alt: 'High-performance red supercar positioned on scenic coastal mountain road showcasing speed and elegance',
  },
  {
    title: 'Essentials',
    subtitle: 'Layers',
    description: 'Timeless pieces for every day',
    image: lifestyleCollectionImage,
    alt: 'Athletic lifestyle scene featuring athlete training beside luxury gullwing supercar representing peak performance mindset',
  },
];

const FeaturedCollections = () => {
  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-4xl md:text-5xl font-display font-normal mb-4">Collections</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-ui">
            Clean lines. Focused fits. Built to move, finished to impress.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {collections.map((collection, index) => (
            <Card
              key={index}
              className="group cursor-pointer overflow-hidden border-0 bg-transparent transition-all"
              style={{ transition: 'transform var(--motion-enter) var(--ease)' }}
            >
              <div className="relative aspect-[3/4] overflow-hidden rounded-xl">
                <img
                  src={collection.image}
                  alt={collection.alt}
                  width="400"
                  height="533"
                  className="w-full h-full object-cover group-hover:scale-105"
                  style={{ transition: 'transform var(--motion-enter) var(--ease)' }}
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-lii-black/80 via-transparent to-transparent" />
                <div className="absolute bottom-6 left-6 right-6">
                  <h3 className="text-2xl font-display font-normal text-white mb-1">
                    {collection.title}
                  </h3>
                  {collection.subtitle && (
                    <p className="text-lg text-white/80 font-ui mb-2">{collection.subtitle}</p>
                  )}
                  <p className="text-white/70 mb-4 font-ui text-sm">{collection.description}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="btn-lii-ghost text-white border-white/20 hover:bg-white/10"
                  >
                    Explore
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturedCollections;
