import { Button } from '@/components/ui/button';
import { X } from 'lucide-react';
import { useState } from 'react';

// Import story images
import coastalImage from '@/assets/red-supercar-coastal-road.jpg';
import urbanImage from '@/assets/red-supercar-cobblestone-street.jpg';
import mountainImage from '@/assets/supercars-mountain-valley.jpg';

interface Story {
  id: string;
  title: string;
  subtitle: string;
  image: string;
  alt: string;
  content: string;
  category: string;
}

const stories: Story[] = [
  {
    id: 'dawn-sessions',
    title: 'DAWN SESSIONS',
    subtitle: 'First light, first move',
    image: mountainImage,
    alt: 'Three supercars parked on mountain valley road with dramatic sunrise lighting and scenic peaks',
    content: `The silence before dawn holds possibility. Each morning presents a choice—settle for ordinary or demand exceptional performance from yourself and everything around you.`,
    category: 'Performance',
  },
  {
    id: 'midnight-canyon',
    title: 'MIDNIGHT CANYON',
    subtitle: 'When the city sleeps, we move',
    image: coastalImage,
    alt: 'Red supercar driving through coastal canyon at night with city lights in distance',
    content: `The canyon roads stretch ahead, empty and inviting. This is where precision meets passion, where the everyday transforms into the extraordinary.`,
    category: 'Lifestyle',
  },
  {
    id: 'grid-stories',
    title: 'GRID STORIES',
    subtitle: 'Behind the scenes of excellence',
    image: urbanImage,
    alt: 'Red supercar parked on urban cobblestone street with modern architecture and evening ambiance',
    content: `Every piece begins with obsession over details. From the first sketch to the final stitch, we refuse to compromise on quality or vision.`,
    category: 'Craft',
  },
];

const Stories = () => {
  const [selectedStory, setSelectedStory] = useState<Story | null>(null);

  return (
    <section className="min-h-screen bg-background py-24">
      <div className="container mx-auto px-6">
        {/* Header */}
        <div className="text-center mb-20">
          <h2 className="text-5xl md:text-7xl font-display font-light tracking-tight text-shimmer mb-8">
            Stories
          </h2>
          <p className="text-lg text-foreground/60 font-ui font-light max-w-xl mx-auto">
            Four collections. Fourteen pieces. Available in select cities.
          </p>
        </div>

        {/* Story Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {stories.map((story, index) => (
            <div
              key={story.id}
              className="group cursor-pointer"
              onClick={() => setSelectedStory(story)}
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <div className="relative overflow-hidden aspect-[4/5] mb-6">
                <img
                  src={story.image}
                  alt={story.alt}
                  width="400"
                  height="500"
                  className="w-full h-full object-cover supercar-filter group-hover:scale-105 transition-transform duration-700 ease-out"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="absolute bottom-6 left-6 transform translate-y-4 group-hover:translate-y-0 transition-transform duration-500">
                  <span className="inline-block px-3 py-1 bg-accent/20 text-accent text-sm font-ui font-medium tracking-wide mb-2">
                    {story.category}
                  </span>
                </div>
              </div>
              <h3 className="text-2xl font-display font-light tracking-wide text-foreground mb-2 group-hover:text-accent transition-colors duration-300">
                {story.title}
              </h3>
              <p className="text-foreground/60 font-ui font-light">{story.subtitle}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Story Modal */}
      {selectedStory && (
        <div className="fixed inset-0 bg-background/95 backdrop-blur-xl z-50 flex items-center justify-center p-6 animate-fade-in">
          <div className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="relative">
              {/* Close Button */}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSelectedStory(null)}
                aria-label="Close story details"
                className="absolute top-4 right-4 z-10 w-12 h-12 text-foreground hover:text-accent bg-background/50 backdrop-blur-sm"
              >
                <X className="h-6 w-6" />
              </Button>

              {/* Story Content */}
              <div className="grid md:grid-cols-2 gap-12">
                <div className="relative">
                  <img
                    src={selectedStory.image}
                    alt={selectedStory.alt}
                    width="400"
                    height="500"
                    className="w-full aspect-[4/5] object-cover supercar-filter"
                  />
                </div>
                <div className="flex flex-col justify-center py-8">
                  <span className="inline-block px-4 py-2 bg-accent/10 text-accent text-sm font-ui font-medium tracking-wide mb-4 self-start">
                    {selectedStory.category}
                  </span>
                  <h3 className="text-4xl md:text-6xl font-display font-light tracking-tight text-foreground mb-4">
                    {selectedStory.title}
                  </h3>
                  <p className="text-xl text-foreground/60 font-ui font-light mb-8">
                    {selectedStory.subtitle}
                  </p>
                  <p className="text-lg text-foreground/80 font-ui font-light leading-relaxed">
                    {selectedStory.content}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

export default Stories;
