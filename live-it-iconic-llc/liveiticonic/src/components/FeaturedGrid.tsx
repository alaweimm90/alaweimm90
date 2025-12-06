import { memo, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

// Product seed data based on the prompt specifications
const FEATURED_PRODUCTS = [
  {
    id: 'LII-TEE-001',
    name: 'Iconic Core Tee',
    price: 32,
    image: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=500&fit=crop',
    alt: 'Black essential cotton tee with minimal design and tailored fit',
    badge: 'Core',
  },
  {
    id: 'LII-TEE-002',
    name: 'Track Stripe Tee',
    price: 36,
    comparePrice: 42,
    image: 'https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=400&h=500&fit=crop',
    alt: 'White athletic tee with signature stripe detail for street style',
    badge: 'Sale',
  },
  {
    id: 'LII-HOOD-001',
    name: 'Pit Lane Hoodie',
    price: 68,
    image: 'https://images.unsplash.com/photo-1556821840-3a9fbc86339e?w=400&h=500&fit=crop',
    alt: 'Charcoal heavyweight fleece hoodie with reinforced details',
    badge: '',
  },
  {
    id: 'LII-JOG-001',
    name: 'Apex Jogger',
    price: 58,
    image: 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&h=500&fit=crop',
    alt: 'Black tapered athletic joggers with zip pockets and stretch fabric',
    badge: '',
  },
  {
    id: 'LII-CAP-001',
    name: 'Overrun Cap',
    price: 28,
    image: 'https://images.unsplash.com/photo-1588850561407-ed78c282e89b?w=400&h=500&fit=crop',
    alt: 'Low profile black cap with tonal logo and breathable design',
    badge: '',
  },
  {
    id: 'LII-TEE-003',
    name: 'Velocity Tee',
    price: 38,
    image: 'https://images.unsplash.com/photo-1583743814966-8936f37f4042?w=400&h=500&fit=crop',
    alt: 'Performance tee with moisture wicking fabric and athletic cut',
    badge: 'New',
  },
  {
    id: 'LII-CREW-001',
    name: 'Grid Crew',
    price: 52,
    image: 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=500&fit=crop',
    alt: 'Minimal crew neck sweatshirt with clean lines and premium cotton',
    badge: '',
  },
  {
    id: 'LII-SHORT-001',
    name: 'Circuit Shorts',
    price: 42,
    image: 'https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=400&h=500&fit=crop',
    alt: 'Athletic shorts with four way stretch and secure zip pockets',
    badge: '',
  },
];

const FeaturedGrid = memo(() => {
  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-4xl md:text-5xl font-display font-normal mb-4">Featured</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-ui">
            Essential gear for those who refuse to compromise.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {FEATURED_PRODUCTS.map((product, index) => (
            <Card
              key={product.id}
              className="group cursor-pointer overflow-hidden bg-card/30 backdrop-blur-sm border border-border/20 transition-all"
              style={{ transition: 'transform var(--motion-enter) var(--ease)' }}
            >
              <div className="relative aspect-[4/5] overflow-hidden">
                <img
                  src={product.image}
                  alt={product.alt}
                  width="400"
                  height="500"
                  className="w-full h-full object-cover group-hover:scale-105"
                  style={{ transition: 'transform var(--motion-enter) var(--ease)' }}
                  loading="lazy"
                />
                {product.badge && (
                  <span
                    className={`absolute top-3 left-3 px-2 py-1 text-xs font-medium rounded ${
                      product.badge === 'Sale'
                        ? 'bg-lii-signal-red text-white'
                        : product.badge === 'New'
                          ? 'bg-accent text-accent-foreground'
                          : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {product.badge}
                  </span>
                )}
              </div>
              <div className="p-4">
                <h3
                  className="font-ui font-medium mb-2 group-hover:text-accent transition-colors"
                  style={{ transition: 'color var(--motion-micro) var(--ease)' }}
                >
                  {product.name}
                </h3>
                <div className="flex items-center gap-2">
                  <span
                    className="font-numeric font-semibold group-hover:text-accent transition-colors"
                    style={{ transition: 'color var(--motion-micro) var(--ease)' }}
                  >
                    ${product.price}
                  </span>
                  {product.comparePrice && (
                    <span className="text-sm text-muted-foreground line-through font-numeric">
                      ${product.comparePrice}
                    </span>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>

        <div className="text-center animate-fade-in-up" style={{ animationDelay: '0.8s' }}>
          <Button size="lg" className="btn-lii-primary px-8 group">
            <span
              className="group-hover:translate-x-1 transition-transform"
              style={{ transition: 'transform var(--motion-micro) var(--ease)' }}
            >
              View All Products
            </span>
          </Button>
        </div>
      </div>
    </section>
  );
});

FeaturedGrid.displayName = 'FeaturedGrid';

export default FeaturedGrid;
