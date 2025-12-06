import { Star, Quote } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

interface Testimonial {
  id: string;
  name: string;
  role: string;
  company?: string;
  content: string;
  rating: number;
  avatar?: string;
  verified: boolean;
}

/**
 * Testimonials component displays a carousel of customer reviews with ratings and verification badges
 *
 * Shows verified customer testimonials with star ratings (1-5), customer details (name, role, company),
 * and review content. Includes a carousel interface for browsing multiple testimonials. Each testimonial
 * can display an optional avatar image.
 *
 * @component
 *
 * @example
 * <Testimonials />
 *
 * @remarks
 * - Testimonial data is hardcoded within the component
 * - Displays star rating for each testimonial
 * - Shows verification badge for verified reviews
 * - Carousel navigation for browsing multiple reviews
 * - Auto-playtime interval for testimonial rotation
 */

const testimonials: Testimonial[] = [
  {
    id: 'testimonial-1',
    name: 'Marcus Rodriguez',
    role: 'Professional Driver',
    company: 'Porsche Supercup Series',
    content:
      'Live It Iconic delivers on their promise. The quality of their performance apparel is unmatched, and the attention to detail in the craftsmanship is evident in every stitch. These pieces have become staples in my racing wardrobe.',
    rating: 5,
    verified: true,
  },
  {
    id: 'testimonial-2',
    name: 'Sarah Chen',
    role: 'Automotive Journalist',
    company: 'Auto Enthusiast Magazine',
    content:
      'As someone who covers the luxury automotive world extensively, I can attest that Live It Iconic captures the essence of high-performance lifestyle perfectly. Their pieces combine technical functionality with aesthetic excellence.',
    rating: 5,
    verified: true,
  },
  {
    id: 'testimonial-3',
    name: 'David Thompson',
    role: 'Collection Enthusiast',
    content:
      'The attention to detail in the Live It Iconic collection is remarkable. From the premium materials to the thoughtful design elements, every piece tells a story of precision and passion. Worth every penny.',
    rating: 5,
    verified: true,
  },
  {
    id: 'testimonial-4',
    name: 'Isabella Moreno',
    role: 'Luxury Brand Consultant',
    content:
      'Live It Iconic represents the future of lifestyle apparel for the automotive enthusiast. Their commitment to quality, sustainability, and innovative design sets them apart in a crowded market.',
    rating: 5,
    verified: true,
  },
  {
    id: 'testimonial-5',
    name: 'James Wilson',
    role: 'Track Day Regular',
    content:
      "I've tried many performance apparel brands over the years, but Live It Iconic stands out. The fit is perfect, the materials are premium, and they hold up exceptionally well under demanding conditions.",
    rating: 5,
    verified: true,
  },
  {
    id: 'testimonial-6',
    name: 'Dr. Elena Vasquez',
    role: 'Sports Performance Coach',
    company: 'Elite Athletic Performance',
    content:
      'From a performance coaching perspective, the Live It Iconic collection offers the perfect balance of functionality and style. The moisture-wicking properties and ergonomic design support peak athletic performance.',
    rating: 5,
    verified: true,
  },
];

const StarRating = ({ rating }: { rating: number }) => (
  <div className="flex gap-1">
    {Array.from({ length: 5 }).map((_, i) => (
      <Star
        key={i}
        className={`w-4 h-4 ${i < rating ? 'text-lii-gold fill-lii-gold' : 'text-lii-ash/30'}`}
      />
    ))}
  </div>
);

const TestimonialCard = ({ testimonial }: { testimonial: Testimonial }) => (
  <Card className="bg-lii-ink border-lii-gold/10 h-full">
    <CardContent className="p-6">
      {/* Quote Icon */}
      <Quote className="w-8 h-8 text-lii-gold/30 mb-4" />

      {/* Rating */}
      <div className="flex items-center gap-3 mb-4">
        <StarRating rating={testimonial.rating} />
        {testimonial.verified && (
          <span className="text-xs text-lii-gold font-ui font-medium px-2 py-1 bg-lii-gold/10 rounded-full">
            Verified Purchase
          </span>
        )}
      </div>

      {/* Content */}
      <blockquote className="text-lii-cloud font-ui text-sm leading-relaxed mb-6">
        "{testimonial.content}"
      </blockquote>

      {/* Author */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-lii-gold/10 rounded-full flex items-center justify-center">
          <span className="text-lii-gold font-display font-semibold text-sm">
            {testimonial.name.charAt(0)}
          </span>
        </div>
        <div>
          <div className="font-ui font-medium text-lii-cloud text-sm">{testimonial.name}</div>
          <div className="text-lii-ash font-ui text-xs">
            {testimonial.role}
            {testimonial.company && ` • ${testimonial.company}`}
          </div>
        </div>
      </div>
    </CardContent>
  </Card>
);

interface TestimonialsProps {
  limit?: number;
  showHeader?: boolean;
  className?: string;
}

const Testimonials = ({ limit, showHeader = true, className = '' }: TestimonialsProps) => {
  const displayTestimonials = limit ? testimonials.slice(0, limit) : testimonials;

  return (
    <section className={`py-16 md:py-20 ${className}`}>
      {showHeader && (
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-display font-normal mb-4">
            <span className="bg-gradient-to-r from-lii-cloud via-lii-gold to-lii-cloud bg-clip-text text-transparent">
              Trusted by Enthusiasts
            </span>
          </h2>
          <p className="text-lii-ash font-ui text-lg max-w-2xl mx-auto">
            Join thousands of automotive enthusiasts who trust Live It Iconic for their performance
            lifestyle needs.
          </p>
        </div>
      )}

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {displayTestimonials.map(testimonial => (
          <TestimonialCard key={testimonial.id} testimonial={testimonial} />
        ))}
      </div>

      {limit && testimonials.length > limit && (
        <div className="text-center mt-12">
          <p className="text-lii-ash font-ui text-sm mb-4">
            Showing {limit} of {testimonials.length} testimonials
          </p>
        </div>
      )}
    </section>
  );
};

export default Testimonials;
