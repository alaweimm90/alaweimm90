import { Truck, RotateCcw, ShieldCheck } from 'lucide-react';

/**
 * TrustBar component displays key USP (Unique Selling Proposition) trust signals
 *
 * Renders three trust-building features with icons to build confidence and reduce
 * purchase friction. Positioned prominently with fine typography and subtle animations.
 *
 * Features:
 * - Free shipping over $100
 * - Easy 30-day returns
 * - Secure checkout
 *
 * @component
 *
 * @example
 * <TrustBar />
 *
 * @remarks
 * - Uses lucide-react icons for consistent visual language
 * - Typography: text-xs (12px) with letter-spacing: wider (+2%)
 * - Responsive layout stacks on mobile
 * - Subtle hover effects for interactivity
 */
const TrustBar = () => {
  const trustSignals = [
    {
      icon: Truck,
      title: 'Free shipping over $100',
      description: 'Fast, reliable delivery worldwide',
    },
    {
      icon: RotateCcw,
      title: 'Easy 30-day returns',
      description: 'Hassle-free returns and exchanges',
    },
    {
      icon: ShieldCheck,
      title: 'Secure checkout',
      description: 'Bank-level security for your data',
    },
  ];

  return (
    <section
      className="py-8 md:py-10 bg-lii-ink/40 border-y border-lii-gold/10 relative overflow-hidden"
      aria-label="Why choose Live It Iconic"
    >
      {/* Subtle background gradient */}
      <div className="absolute inset-0 opacity-5 pointer-events-none">
        <div className="absolute top-1/2 left-1/4 w-96 h-96 bg-lii-gold rounded-full blur-[120px]"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
          {trustSignals.map((signal, index) => {
            const Icon = signal.icon;
            return (
              <div
                key={index}
                className="flex flex-col items-center md:items-start text-center md:text-left group"
              >
                {/* Icon Container */}
                <div className="mb-4 p-3 rounded-lg bg-lii-gold/10 border border-lii-gold/20 group-hover:bg-lii-gold/15 group-hover:border-lii-gold/40 transition-all duration-300">
                  <Icon className="w-6 h-6 text-lii-gold" aria-hidden="true" />
                </div>

                {/* Content */}
                <h3 className="text-xs sm:text-sm font-ui font-semibold tracking-wider text-lii-cloud mb-1.5 uppercase">
                  {signal.title}
                </h3>
                <p className="text-xs font-ui text-lii-ash leading-relaxed">
                  {signal.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default TrustBar;
