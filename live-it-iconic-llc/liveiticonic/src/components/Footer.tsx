import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Instagram, Twitter, Youtube } from 'lucide-react';
import { Link } from 'react-router-dom';
import DiamondLogo from './icons/DiamondLogo';

/**
 * Footer component displays site-wide footer with brand info, navigation, and email signup
 *
 * Renders multi-column footer layout with brand description, navigation links, social media,
 * email signup form, and legal information. Features animated background decoration and
 * responsive grid layout that adapts to mobile screens.
 *
 * @component
 *
 * @example
 * <Footer />
 *
 * @remarks
 * - 4-column grid layout on large screens
 * - Email signup functionality
 * - Social media links (Instagram, Twitter, YouTube)
 * - Animated background orbs (decorative elements)
 * - Newsletter subscription prompt
 */
const Footer = () => {
  return (
    <footer className="bg-lii-ink border-t border-lii-gold/10 relative overflow-hidden">
      <div className="container mx-auto px-6 py-16 relative z-10">
        {/* Main Footer Content */}
        <div className="grid lg:grid-cols-4 gap-12 mb-12">
          {/* Brand Section */}
          <div className="lg:col-span-2">
            <Link
              to="/"
              className="inline-flex items-center gap-4 sm:gap-5 mb-6 group"
              aria-label="Live It Iconic Home"
            >
              <DiamondLogo className="w-16 h-16 sm:w-16 sm:h-16 text-lii-cloud transition-all duration-500 group-hover:scale-110 group-hover:drop-shadow-[0_0_15px_rgba(230,233,239,0.8)] flex-shrink-0" />
              <span className="flex items-center h-16 text-2xl sm:text-3xl font-display tracking-wider leading-none">
                <span className="text-lii-cloud">LIVE</span>
                <span className="text-lii-ash mx-2">IT</span>
                <span className="text-lii-gold">ICONIC</span>
              </span>
            </Link>
            <p className="text-lii-ash font-ui font-light leading-relaxed max-w-sm mb-6">
              Premium lifestyle merchandise for those who demand excellence in every detail
            </p>
            <div className="w-16 h-px bg-lii-gold/30"></div>
          </div>
          {/* Connect Section */}
          <div>
            <h4 className="text-base font-ui font-semibold text-lii-gold mb-6 uppercase tracking-wider">
              Connect
            </h4>
            <div className="space-y-4">
              <a
                href="mailto:support@liveiconic.com"
                className="block text-lii-ash hover:text-lii-cloud font-ui font-light text-sm transition-colors duration-micro"
              >
                support@liveiconic.com
              </a>
              <div className="flex space-x-3" role="list" aria-label="Social media links">
                {[
                  {
                    Icon: Instagram,
                    label: 'Instagram',
                    href: 'https://instagram.com/liveiticonic',
                  },
                  { Icon: Twitter, label: 'Twitter', href: 'https://twitter.com/liveiticonic' },
                  { Icon: Youtube, label: 'YouTube', href: 'https://youtube.com/@liveiticonic' },
                ].map(({ Icon, label, href }) => (
                  <a
                    key={label}
                    href={href}
                    aria-label={`Follow us on ${label}`}
                    className="w-11 h-11 flex items-center justify-center border border-lii-gold/20 rounded-lg text-lii-ash hover:text-lii-gold hover:border-lii-gold/40 transition-all duration-micro"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon className="w-5 h-5" />
                  </a>
                ))}
              </div>
            </div>
          </div>

          {/* Updates Section */}
          <div>
            <h4 className="text-base font-ui font-semibold text-lii-gold mb-4 uppercase tracking-wider">
              Updates
            </h4>
            <form className="space-y-3" onSubmit={e => e.preventDefault()}>
              <label htmlFor="newsletter-email" className="sr-only">
                Email address for newsletter
              </label>
              <Input
                id="newsletter-email"
                type="email"
                placeholder="your@email.com"
                required
                aria-required="true"
                className="h-12 bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50 focus:border-lii-gold font-ui font-light"
              />
              <Button type="submit" variant="primary" className="w-full font-ui font-medium">
                Join
              </Button>
            </form>
          </div>
        </div>

        {/* Trust Signals */}
        <div className="py-8 border-t border-lii-gold/10">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            {/* Security & Payment */}
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 bg-lii-gold/10 rounded-full flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-lii-gold"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                  />
                </svg>
              </div>
              <h5 className="text-lii-cloud font-ui font-semibold text-sm mb-2">Secure Payments</h5>
              <p className="text-lii-ash font-ui font-light text-xs leading-relaxed">
                256-bit SSL encryption. All transactions protected.
              </p>
            </div>

            {/* Shipping Guarantee */}
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 bg-lii-gold/10 rounded-full flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-lii-gold"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
                  />
                </svg>
              </div>
              <h5 className="text-lii-cloud font-ui font-semibold text-sm mb-2">Free Shipping</h5>
              <p className="text-lii-ash font-ui font-light text-xs leading-relaxed">
                Free worldwide shipping on orders over $100. Express delivery available.
              </p>
            </div>

            {/* Customer Guarantee */}
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 bg-lii-gold/10 rounded-full flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-lii-gold"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h5 className="text-lii-cloud font-ui font-semibold text-sm mb-2">
                30-Day Guarantee
              </h5>
              <p className="text-lii-ash font-ui font-light text-xs leading-relaxed">
                Not satisfied? Full refund within 30 days. No questions asked.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-lii-gold/10 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-lii-ash font-ui font-light text-sm">© 2025 Live It Iconic</p>
          <nav className="flex flex-wrap justify-center gap-6" aria-label="Footer navigation">
            {[
              { label: 'Shipping', href: '/policies#shipping' },
              { label: 'Returns', href: '/policies#returns' },
              { label: 'Privacy', href: '/policies#privacy' },
              { label: 'Terms', href: '/policies#terms' },
            ].map(link => (
              <Link
                key={link.label}
                to={link.href}
                className="text-lii-ash hover:text-lii-cloud font-ui font-light text-sm transition-colors duration-micro"
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
