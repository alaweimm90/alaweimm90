import { Button } from '@/components/ui/button';
import { useCart } from '@/contexts/CartContext';
import { Menu, X } from 'lucide-react';
import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Link } from 'react-router-dom';
import CartDrawer from './CartDrawer';
import CartIcon from './CartIcon';
import DiamondLogo from './icons/DiamondLogo';
import BirdLogoShowcase from './BirdLogoShowcase';

const NAV_ITEMS = [
  { name: 'About', href: '/about' },
  { name: 'Lifestyle', href: '/lifestyle' },
  { name: 'Collection', href: '/collection' },
  { name: 'Brand', href: '/brand' },
  { name: 'Launch', href: '/launch' },
  { name: 'Contact', href: '/contact' },
];

/**
 * Navigation component provides main site navigation with responsive mobile menu and cart drawer
 *
 * Features a sticky header that appears after scrolling, logo with brand wordmark, navigation links,
 * and cart icon. Includes mobile-responsive hamburger menu with keyboard support (Escape to close).
 * Integrates with CartContext for cart drawer functionality.
 *
 * @component
 *
 * @example
 * <Navigation />
 *
 * @remarks
 * - Becomes sticky after 30px scroll
 * - Mobile menu toggles with hamburger button
 * - Escape key closes mobile menu
 * - Cart drawer integrated with cart icon click
 * - Includes focus management for accessibility
 */
const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const firstLinkRef = useRef<HTMLAnchorElement>(null);
  const { setIsOpen } = useCart();

  const handleOpenCart = useCallback(() => {
    setIsOpen(true);
  }, [setIsOpen]);

  const handleScroll = useCallback(() => {
    setIsScrolled(window.scrollY > 30);
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMenuOpen) {
        setIsMenuOpen(false);
      }
    },
    [isMenuOpen]
  );

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleScroll, handleKeyDown]);

  useEffect(() => {
    if (isMenuOpen && firstLinkRef.current) {
      firstLinkRef.current.focus();
    }
  }, [isMenuOpen]);

  return (
    <nav
      id="navigation"
      className={`fixed top-0 w-full z-50 transition-all duration-300 ease-in-out ${
        isScrolled
          ? 'backdrop-blur-2xl bg-lii-bg/95 border-b border-lii-gold/30 shadow-[0_4px_32px_rgba(193,160,96,0.15)]'
          : 'backdrop-blur-xl bg-lii-bg/70 border-b border-lii-gold/10'
      }`}
    >
      {/* Animated top border glow */}
      <div className="absolute top-0 left-0 right-0 h-px overflow-hidden">
        <div
          className={`w-full h-full bg-gradient-to-r from-transparent via-lii-gold/50 to-transparent animate-borderShimmer transition-opacity duration-300 ${
            isScrolled ? 'opacity-100' : 'opacity-0'
          }`}
        ></div>
      </div>

      <div className="container mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-24">
          {/* Logo - Perfectly aligned with Bird Showcase */}
          <div className="flex items-center gap-4 sm:gap-6">
            <Link
              to="/"
              className="flex items-center gap-3 sm:gap-4 group relative"
              aria-label="Live It Iconic Home"
            >
              <DiamondLogo className="w-14 h-14 sm:w-16 sm:h-16 text-lii-gold transition-all duration-500 group-hover:scale-110 group-hover:drop-shadow-[0_0_12px_rgba(193,160,96,0.8)] flex-shrink-0" />
              <span className="flex items-center h-14 sm:h-16 text-xl sm:text-2xl font-display tracking-wider transition-all duration-300 leading-none">
                <span className="text-lii-cloud group-hover:drop-shadow-[0_0_8px_rgba(230,233,239,0.6)] transition-all duration-300">
                  LIVE
                </span>
                <span className="text-lii-ash mx-1.5 group-hover:text-lii-cloud transition-all duration-300">
                  IT
                </span>
                <span className="text-lii-gold group-hover:drop-shadow-[0_0_10px_rgba(193,160,96,0.8)] transition-all duration-300">
                  ICONIC
                </span>
              </span>
            </Link>

            {/* Bird Logo Showcase - Compact variant */}
            <div className="hidden md:flex items-center border-l border-lii-gold/20 pl-4 sm:pl-6 h-14 sm:h-16">
              <BirdLogoShowcase variant="compact" />
            </div>
          </div>

          {/* Enhanced Navigation - Desktop */}
          <nav
            className="hidden lg:flex items-center space-x-10 lg:space-x-14"
            aria-label="Primary navigation"
          >
            {NAV_ITEMS.map(item => (
              <Link
                key={item.name}
                to={item.href.replace('/#', '/')}
                className={`group relative text-base lg:text-lg font-ui tracking-wide transition-all duration-300 ease-out px-4 py-2 rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2`}
              >
                {/* Glass background on hover */}
                <div className="absolute inset-0 backdrop-blur-sm bg-lii-gold/5 rounded-xl border border-lii-gold/0 opacity-0 group-hover:opacity-100 group-hover:border-lii-gold/20 transition-all duration-300 scale-95 group-hover:scale-100"></div>

                <span className="relative text-lii-cloud transition-all duration-300 group-hover:text-lii-gold group-hover:drop-shadow-[0_0_10px_rgba(193,160,96,0.7)]">
                  {item.name}
                </span>

                {/* Animated underline */}
                <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-0 h-[2px] bg-gradient-to-r from-transparent via-lii-gold to-transparent transition-all duration-300 ease-out group-hover:w-full shadow-[0_0_12px_rgba(193,160,96,0.9)]"></div>
              </Link>
            ))}

            {/* Cart Icon - Desktop */}
            <CartIcon onClick={handleOpenCart} />
          </nav>

          {/* Mobile Menu Button & Cart */}
          <div className="lg:hidden flex items-center gap-2">
            <CartIcon onClick={handleOpenCart} />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              aria-label={isMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
              aria-expanded={isMenuOpen}
              className="min-w-[44px] min-h-[44px] text-lii-cloud hover:text-lii-gold transition-all duration-micro"
            >
              {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation Overlay */}
        {isMenuOpen && (
          <div
            className="fixed inset-0 top-24 backdrop-blur-2xl bg-lii-bg/95 lg:hidden z-40 transition-all duration-300"
            role="dialog"
            aria-modal="true"
            aria-label="Mobile navigation"
            aria-hidden="false"
          >
            {/* Decorative glow */}
            <div className="absolute inset-0 opacity-10">
              <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-96 h-96 bg-lii-gold rounded-full blur-[120px] animate-[luxuryPulse_8s_ease-in-out_infinite]"></div>
            </div>

            <nav className="flex flex-col items-center justify-center h-full space-y-12 px-6 relative z-10">
              {NAV_ITEMS.map((item, index) => (
                <Link
                  key={item.name}
                  ref={index === 0 ? firstLinkRef : null}
                  to={item.href.replace('/#', '/')}
                  className="text-3xl font-display text-lii-cloud hover:text-lii-gold transition-all duration-300 min-h-[44px] flex items-center hover:drop-shadow-[0_0_12px_rgba(193,160,96,0.8)] opacity-0 animate-fade-in-up focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2 rounded-lg"
                  onClick={() => setIsMenuOpen(false)}
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
        )}

        {/* Cart Drawer */}
        <CartDrawer isOpen={useCart().isOpen} onClose={() => setIsOpen(false)} />
      </div>
    </nav>
  );
};

export default Navigation;
