import { useState, useEffect, useRef } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogClose } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { X, Gift } from 'lucide-react';
import { emailService } from '@/services/emailService';

/**
 * EmailCaptureModal component displays a modal for email capture with time/scroll triggers
 *
 * Displays an email signup modal that triggers after 5 seconds on page or when user
 * scrolls to 50% of page height. Stores preference in localStorage to not show again
 * to returning users.
 *
 * Features:
 * - 10% first order discount incentive
 * - Time-based trigger (5 seconds)
 * - Scroll-based trigger (50% down page)
 * - localStorage persistence (24-hour window)
 * - Email validation
 * - Accessible dialog using Radix UI
 *
 * @component
 *
 * @example
 * <EmailCaptureModal />
 *
 * @remarks
 * - Uses localStorage key 'email_modal_dismissed' to track if user has seen modal
 * - Dismissal preference lasts 24 hours before showing again
 * - Modal renders at root level for proper z-index stacking
 * - Includes close button (X) and backdrop click to dismiss
 */
const EmailCaptureModal = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const triggerRef = useRef(false);

  // Email validation regex
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  const validateEmail = (email: string) => {
    if (!email) {
      return 'Email address is required';
    }
    if (!emailRegex.test(email)) {
      return 'Please enter a valid email address';
    }
    return '';
  };

  // Check if modal has been dismissed
  const isModalDismissed = () => {
    const dismissed = localStorage.getItem('email_modal_dismissed');
    if (!dismissed) return false;

    const dismissedTime = parseInt(dismissed, 10);
    const now = Date.now();
    const twentyFourHours = 24 * 60 * 60 * 1000;

    // If more than 24 hours have passed, show modal again
    if (now - dismissedTime > twentyFourHours) {
      localStorage.removeItem('email_modal_dismissed');
      return false;
    }

    return true;
  };

  // Dismiss modal and store in localStorage
  const handleDismiss = () => {
    setIsOpen(false);
    localStorage.setItem('email_modal_dismissed', Date.now().toString());
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const validationError = validateEmail(email);
    if (validationError) {
      setError(validationError);
      return;
    }

    setError('');
    setIsLoading(true);

    try {
      const success = await emailService.subscribeNewsletter(email);
      if (success) {
        setIsSubmitted(true);
        setEmail('');
        // Auto-close after 2 seconds
        setTimeout(() => {
          handleDismiss();
          setIsSubmitted(false);
        }, 2000);
      } else {
        throw new Error('Subscription failed');
      }
    } catch (err) {
      setError('Failed to subscribe. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Set up triggers for modal
  useEffect(() => {
    // Don't show if already dismissed
    if (isModalDismissed()) {
      return;
    }

    // Time-based trigger (5 seconds)
    const timeoutId = setTimeout(() => {
      if (!triggerRef.current) {
        setIsOpen(true);
        triggerRef.current = true;
      }
    }, 5000);

    // Scroll-based trigger (50% down page)
    const handleScroll = () => {
      if (triggerRef.current) return;

      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrolled = window.scrollY;
      const scrollPercentage = (scrolled / scrollHeight) * 100;

      if (scrollPercentage >= 50) {
        setIsOpen(true);
        triggerRef.current = true;
        window.removeEventListener('scroll', handleScroll);
      }
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  if (isSubmitted) {
    return (
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-md bg-lii-bg border border-lii-gold/20 backdrop-blur-xl">
          <div className="flex justify-center mb-4">
            <div className="w-12 h-12 bg-lii-gold/20 rounded-full flex items-center justify-center">
              <Gift className="w-6 h-6 text-lii-gold" />
            </div>
          </div>
          <div className="text-center">
            <h2 className="text-2xl font-display text-lii-cloud mb-2">
              Welcome to the Iconic List
            </h2>
            <p className="text-lii-ash mb-6">
              Check your email for your 10% discount code. It's on the way!
            </p>
            <Button
              variant="primary"
              size="lg"
              onClick={handleDismiss}
              className="w-full"
            >
              Continue Shopping
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="sm:max-w-md bg-lii-bg border border-lii-gold/20 backdrop-blur-xl">
        <DialogClose className="absolute right-4 top-4 rounded-sm opacity-70 hover:opacity-100 transition-opacity">
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </DialogClose>

        <DialogHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className="w-12 h-12 bg-lii-gold/20 rounded-full flex items-center justify-center">
              <Gift className="w-6 h-6 text-lii-gold" />
            </div>
          </div>

          <DialogTitle className="text-2xl font-display text-lii-cloud">
            Join the Iconic List
          </DialogTitle>

          <DialogDescription className="text-lii-ash mt-2">
            Get early access to drops and exclusive events
          </DialogDescription>
        </DialogHeader>

        {/* Email Form */}
        <form onSubmit={handleSubmit} className="space-y-4 pt-4">
          <div>
            <label htmlFor="modal-email" className="sr-only">
              Email address
            </label>
            <Input
              id="modal-email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                if (error) setError('');
              }}
              className="bg-lii-ink/50 border-lii-gold/20 text-foreground placeholder:text-foreground/40 focus:border-lii-gold/60 focus:ring-lii-gold/20 h-11"
              aria-invalid={error ? 'true' : 'false'}
              aria-describedby={error ? 'modal-email-error' : undefined}
            />
            {error && (
              <div
                id="modal-email-error"
                className="text-red-400 text-sm mt-2"
                role="alert"
              >
                {error}
              </div>
            )}
          </div>

          {/* Incentive Callout */}
          <div className="bg-lii-gold/10 border border-lii-gold/20 rounded-lg p-3 text-center">
            <p className="text-sm font-semibold text-lii-gold">
              10% off your first order
            </p>
          </div>

          <Button
            type="submit"
            variant="primary"
            size="lg"
            disabled={isLoading || !email}
            className="w-full"
          >
            {isLoading ? 'Sending...' : 'Get My Discount'}
          </Button>

          <button
            type="button"
            onClick={handleDismiss}
            className="w-full text-sm text-lii-ash hover:text-lii-cloud transition-colors"
          >
            Not now, maybe later
          </button>
        </form>

        {/* Disclaimer */}
        <p className="text-xs text-lii-ash/60 text-center">
          We respect your privacy. Unsubscribe anytime.
        </p>
      </DialogContent>
    </Dialog>
  );
};

export default EmailCaptureModal;
