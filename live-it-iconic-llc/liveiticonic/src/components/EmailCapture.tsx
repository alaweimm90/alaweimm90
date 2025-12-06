import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ArrowRight, Check, Mail } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { emailService } from '@/services/emailService';

/**
 * EmailCapture component collects newsletter email subscriptions with validation
 *
 * Displays an email input form with validation for newsletter signups. Provides
 * user feedback with error messages, loading state during submission, and success
 * confirmation. Includes email format validation and accessibility features.
 *
 * @component
 *
 * @example
 * <EmailCapture />
 *
 * @remarks
 * - Email validation uses regex pattern for basic format checking
 * - Loading state during API submission
 * - Success state persists briefly before resetting form
 * - Announces status changes to screen readers
 * - Submits email to emailService for newsletter signup
 */
const EmailCapture = () => {
  const [email, setEmail] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const statusRef = useRef<HTMLDivElement>(null);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const validationError = validateEmail(email);
    if (validationError) {
      setError(validationError);
      setStatusMessage(validationError);
      inputRef.current?.focus();
      return;
    }

    setError('');
    setIsLoading(true);
    setStatusMessage('Submitting your email...');

    try {
      const success = await emailService.subscribeNewsletter(email);

      if (success) {
        setIsSubmitted(true);
        setStatusMessage('Successfully subscribed! Welcome to the inner circle.');
        setEmail('');
      } else {
        throw new Error('Subscription failed');
      }
    } catch (err) {
      setError('Failed to subscribe. Please try again.');
      setStatusMessage('Subscription failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setEmail(value);

    // Clear error when user starts typing
    if (error) {
      setError('');
    }
  };

  // Announce status changes to screen readers
  useEffect(() => {
    if (statusMessage && statusRef.current) {
      statusRef.current.focus();
    }
  }, [statusMessage]);

  if (isSubmitted) {
    return (
      <section id="connect" className="py-16 md:py-20 bg-lii-bg relative overflow-hidden">
        <div className="container mx-auto px-6">
          <div className="max-w-2xl mx-auto text-center">
            <div className="backdrop-blur-xl bg-lii-ink/30 border border-lii-gold/20 rounded-2xl p-8 md:p-10 shadow-[0_8px_32px_rgba(193,160,96,0.12)]">
              <div className="w-14 h-14 bg-lii-gold/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Check className="w-6 h-6 text-lii-gold" />
              </div>
              <h3 className="text-2xl sm:text-3xl font-display text-lii-cloud mb-3">
                Welcome to the Experience
              </h3>
              <p className="text-lii-ash font-ui mb-6">
                You'll be the first to know about new collections and exclusive events.
              </p>
              <Button
                onClick={() => setIsSubmitted(false)}
                variant="outline"
                className="border-lii-gold/30 text-lii-gold hover:bg-lii-gold/10"
              >
                Subscribe Another Email
              </Button>
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="py-16 md:py-20 bg-lii-bg relative overflow-hidden">
      {/* Subtle orb */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-1/2 left-1/3 w-64 h-64 bg-lii-gold rounded-full blur-[100px] animate-[luxuryPulse_10s_ease-in-out_infinite]"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-3xl mx-auto text-center">
          {/* Glass Header Container */}
          <div className="backdrop-blur-xl bg-lii-ink/30 border border-lii-gold/20 rounded-2xl p-8 md:p-10 mb-8 shadow-[0_8px_32px_rgba(193,160,96,0.12)]">
            <div className="w-14 h-14 bg-lii-gold/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-lii-gold/20">
              <Mail className="w-6 h-6 text-lii-gold" />
            </div>

            <h2 className="text-3xl sm:text-4xl font-display tracking-tight mb-3">
              <span className="bg-gradient-to-r from-lii-cloud via-lii-gold to-lii-cloud bg-clip-text text-transparent">
                Join the Inner Circle
              </span>
            </h2>

            <div className="w-16 h-px bg-gradient-to-r from-transparent via-lii-gold to-transparent mx-auto mb-4"></div>

            <p className="text-base sm:text-lg font-ui text-lii-ash max-w-2xl mx-auto leading-relaxed">
              Be first to experience new collections and exclusive launches.
            </p>
          </div>

          {/* Email Form */}
          <form onSubmit={handleSubmit} className="max-w-md mx-auto" noValidate>
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <label htmlFor="email-capture" className="sr-only">
                  Email address
                </label>
                <Input
                  ref={inputRef}
                  id="email-capture"
                  type="email"
                  placeholder="Enter your email address"
                  value={email}
                  onChange={handleEmailChange}
                  className={`bg-lii-ink/50 border-lii-gold/20 text-foreground placeholder:text-foreground/40 focus:border-lii-gold/60 focus:ring-lii-gold/20 h-12 ${
                    error ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''
                  }`}
                  required
                  aria-invalid={error ? 'true' : 'false'}
                  aria-describedby={error ? 'email-error' : undefined}
                />
                {error && (
                  <div
                    id="email-error"
                    className="text-red-400 text-sm mt-2 font-ui"
                    role="alert"
                    aria-live="polite"
                  >
                    {error}
                  </div>
                )}
              </div>

              <Button
                type="submit"
                disabled={isLoading || !email}
                className="luxury-button text-lii-black px-8 h-12 font-ui font-medium tracking-wide group disabled:opacity-50 disabled:cursor-not-allowed"
                aria-describedby={isLoading ? 'loading-status' : undefined}
              >
                {isLoading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-lii-black/30 border-t-lii-black rounded-full animate-spin mr-2" />
                    <span id="loading-status">Submitting...</span>
                  </>
                ) : (
                  <>
                    Join Now
                    <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-300" />
                  </>
                )}
              </Button>
            </div>

            {/* ARIA Live Region for Status Announcements */}
            <div
              ref={statusRef}
              aria-live="polite"
              aria-atomic="true"
              className="sr-only"
              tabIndex={-1}
            >
              {statusMessage}
            </div>
          </form>

          {/* Additional Info */}
          <div className="mt-8 space-y-4">
            <div className="flex items-center justify-center gap-6 text-sm font-ui text-foreground/50">
              <span>Exclusive Access</span>
              <span>•</span>
              <span>No Spam</span>
              <span>•</span>
              <span>Unsubscribe Anytime</span>
            </div>

            <p className="text-xs font-ui text-foreground/40">
              By subscribing, you agree to receive marketing communications from Live It Iconic.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EmailCapture;
