import { useState } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { CreditCard } from 'lucide-react';

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLIC_KEY || '');

interface OneTimeCheckoutProps {
  planType: 'one-time-basic' | 'one-time-premium' | 'one-time-concierge';
  email: string;
  name: string;
  intakeData?: Record<string, any>;
  clientId?: string;
  onSuccess?: () => void;
}

const PLAN_PRICES = {
  'one-time-basic': {
    amount: 29900, // $299 in cents
    name: 'Basic Plan',
    description: '8-week personalized program',
  },
  'one-time-premium': {
    amount: 59900, // $599 in cents
    name: 'Premium Plan',
    description: '12-week comprehensive program',
  },
  'one-time-concierge': {
    amount: 149900, // $1,499 in cents
    name: 'Concierge',
    description: '12-week personalized concierge service',
  },
};

export function OneTimeCheckout({
  planType,
  email,
  name,
  intakeData,
  clientId,
  onSuccess,
}: OneTimeCheckoutProps) {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);

  const handlePayment = async () => {
    if (!planType || !email || !name) {
      toast({
        title: 'Missing Information',
        description: 'Please complete the intake form before proceeding to payment.',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);

    try {
      const stripe = await stripePromise;

      if (!stripe) {
        throw new Error('Stripe failed to load');
      }

      // Call Edge function to create Stripe checkout session
      const { data, error } = await supabase.functions.invoke('create-one-time-checkout', {
        body: {
          planType,
          email,
          name,
          intakeData,
          clientId,
        },
      });

      if (error) throw error;

      if (!data?.sessionId) {
        throw new Error('Failed to create checkout session');
      }

      // Redirect to Stripe Checkout
      const result = await stripe.redirectToCheckout({
        sessionId: data.sessionId,
      });

      if (result.error) {
        throw new Error(result.error.message);
      }
    } catch (error) {
      console.error('Payment error:', error);
      toast({
        title: 'Payment Failed',
        description: error instanceof Error ? error.message : 'An error occurred during checkout.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const planInfo = PLAN_PRICES[planType];

  return (
    <div className="space-y-4">
      <div className="p-4 bg-surface-elevated rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-2">{planInfo.name}</h3>
        <p className="text-white/70 mb-4">{planInfo.description}</p>
        <div className="flex justify-between items-center">
          <span className="text-2xl font-bold text-repz-orange">
            ${(planInfo.amount / 100).toFixed(2)}
          </span>
          <Button
            onClick={handlePayment}
            disabled={isLoading}
            className="flex items-center gap-2 bg-repz-orange hover:bg-repz-orange-dark"
          >
            {isLoading ? (
              'Processing...'
            ) : (
              <>
                <CreditCard className="h-4 w-4" />
                Pay with Card
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="text-xs text-white/50 text-center">
        Secure payment processing powered by Stripe.
        <br />
        Your payment information is never stored on our servers.
      </div>
    </div>
  );
}
