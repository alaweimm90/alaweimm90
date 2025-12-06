import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { ShippingAddress } from '@/types/order';
import { CreditCard, Shield, ArrowLeft, Loader2 } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { paymentService } from '@/services/paymentService';
import { orderService } from '@/services/orderService';
import { loadStripe, StripeElementsOptions } from '@stripe/stripe-js';
import { Elements, CardElement, useStripe, useElements } from '@stripe/react-stripe-js';
import { useToast } from '@/hooks/use-toast';
import { getErrorMessage } from '@/types/api';

// Initialize Stripe
const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY || '');

interface PaymentFormProps {
  shippingData: ShippingAddress;
  amount: number;
  onSuccess: (orderId: string) => void;
  onBack: () => void;
}

/**
 * PaymentFormContent component handles Stripe payment processing
 *
 * Manages payment intent creation, card validation via Stripe Elements, and order creation
 * upon successful payment. Displays cardholder name field and secure payment confirmation.
 *
 * @component
 * @internal Used internally by PaymentForm wrapper
 * @param {PaymentFormProps} props - Component props
 * @param {ShippingAddress} props.shippingData - Customer shipping information for order creation
 * @param {number} props.amount - Order total amount in USD cents
 * @param {Function} props.onSuccess - Callback fired with orderId upon successful payment
 * @param {Function} props.onBack - Callback to navigate back to shipping form
 */
const PaymentFormContent: React.FC<PaymentFormProps> = ({
  shippingData,
  amount,
  onSuccess,
  onBack,
}) => {
  const stripe = useStripe();
  const elements = useElements();
  const { toast } = useToast();
  const [cardholderName, setCardholderName] = useState('');
  const [clientSecret, setClientSecret] = useState<string | null>(null);

  // Create payment intent on mount
  useEffect(() => {
    const createIntent = async () => {
      try {
        const response = await paymentService.createPaymentIntent({
          amount,
          orderData: {
            items: [], // Will be populated from cart
            shippingAddress: shippingData,
          },
        });
        setClientSecret(response.clientSecret);
      } catch (error: unknown) {
        toast({
          title: 'Error',
          description: getErrorMessage(error) || 'Failed to initialize payment',
        });
      }
    };
    createIntent();
  }, [amount, shippingData, toast]);

  const paymentMutation = useMutation({
    mutationFn: async () => {
      if (!stripe || !elements || !clientSecret) {
        throw new Error('Stripe not loaded');
      }

      const cardElement = elements.getElement(CardElement);
      if (!cardElement) {
        throw new Error('Card element not found');
      }

      // Confirm payment with Stripe
      const { error: confirmError, paymentIntent } = await stripe.confirmCardPayment(clientSecret, {
        payment_method: {
          card: cardElement,
          billing_details: {
            name: cardholderName,
            email: shippingData.email,
            phone: shippingData.phone,
            address: {
              line1: shippingData.address,
              city: shippingData.city,
              state: shippingData.state,
              postal_code: shippingData.zipCode,
              country: shippingData.country,
            },
          },
        },
      });

      if (confirmError) {
        throw new Error(confirmError.message || 'Payment failed');
      }

      if (!paymentIntent || paymentIntent.status !== 'succeeded') {
        throw new Error('Payment not completed');
      }

      // Create order
      const orderId = await orderService.createOrder({
        shippingData,
        amount,
        paymentMethod: 'card',
      });

      return orderId;
    },
    onSuccess: orderId => {
      toast({
        title: 'Payment Successful',
        description: 'Your order has been placed successfully!',
      });
      onSuccess(orderId);
    },
    onError: (error: Error) => {
      toast({
        title: 'Payment Failed',
        description: getErrorMessage(error) || 'Please check your card details and try again.',
      });
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cardholderName.trim()) {
      toast({
        title: 'Validation Error',
        description: 'Please enter the cardholder name',
      });
      return;
    }
    paymentMutation.mutate();
  };

  const cardElementOptions = {
    style: {
      base: {
        fontSize: '16px',
        color: '#E5E7EB',
        fontFamily: 'system-ui, sans-serif',
        '::placeholder': {
          color: '#9CA3AF',
        },
      },
      invalid: {
        color: '#EF4444',
        iconColor: '#EF4444',
      },
    },
  };

  return (
    <Card className="bg-lii-ink border-lii-gold/10">
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-lii-cloud">
          <CreditCard className="w-5 h-5 text-lii-gold" />
          Payment Information
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Security Notice */}
          <div className="flex items-center gap-3 p-4 bg-lii-gold/5 rounded-lg border border-lii-gold/20">
            <Shield className="w-5 h-5 text-lii-gold flex-shrink-0" />
            <div>
              <p className="text-lii-cloud font-ui font-medium text-sm">Secure Payment</p>
              <p className="text-lii-ash font-ui text-xs">
                Your payment information is encrypted and secure. Powered by Stripe.
              </p>
            </div>
          </div>

          {/* Cardholder Name */}
          <div>
            <Label htmlFor="cardholderName" className="text-lii-ash">
              Cardholder Name *
            </Label>
            <Input
              id="cardholderName"
              required
              value={cardholderName}
              onChange={e => setCardholderName(e.target.value)}
              placeholder="John Doe"
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
            />
          </div>

          {/* Stripe Card Element */}
          <div>
            <Label className="text-lii-ash mb-2 block">Card Details *</Label>
            <div className="p-4 bg-lii-charcoal/20 border border-lii-gold/20 rounded-lg">
              <CardElement options={cardElementOptions} />
            </div>
          </div>

          {/* Error Message */}
          {paymentMutation.isError && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-red-400 font-ui text-sm">
                {paymentMutation.error?.message || 'Payment failed. Please try again.'}
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            <Button
              type="button"
              onClick={onBack}
              variant="outline"
              className="flex-1 border-lii-gold/20 text-lii-ash hover:text-lii-cloud hover:border-lii-gold/40"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <Button
              type="submit"
              disabled={paymentMutation.isPending || !stripe || !clientSecret}
              variant="primary"
              className="flex-1 font-ui font-medium"
            >
              {paymentMutation.isPending ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing...
                </div>
              ) : (
                `Pay $${amount.toFixed(2)}`
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};

/**
 * PaymentForm component wraps payment UI with Stripe Elements provider
 *
 * Initializes Stripe Elements context with payment intent options and renders
 * the PaymentFormContent component. Handles the complete payment workflow including
 * card validation, payment processing, and order confirmation.
 *
 * @component
 * @param {PaymentFormProps} props - Component props
 * @param {ShippingAddress} props.shippingData - Customer shipping information
 * @param {number} props.amount - Order total amount in USD
 * @param {Function} props.onSuccess - Callback with orderId on successful payment
 * @param {Function} props.onBack - Callback to navigate back to shipping
 *
 * @example
 * <PaymentForm
 *   shippingData={shippingInfo}
 *   amount={99.99}
 *   onSuccess={(orderId) => navigate(`/order/${orderId}`)}
 *   onBack={() => setCurrentStep('shipping')}
 * />
 */
export const PaymentForm: React.FC<PaymentFormProps> = props => {
  const stripeOptions: StripeElementsOptions = {
    mode: 'payment',
    amount: Math.round(props.amount * 100),
    currency: 'usd',
  };

  return (
    <Elements stripe={stripePromise} options={stripeOptions}>
      <PaymentFormContent {...props} />
    </Elements>
  );
};

export default PaymentForm;
