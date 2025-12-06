/**
 * Order Confirmation Page
 *
 * This page is displayed after successful payment via Stripe.
 * It retrieves order details using the payment intent ID from the URL
 * and displays a confirmation with order details and next steps.
 */

import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckCircle, Package, Truck, Mail, Loader2 } from 'lucide-react';
import SEO from '@/components/SEO';
import { Order } from '@/types/order';

interface OrderConfirmationState {
  order: Order | null;
  loading: boolean;
  error: string | null;
}

export function OrderConfirmation() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [state, setState] = useState<OrderConfirmationState>({
    order: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    const fetchOrderDetails = async () => {
      try {
        const paymentIntent = searchParams.get('payment_intent');

        if (!paymentIntent) {
          setState({
            order: null,
            loading: false,
            error: 'No payment intent found. Redirecting to collection...',
          });

          // Redirect after 3 seconds
          setTimeout(() => {
            navigate('/collection');
          }, 3000);

          return;
        }

        // Fetch order details using payment intent ID
        const response = await fetch(`/api/orders/by-payment-intent/${paymentIntent}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          throw new Error('Failed to fetch order details');
        }

        const order = (await response.json()) as Order;

        setState({
          order,
          loading: false,
          error: null,
        });
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to load order confirmation';

        setState({
          order: null,
          loading: false,
          error: errorMessage,
        });

        // Redirect after 5 seconds on error
        setTimeout(() => {
          navigate('/collection');
        }, 5000);
      }
    };

    fetchOrderDetails();
  }, [searchParams, navigate]);

  if (state.loading) {
    return (
      <>
        <SEO
          title="Confirming Order - Live It Iconic"
          description="Your order is being confirmed. Please wait..."
        />

        <div className="min-h-screen bg-lii-bg flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="w-12 h-12 text-lii-gold mx-auto mb-4 animate-spin" />
            <h1 className="text-2xl font-display text-lii-cloud">Confirming Your Order...</h1>
            <p className="text-lii-ash mt-2">Please wait while we process your payment.</p>
          </div>
        </div>
      </>
    );
  }

  if (state.error || !state.order) {
    return (
      <>
        <SEO
          title="Order Confirmation Error - Live It Iconic"
          description="We encountered an error while confirming your order."
        />

        <div className="min-h-screen bg-lii-bg pt-24">
          <div className="container mx-auto px-6 py-12">
            <div className="max-w-2xl mx-auto text-center">
              <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <CheckCircle className="w-12 h-12 text-red-500" />
              </div>
              <h1 className="text-4xl font-display font-semibold text-lii-cloud mb-4">
                Order Confirmation Error
              </h1>
              <p className="text-lii-ash font-ui text-lg mb-8">{state.error}</p>
              <p className="text-lii-ash text-sm mb-8">
                You will be redirected to our collection shortly...
              </p>
              <Button
                onClick={() => navigate('/collection')}
                variant="primary"
                className="font-ui font-medium"
              >
                Return to Collection
              </Button>
            </div>
          </div>
        </div>
      </>
    );
  }

  const { order } = state;

  return (
    <>
      <SEO
        title="Order Confirmed - Live It Iconic"
        description={`Order ${order.id} has been successfully confirmed. Thank you for your purchase!`}
      />

      <div className="min-h-screen bg-lii-bg pt-24">
        <div className="container mx-auto px-6 py-12">
          <div className="max-w-2xl mx-auto">
            {/* Success Header */}
            <div className="text-center mb-8">
              <div className="w-20 h-20 bg-lii-gold/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <CheckCircle className="w-12 h-12 text-lii-gold" />
              </div>
              <h1 className="text-4xl font-display font-semibold text-lii-cloud mb-4">
                Order Confirmed!
              </h1>
              <p className="text-lii-ash font-ui text-lg">
                Thank you for your purchase. Your order has been successfully placed.
              </p>
            </div>

            {/* Order Details */}
            <Card className="bg-lii-ink border-lii-gold/10 mb-8">
              <CardHeader>
                <CardTitle className="text-lii-cloud flex items-center gap-3">
                  <Package className="w-5 h-5 text-lii-gold" />
                  Order #{order.orderNumber}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Order Items */}
                <div className="space-y-4">
                  <h3 className="text-lii-cloud font-ui font-medium">Order Items</h3>
                  {order.items.map(item => (
                    <div key={item.id} className="flex gap-4 p-4 bg-lii-charcoal/20 rounded-lg">
                      <img
                        src={item.image}
                        alt={item.name}
                        className="w-16 h-16 object-cover rounded-md"
                      />
                      <div className="flex-1">
                        <h4 className="text-lii-cloud font-ui font-medium">{item.name}</h4>
                        {item.variant && (
                          <p className="text-lii-ash font-ui text-sm mt-1">{item.variant}</p>
                        )}
                        <p className="text-lii-gold font-ui text-sm mt-2">
                          ${item.price.toFixed(2)} × {item.quantity} = $
                          {(item.price * item.quantity).toFixed(2)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Order Summary */}
                <div className="border-t border-lii-gold/10 pt-4 space-y-2">
                  <div className="flex justify-between text-lii-ash font-ui text-sm">
                    <span>Subtotal</span>
                    <span>${order.subtotal.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-lii-ash font-ui text-sm">
                    <span>Shipping</span>
                    <span>${order.shipping.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-lii-ash font-ui text-sm">
                    <span>Tax</span>
                    <span>${order.tax.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-lii-cloud font-ui font-semibold text-lg border-t border-lii-gold/10 pt-2">
                    <span>Total</span>
                    <span>${order.total.toFixed(2)}</span>
                  </div>
                </div>

                {/* Shipping Address */}
                <div className="border-t border-lii-gold/10 pt-4">
                  <h3 className="text-lii-cloud font-ui font-medium mb-3 flex items-center gap-2">
                    <Truck className="w-4 h-4 text-lii-gold" />
                    Shipping Address
                  </h3>
                  <div className="text-lii-ash font-ui text-sm space-y-1">
                    <p>
                      {order.shippingAddress.firstName} {order.shippingAddress.lastName}
                    </p>
                    <p>{order.shippingAddress.address}</p>
                    <p>
                      {order.shippingAddress.city}, {order.shippingAddress.state}{' '}
                      {order.shippingAddress.zipCode}
                    </p>
                    <p>{order.shippingAddress.country}</p>
                    <p className="flex items-center gap-2 mt-2">
                      <Mail className="w-4 h-4" />
                      {order.shippingAddress.email}
                    </p>
                  </div>
                </div>

                {/* Payment Status */}
                <div className="border-t border-lii-gold/10 pt-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lii-cloud font-ui font-medium">Payment Status</h3>
                    <span className="px-3 py-1 bg-green-500/10 border border-green-500/30 text-green-400 text-sm font-ui rounded-full">
                      {order.paymentStatus === 'succeeded' ? 'Completed' : 'Processing'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Next Steps */}
            <Card className="bg-lii-ink border-lii-gold/10 mb-8">
              <CardHeader>
                <CardTitle className="text-lii-cloud">What's Next?</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-lii-gold/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Mail className="w-4 h-4 text-lii-gold" />
                  </div>
                  <div>
                    <h4 className="text-lii-cloud font-ui font-medium">Order Confirmation Email</h4>
                    <p className="text-lii-ash font-ui text-sm">
                      You'll receive a confirmation email shortly with your order details and tracking
                      information.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-lii-gold/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Package className="w-4 h-4 text-lii-gold" />
                  </div>
                  <div>
                    <h4 className="text-lii-cloud font-ui font-medium">Order Processing</h4>
                    <p className="text-lii-ash font-ui text-sm">
                      We'll process your order within 1-2 business days and send you shipping updates
                      via email.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-lii-gold/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Truck className="w-4 h-4 text-lii-gold" />
                  </div>
                  <div>
                    <h4 className="text-lii-cloud font-ui font-medium">Shipping</h4>
                    <p className="text-lii-ash font-ui text-sm">
                      Free worldwide shipping on orders over $100. Standard delivery takes 5-7
                      business days.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Button asChild className="flex-1 font-ui font-medium" variant="primary">
                <a href="/collection">Continue Shopping</a>
              </Button>
              <Button
                asChild
                variant="outline"
                className="flex-1 font-ui font-medium border-lii-gold/20 text-lii-ash hover:text-lii-cloud hover:border-lii-gold/40"
              >
                <a href="/contact">Contact Support</a>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default OrderConfirmation;
