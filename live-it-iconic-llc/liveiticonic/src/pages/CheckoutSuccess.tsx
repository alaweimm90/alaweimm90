import { useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { Order } from '@/types/order';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckCircle, Package, Truck, Mail } from 'lucide-react';
import SEO from '@/components/SEO';

const CheckoutSuccess = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const order = location.state?.order as Order | undefined;

  useEffect(() => {
    if (!order) {
      navigate('/collection');
    }
  }, [order, navigate]);

  if (!order) {
    return null;
  }

  return (
    <>
      <SEO
        title="Order Confirmed - Live It Iconic"
        description="Your order has been successfully placed. Thank you for shopping with Live It Iconic."
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
                  Order #{order.id}
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
                    <h4 className="text-lii-cloud font-ui font-medium">Order Confirmation</h4>
                    <p className="text-lii-ash font-ui text-sm">
                      You'll receive an email confirmation with your order details and tracking
                      information.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-lii-gold/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Package className="w-4 h-4 text-lii-gold" />
                  </div>
                  <div>
                    <h4 className="text-lii-cloud font-ui font-medium">Processing</h4>
                    <p className="text-lii-ash font-ui text-sm">
                      We'll process your order within 1-2 business days and send you shipping
                      updates.
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
                <Link to="/collection">Continue Shopping</Link>
              </Button>
              <Button
                asChild
                variant="outline"
                className="flex-1 font-ui font-medium border-lii-gold/20 text-lii-ash hover:text-lii-cloud hover:border-lii-gold/40"
              >
                <Link to="/contact">Contact Support</Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default CheckoutSuccess;
