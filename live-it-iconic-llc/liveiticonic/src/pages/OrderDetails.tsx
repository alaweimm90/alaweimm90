import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate } from 'react-router-dom';
import { orderService } from '@/services/orderService';
import { Order } from '@/types/order';
import SEO from '@/components/SEO';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ArrowLeft, Package, Truck, CheckCircle, Clock, XCircle } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';

const OrderDetails: React.FC = () => {
  const { orderId } = useParams<{ orderId: string }>();
  const navigate = useNavigate();

  const { data: order, isLoading } = useQuery({
    queryKey: ['order', orderId],
    queryFn: () => orderService.getOrder(orderId!),
    enabled: !!orderId,
  });

  const getStatusIcon = (status: Order['status']) => {
    switch (status) {
      case 'delivered':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'shipped':
      case 'processing':
        return <Truck className="w-5 h-5 text-blue-400" />;
      case 'pending':
      case 'confirmed':
        return <Clock className="w-5 h-5 text-yellow-400" />;
      case 'cancelled':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Package className="w-5 h-5" />;
    }
  };

  const getStatusColor = (status: Order['status']) => {
    switch (status) {
      case 'delivered':
        return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'shipped':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'processing':
        return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      case 'pending':
      case 'confirmed':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'cancelled':
        return 'bg-red-500/10 text-red-400 border-red-500/20';
      default:
        return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  if (isLoading) {
    return (
      <>
        <SEO title="Order Details - Live It Iconic" description="View your order details" />
        <div className="min-h-screen bg-lii-bg">
          <Navigation />
          <div className="container mx-auto px-6 py-12 pt-32">
            <Skeleton className="h-8 w-64 mb-8" />
            <Skeleton className="h-96 w-full" />
          </div>
          <Footer />
        </div>
      </>
    );
  }

  if (!order) {
    return (
      <>
        <SEO title="Order Not Found - Live It Iconic" description="Order not found" />
        <div className="min-h-screen bg-lii-bg">
          <Navigation />
          <div className="container mx-auto px-6 py-12 pt-32 text-center">
            <h1 className="text-2xl font-display text-lii-cloud mb-4">Order Not Found</h1>
            <Button onClick={() => navigate('/shop')} variant="primary">
              Continue Shopping
            </Button>
          </div>
          <Footer />
        </div>
      </>
    );
  }

  return (
    <>
      <SEO
        title={`Order ${order.orderNumber} - Live It Iconic`}
        description={`Order details for ${order.orderNumber}`}
      />
      <div className="min-h-screen bg-lii-bg">
        <Navigation />
        <main className="pt-32 pb-16">
          <div className="container mx-auto px-6">
            {/* Header */}
            <div className="flex items-center gap-4 mb-8">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate(-1)}
                className="text-lii-ash hover:text-lii-cloud"
              >
                <ArrowLeft className="w-5 h-5" />
              </Button>
              <div>
                <h1 className="text-3xl font-display font-semibold text-lii-cloud">
                  Order {order.orderNumber}
                </h1>
                <p className="text-lii-ash font-ui text-sm mt-1">
                  Placed on {new Date(order.createdAt).toLocaleDateString()}
                </p>
              </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Main Content */}
              <div className="lg:col-span-2 space-y-6">
                {/* Order Status */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3 text-lii-cloud">
                      {getStatusIcon(order.status)}
                      Order Status
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between">
                      <Badge className={getStatusColor(order.status)} variant="outline">
                        {order.status.charAt(0).toUpperCase() + order.status.slice(1)}
                      </Badge>
                      <Badge
                        className={
                          order.paymentStatus === 'paid'
                            ? 'bg-green-500/10 text-green-400 border-green-500/20'
                            : 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
                        }
                        variant="outline"
                      >
                        {order.paymentStatus.charAt(0).toUpperCase() + order.paymentStatus.slice(1)}
                      </Badge>
                    </div>
                    {order.trackingNumber && (
                      <div className="mt-4">
                        <p className="text-lii-ash font-ui text-sm mb-2">Tracking Number</p>
                        <p className="text-lii-cloud font-ui font-mono">{order.trackingNumber}</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Order Items */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardHeader>
                    <CardTitle className="text-lii-cloud">Order Items</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {order.items.length > 0 ? (
                      <div className="space-y-4">
                        {order.items.map(item => (
                          <div key={item.id} className="flex gap-4">
                            <img
                              src={item.image}
                              alt={item.name}
                              className="w-16 h-16 object-cover rounded"
                            />
                            <div className="flex-1">
                              <p className="text-lii-cloud font-ui font-medium">{item.name}</p>
                              {item.variant && (
                                <p className="text-lii-ash font-ui text-sm">{item.variant}</p>
                              )}
                              <p className="text-lii-gold font-ui text-sm mt-1">
                                ${item.price.toFixed(2)} × {item.quantity}
                              </p>
                            </div>
                            <p className="text-lii-cloud font-ui font-semibold">
                              ${(item.price * item.quantity).toFixed(2)}
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-lii-ash font-ui text-center py-8">
                        No items in this order
                      </p>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                {/* Shipping Address */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardHeader>
                    <CardTitle className="text-lii-cloud">Shipping Address</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1 text-lii-ash font-ui text-sm">
                      <p className="text-lii-cloud font-medium">
                        {order.shippingAddress.firstName} {order.shippingAddress.lastName}
                      </p>
                      <p>{order.shippingAddress.address}</p>
                      <p>
                        {order.shippingAddress.city}, {order.shippingAddress.state}{' '}
                        {order.shippingAddress.zipCode}
                      </p>
                      <p>{order.shippingAddress.country}</p>
                    </div>
                  </CardContent>
                </Card>

                {/* Order Summary */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardHeader>
                    <CardTitle className="text-lii-cloud">Order Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
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
                    <Separator className="bg-lii-gold/10" />
                    <div className="flex justify-between text-lii-cloud font-ui font-semibold text-lg">
                      <span>Total</span>
                      <span>${order.total.toFixed(2)}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </main>
        <Footer />
      </div>
    </>
  );
};

export default OrderDetails;
