import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { useCart } from '@/contexts/CartContext';
import { ShippingAddress } from '@/types/order';

interface OrderSummaryProps {
  shippingData?: ShippingAddress | null;
}

/**
 * OrderSummary component displays a sticky order breakdown with items, costs, and shipping address
 *
 * Shows cart items with product details, calculates subtotal, shipping (free over $100),
 * and tax (8%). Displays final total and previews shipping address if provided. Sticky
 * positioning keeps it visible during checkout form scrolling.
 *
 * @component
 * @param {OrderSummaryProps} props - Component props
 * @param {ShippingAddress} [props.shippingData] - Optional shipping address to preview in summary
 *
 * @example
 * <OrderSummary shippingData={customerAddress} />
 *
 * @remarks
 * - Free shipping for orders over $100
 * - Tax rate hardcoded at 8%
 * - Cart items fetched from CartContext
 */
export const OrderSummary: React.FC<OrderSummaryProps> = ({ shippingData }) => {
  const { items, total, itemCount } = useCart();

  const shipping = total > 100 ? 0 : 15;
  const tax = total * 0.08; // 8% tax
  const finalTotal = total + shipping + tax;

  return (
    <Card className="bg-lii-ink border-lii-gold/10 sticky top-24">
      <CardHeader>
        <CardTitle className="text-lii-cloud font-display">Order Summary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Items */}
        <div className="space-y-3 max-h-64 overflow-y-auto">
          {items.map(item => (
            <div key={item.id} className="flex gap-3">
              <img
                src={item.image}
                alt={item.name}
                className="w-16 h-16 object-cover rounded flex-shrink-0"
              />
              <div className="flex-1 min-w-0">
                <p className="text-lii-cloud font-ui text-sm truncate">{item.name}</p>
                {item.variant && (
                  <p className="text-lii-ash font-ui text-xs mt-1">{item.variant}</p>
                )}
                <p className="text-lii-gold font-ui text-sm mt-1">
                  ${item.price.toFixed(2)} × {item.quantity}
                </p>
              </div>
            </div>
          ))}
        </div>

        <Separator className="bg-lii-gold/10" />

        {/* Totals */}
        <div className="space-y-2">
          <div className="flex justify-between text-lii-ash font-ui text-sm">
            <span>Subtotal ({itemCount} items)</span>
            <span>${total.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-lii-ash font-ui text-sm">
            <span>Shipping</span>
            <span>{shipping === 0 ? 'Free' : `$${shipping.toFixed(2)}`}</span>
          </div>
          <div className="flex justify-between text-lii-ash font-ui text-sm">
            <span>Tax</span>
            <span>${tax.toFixed(2)}</span>
          </div>
          <Separator className="bg-lii-gold/10" />
          <div className="flex justify-between text-lii-cloud font-ui font-semibold text-lg">
            <span>Total</span>
            <span>${finalTotal.toFixed(2)}</span>
          </div>
        </div>

        {/* Shipping Address Preview */}
        {shippingData && (
          <>
            <Separator className="bg-lii-gold/10" />
            <div className="space-y-1">
              <p className="text-lii-ash font-ui text-xs uppercase tracking-wider">Shipping To</p>
              <p className="text-lii-cloud font-ui text-sm">
                {shippingData.firstName} {shippingData.lastName}
              </p>
              <p className="text-lii-ash font-ui text-xs">
                {shippingData.address}, {shippingData.city}, {shippingData.state}{' '}
                {shippingData.zipCode}
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default OrderSummary;
