import { X, Minus, Plus, ShoppingBag } from 'lucide-react';
import { useCart } from '@/contexts/CartContext';
import { Button } from '@/components/ui/button';
import { CartItem } from '@/types/cart';
import { Link } from 'react-router-dom';
import { useEffect, useRef, useState, useCallback, memo } from 'react';

interface CartDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * CartDrawer component displays a slide-out shopping cart with item management
 *
 * Shows all cart items with product details, quantity controls (increase/decrease), and remove buttons.
 * Displays cart total and provides actions to proceed to checkout or continue shopping. Includes
 * accessibility features like focus management and screen reader announcements for quantity changes.
 *
 * @component
 * @param {CartDrawerProps} props - Component props
 * @param {boolean} props.isOpen - Controls visibility of the cart drawer
 * @param {Function} props.onClose - Callback fired when drawer should close
 *
 * @example
 * <CartDrawer isOpen={isCartOpen} onClose={() => setCartOpen(false)} />
 *
 * @remarks
 * - Fetches cart data from CartContext
 * - Focuses first button when drawer opens for accessibility
 * - Announces cart updates to screen readers
 * - Quantity controls with +/- buttons
 * - Clear cart functionality
 */
const CartDrawer = memo(({ isOpen, onClose }: CartDrawerProps) => {
  const { items, total, itemCount, updateQuantity, removeItem, clearCart } = useCart();
  const [statusMessage, setStatusMessage] = useState('');
  const statusRef = useRef<HTMLDivElement>(null);
  const firstFocusableRef = useRef<HTMLButtonElement>(null);

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;

  // Focus management and status announcements
  useEffect(() => {
    if (isOpen && firstFocusableRef.current) {
      firstFocusableRef.current.focus();
    }
  }, [isOpen]);

  // Announce status changes to screen readers
  useEffect(() => {
    if (statusMessage && statusRef.current) {
      statusRef.current.focus();
    }
  }, [statusMessage]);

  const handleQuantityChange = useCallback(
    (itemId: string, newQuantity: number, itemName: string) => {
      updateQuantity(itemId, newQuantity);
      if (newQuantity === 0) {
        setStatusMessage(`${itemName} removed from cart`);
      } else {
        setStatusMessage(`Quantity updated to ${newQuantity} for ${itemName}`);
      }
    },
    [updateQuantity]
  );

  const handleRemoveItem = useCallback(
    (itemId: string, itemName: string) => {
      removeItem(itemId);
      setStatusMessage(`${itemName} removed from cart`);
    },
    [removeItem]
  );

  const handleClearCart = useCallback(() => {
    clearCart();
    setStatusMessage('Cart cleared');
  }, [clearCart]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className="fixed right-0 top-0 h-full w-full max-w-md bg-lii-ink border-l border-lii-gold/10 z-50 transform transition-transform duration-300 ease-in-out"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cart-title"
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-lii-gold/10">
            <h2 id="cart-title" className="text-xl font-display font-semibold text-lii-cloud">
              Shopping Cart ({itemCount})
            </h2>
            <button
              ref={firstFocusableRef}
              onClick={onClose}
              className="p-2 text-lii-ash hover:text-lii-cloud transition-colors duration-micro focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2 rounded-lg min-w-[44px] min-h-[44px] flex items-center justify-center"
              aria-label="Close shopping cart"
            >
              <X className="w-6 h-6" />
            </button>
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

          {/* Cart Items */}
          <div className="flex-1 overflow-y-auto p-6">
            {items.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <ShoppingBag className="w-16 h-16 text-lii-ash/50 mb-4" />
                <h3 className="text-lg font-ui font-medium text-lii-ash mb-2">
                  Your cart is empty
                </h3>
                <p className="text-lii-ash/70 font-ui text-sm">
                  Add some iconic products to get started
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {items.map((item: CartItem) => (
                  <div key={item.id} className="flex gap-4 p-4 bg-lii-charcoal/20 rounded-lg">
                    <img
                      src={item.image}
                      alt={item.name}
                      className="w-16 h-16 object-cover rounded-md flex-shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      <h4 className="font-ui font-medium text-lii-cloud text-sm truncate">
                        {item.name}
                      </h4>
                      {item.variant && (
                        <p className="text-lii-ash/70 font-ui text-xs mt-1">{item.variant}</p>
                      )}
                      <div className="flex items-center justify-between mt-2">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() =>
                              handleQuantityChange(item.id, item.quantity - 1, item.name)
                            }
                            className="min-w-[44px] min-h-[44px] flex items-center justify-center border border-lii-gold/20 rounded text-lii-ash hover:text-lii-cloud hover:border-lii-gold/40 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2"
                            aria-label={`Decrease quantity of ${item.name}`}
                          >
                            <Minus className="w-4 h-4" />
                          </button>
                          <span
                            className="flex-1 text-center text-lii-cloud font-ui text-sm"
                            aria-label={`Current quantity: ${item.quantity}`}
                          >
                            {item.quantity}
                          </span>
                          <button
                            onClick={() =>
                              handleQuantityChange(item.id, item.quantity + 1, item.name)
                            }
                            className="min-w-[44px] min-h-[44px] flex items-center justify-center border border-lii-gold/20 rounded text-lii-ash hover:text-lii-cloud hover:border-lii-gold/40 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2"
                            aria-label={`Increase quantity of ${item.name}`}
                          >
                            <Plus className="w-4 h-4" />
                          </button>
                        </div>
                        <button
                          onClick={() => handleRemoveItem(item.id, item.name)}
                          className="text-lii-ash/70 hover:text-lii-gold font-ui text-xs transition-colors min-h-[44px] flex items-center focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2 rounded px-2"
                          aria-label={`Remove ${item.name} from cart`}
                        >
                          Remove
                        </button>
                      </div>
                      <p className="text-lii-gold font-ui font-semibold text-sm mt-2">
                        {formatPrice(item.price * item.quantity)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          {items.length > 0 && (
            <div className="border-t border-lii-gold/10 p-6 space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-lii-ash font-ui font-medium">Total</span>
                <span className="text-lii-gold font-display font-semibold text-xl">
                  {formatPrice(total)}
                </span>
              </div>

              <div className="space-y-3">
                <Button
                  asChild
                  onClick={onClose}
                  className="w-full font-ui font-medium"
                  variant="primary"
                >
                  <Link to="/checkout">Checkout</Link>
                </Button>
                <Button
                  onClick={handleClearCart}
                  variant="outline"
                  className="w-full font-ui font-medium border-lii-gold/20 text-lii-ash hover:text-lii-cloud hover:border-lii-gold/40"
                  aria-label="Clear all items from cart"
                >
                  Clear Cart
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
});

CartDrawer.displayName = 'CartDrawer';

export default CartDrawer;
