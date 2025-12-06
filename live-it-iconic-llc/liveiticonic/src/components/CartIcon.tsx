import { ShoppingBag } from 'lucide-react';
import { useCart } from '@/contexts/CartContext';

interface CartIconProps {
  onClick: () => void;
  className?: string;
}

/**
 * CartIcon component displays a shopping bag icon with item count badge
 *
 * Shows a clickable shopping cart icon with optional custom className. Displays
 * a gold badge with current item count in the cart. Badge shows '99+' for counts
 * over 99 items. Updates dynamically when cart contents change.
 *
 * @component
 * @param {CartIconProps} props - Component props
 * @param {Function} props.onClick - Callback fired when icon is clicked
 * @param {string} [props.className] - Optional CSS class for styling
 *
 * @example
 * <CartIcon onClick={() => openCart()} className="text-lii-gold" />
 *
 * @remarks
 * - Fetches item count from CartContext
 * - Badge only displays if itemCount > 0
 * - Accessible with proper aria-label
 */
const CartIcon = ({ onClick, className = '' }: CartIconProps) => {
  const { itemCount } = useCart();

  return (
    <button
      onClick={onClick}
      className={`relative p-2 text-lii-ash hover:text-lii-cloud transition-colors duration-micro ${className}`}
      aria-label={`Shopping cart with ${itemCount} items`}
    >
      <ShoppingBag className="w-6 h-6" />
      {itemCount > 0 && (
        <span className="absolute -top-1 -right-1 bg-lii-gold text-lii-ink text-xs font-semibold rounded-full h-5 w-5 flex items-center justify-center min-w-[20px]">
          {itemCount > 99 ? '99+' : itemCount}
        </span>
      )}
    </button>
  );
};

export default CartIcon;
