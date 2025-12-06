import React from 'react';
import { Badge } from '@/components/ui/badge';

interface InventoryBadgeProps {
  quantity: number;
  trackInventory: boolean;
  className?: string;
}

/**
 * InventoryBadge component displays inventory status with color-coded badges
 *
 * Shows product inventory status based on quantity levels when inventory tracking is enabled.
 * Displays different colored badges for critical inventory states: Out of Stock (red),
 * Low Stock (amber), Limited Quantity (yellow), or no badge if stock is adequate.
 *
 * @component
 * @param {InventoryBadgeProps} props - Component props
 * @param {number} props.quantity - Current inventory quantity
 * @param {boolean} props.trackInventory - Whether to display inventory status badge
 * @param {string} [props.className] - Optional CSS class for styling
 *
 * @example
 * <InventoryBadge quantity={2} trackInventory={true} />
 *
 * @remarks
 * - Returns null if trackInventory is false
 * - Out of Stock: quantity === 0 (red)
 * - Only N left: quantity < 5 (yellow)
 * - Low Stock: quantity < 10 (amber)
 * - In Stock: quantity >= 10 (no badge)
 */
export const InventoryBadge: React.FC<InventoryBadgeProps> = ({
  quantity,
  trackInventory,
  className = '',
}) => {
  if (!trackInventory) {
    return null;
  }

  if (quantity === 0) {
    return (
      <Badge
        className={`bg-red-500/10 text-red-400 border-red-500/20 ${className}`}
        variant="outline"
      >
        Out of Stock
      </Badge>
    );
  }

  if (quantity < 5) {
    return (
      <Badge
        className={`bg-yellow-500/10 text-yellow-400 border-yellow-500/20 ${className}`}
        variant="outline"
      >
        Only {quantity} left
      </Badge>
    );
  }

  if (quantity < 10) {
    return (
      <Badge
        className={`bg-amber-500/10 text-amber-400 border-amber-500/20 ${className}`}
        variant="outline"
      >
        Low Stock
      </Badge>
    );
  }

  return null;
};

export default InventoryBadge;
