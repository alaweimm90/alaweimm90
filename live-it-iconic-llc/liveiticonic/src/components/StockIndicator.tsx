import React from 'react';
import { Badge } from '@/components/ui/badge';

interface StockIndicatorProps {
  quantity: number;
  showLabel?: boolean;
  restockDate?: string;
  lowStockThreshold?: number;
}

/**
 * StockIndicator component displays inventory status with visual feedback
 *
 * Shows product stock status with color-coded badges:
 * - Green: In Stock
 * - Orange: Low Stock (quantity < lowStockThreshold)
 * - Red: Out of Stock
 * - Gold: Pre-Order available
 *
 * @component
 * @param {StockIndicatorProps} props - Component props
 * @param {number} props.quantity - Available quantity
 * @param {boolean} [props.showLabel=true] - Show status label text
 * @param {string} [props.restockDate] - Expected restock date
 * @param {number} [props.lowStockThreshold=5] - Quantity threshold for low stock state
 *
 * @example
 * <StockIndicator quantity={0} showLabel={true} />
 * <StockIndicator quantity={3} restockDate="2024-12-15" />
 */
const StockIndicator: React.FC<StockIndicatorProps> = ({
  quantity,
  showLabel = true,
  restockDate,
  lowStockThreshold = 5,
}) => {
  const getStatus = () => {
    if (quantity === 0 && restockDate) {
      return { status: 'pre-order', label: 'Pre-Order', color: 'bg-amber-600' };
    }
    if (quantity === 0) {
      return { status: 'out-of-stock', label: 'Out of Stock', color: 'bg-red-600' };
    }
    if (quantity < lowStockThreshold) {
      return { status: 'low-stock', label: `Only ${quantity} Left!`, color: 'bg-orange-600' };
    }
    return { status: 'in-stock', label: 'In Stock', color: 'bg-emerald-600' };
  };

  const { status, label, color } = getStatus();

  return (
    <div className="flex flex-col gap-2">
      <Badge className={`${color} text-white font-ui text-xs px-3 py-1 w-fit`}>
        {showLabel ? label : ''}
      </Badge>
      {status === 'pre-order' && restockDate && (
        <p className="text-xs text-lii-ash font-ui">Expected: {new Date(restockDate).toLocaleDateString()}</p>
      )}
    </div>
  );
};

export default StockIndicator;
