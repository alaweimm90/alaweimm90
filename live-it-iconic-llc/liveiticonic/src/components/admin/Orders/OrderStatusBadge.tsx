/**
 * Order Status Badge Component
 * Displays order status with appropriate styling
 */

interface OrderStatusBadgeProps {
  status: string;
}

const statusConfig = {
  pending: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Pending' },
  processing: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Processing' },
  shipped: { bg: 'bg-purple-100', text: 'text-purple-800', label: 'Shipped' },
  delivered: { bg: 'bg-green-100', text: 'text-green-800', label: 'Delivered' },
  cancelled: { bg: 'bg-red-100', text: 'text-red-800', label: 'Cancelled' },
};

export function OrderStatusBadge({ status }: OrderStatusBadgeProps) {
  const config =
    statusConfig[status.toLowerCase() as keyof typeof statusConfig] ||
    statusConfig.pending;

  return (
    <span
      className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${config.bg} ${config.text}`}
    >
      {config.label}
    </span>
  );
}
