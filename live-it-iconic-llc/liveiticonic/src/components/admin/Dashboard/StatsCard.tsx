/**
 * Stats Card Component
 * Displays key metrics with icon, value, and change percentage
 */
import { LucideIcon } from 'lucide-react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StatsCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
}

export function StatsCard({
  icon: Icon,
  label,
  value,
  change,
  changeLabel,
}: StatsCardProps) {
  const isPositive = change && change > 0;

  return (
    <div className="bg-white rounded-lg p-6 border border-lii-cloud shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <div className="p-3 bg-lii-gold/10 rounded-lg">
          <Icon className="text-lii-gold" size={24} />
        </div>
        {change !== undefined && (
          <div
            className={`flex items-center gap-1 text-sm font-semibold ${
              isPositive ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {isPositive ? (
              <TrendingUp size={16} />
            ) : (
              <TrendingDown size={16} />
            )}
            {isPositive ? '+' : ''}{change}%
          </div>
        )}
      </div>
      <p className="text-2xl font-bold text-lii-bg">{value}</p>
      <p className="text-sm text-lii-ash">{label}</p>
      {changeLabel && (
        <p className="text-xs text-lii-ash mt-2">{changeLabel}</p>
      )}
    </div>
  );
}
