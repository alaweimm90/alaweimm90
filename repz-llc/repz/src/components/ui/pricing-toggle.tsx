import { cn } from "@/lib/utils";

type BillingPeriod = 'monthly' | 'quarterly' | 'semi-annual' | 'annual';

interface PricingToggleProps {
  billingPeriod: BillingPeriod;
  onToggle: (period: BillingPeriod) => void;
  className?: string;
}

const billingOptions = [
  { id: 'monthly', label: 'Monthly', savings: null },
  { id: 'quarterly', label: 'Quarterly', savings: '+5%' },
  { id: 'semi-annual', label: 'Semi-Annual', savings: '+10%' },
  { id: 'annual', label: 'Annual', savings: '+20%' }
] as const;

export const PricingToggle = ({ billingPeriod = 'annual', onToggle, className }: PricingToggleProps) => {
  return (
    <div className={cn("flex items-center justify-center mb-8", className)}>
      <div className="bg-gradient-to-r from-gray-900/90 via-slate-800/95 to-gray-900/90 backdrop-blur-xl border border-gray-600/40 rounded-2xl p-2 shadow-xl">
        {/* Desktop: Single row flex layout */}
        <div className="hidden sm:flex flex-wrap gap-2">
          {billingOptions.map((option) => (
            <button
              key={option.id}
              onClick={() => onToggle(option.id)}
              className={cn(
                "px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-300 relative group shadow-md",
                billingPeriod === option.id
                  ? "bg-gradient-to-r from-red-600 to-red-500 text-white shadow-red-500/25 ring-2 ring-red-400/50 transform scale-105"
                  : "bg-gradient-to-r from-gray-700 to-gray-600 text-gray-200 hover:from-gray-600 hover:to-gray-500 hover:text-white hover:shadow-gray-500/20 hover:scale-102"
              )}
            >
              {option.label}
              {option.savings && (
                <span className="absolute -top-2 -right-2 bg-gradient-to-r from-emerald-500 to-green-600 text-xs px-2 py-1 rounded-full text-white font-bold shadow-lg animate-pulse">
                  {option.savings}
                </span>
              )}
            </button>
          ))}
        </div>
        
        {/* Mobile: 2x2 Grid layout */}
        <div className="grid grid-cols-2 gap-2 sm:hidden">
          {billingOptions.map((option) => (
            <button
              key={option.id}
              onClick={() => onToggle(option.id)}
              className={cn(
                "px-4 py-3 rounded-xl text-xs font-semibold transition-all duration-300 relative group shadow-md min-w-[120px]",
                billingPeriod === option.id
                  ? "bg-gradient-to-r from-red-600 to-red-500 text-white shadow-red-500/25 ring-2 ring-red-400/50 transform scale-105"
                  : "bg-gradient-to-r from-gray-700 to-gray-600 text-gray-200 hover:from-gray-600 hover:to-gray-500 hover:text-white hover:shadow-gray-500/20 hover:scale-102"
              )}
            >
              {option.label}
              {option.savings && (
                <span className="absolute -top-1 -right-1 bg-gradient-to-r from-emerald-500 to-green-600 text-xs px-1.5 py-0.5 rounded-full text-white font-bold shadow-lg animate-pulse text-[10px]">
                  {option.savings}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export type { BillingPeriod };