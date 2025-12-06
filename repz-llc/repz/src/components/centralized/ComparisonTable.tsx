// Centralized Feature Comparison Table - Industry Standard
// Responsive table with tier-based theming

import React from 'react';
import { Card } from '@/ui/molecules/Card';
import { Check, X } from 'lucide-react';
import { getTierConfigByType } from '@/constants/tiers';
import type { TierType } from '@/types/business';

interface ComparisonTableProps {
  tiers: TierType[];
  features: Array<{
    category: string;
    items: Array<{
      name: string;
      core: boolean | string;
      adaptive: boolean | string;
      performance: boolean | string;
      longevity: boolean | string;
    }>;
  }>;
  className?: string;
}

export function ComparisonTable({ tiers, features, className = '' }: ComparisonTableProps) {
  const renderFeatureValue = (value: boolean | string, tier: TierType) => {
    const config = getTierConfigByType(tier);
    
    if (typeof value === 'boolean') {
      return value ? (
        <Check className={`h-5 w-5`} style={{ color: config.color }} />
      ) : (
        <X className="h-5 w-5 text-muted-foreground opacity-50" />
      );
    }
    
    return (
      <span className="text-sm font-medium" style={{ color: config.color }}>
        {value}
      </span>
    );
  };

  return (
    <Card className={`bg-card border overflow-hidden ${className}`}>
      {/* Mobile-First: Stack on small screens */}
      <div className="block md:hidden">
        {tiers.map((tier) => {
          const config = getTierConfigByType(tier);
          return (
            <div key={tier} className="border-b last:border-b-0">
              {/* Tier Header */}
              <div 
                className="p-4 font-semibold text-white"
                style={{ backgroundColor: config.color }}
              >
                {config.displayName}
                <div className="text-sm opacity-90">${config.monthlyPrice}/month</div>
              </div>
              
              {/* Features for this tier */}
              <div className="p-4 space-y-4">
                {features.map((category, categoryIndex) => (
                  <div key={categoryIndex}>
                    <h4 className="font-medium text-foreground mb-2 text-sm uppercase tracking-wide">
                      {category.category}
                    </h4>
                    <div className="space-y-2">
                      {category.items.map((item, itemIndex) => (
                        <div key={itemIndex} className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">{item.name}</span>
                          <div className="flex justify-center">
                            {renderFeatureValue(item[tier], tier)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Desktop: Traditional table layout */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full">
          {/* Header */}
          <thead>
            <tr className="border-b">
              <th className="text-left p-4 font-medium text-foreground">Features</th>
              {tiers.map((tier) => {
                const config = getTierConfigByType(tier);
                return (
                  <th key={tier} className="text-center p-4">
                    <div className="space-y-2">
                      <div 
                        className="font-semibold text-white px-3 py-2 rounded-lg"
                        style={{ backgroundColor: config.color }}
                      >
                        {config.displayName}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        ${config.monthlyPrice}/month
                      </div>
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>

          {/* Body */}
          <tbody>
            {features.map((category, categoryIndex) => (
              <React.Fragment key={categoryIndex}>
                {/* Category Header */}
                <tr className="border-b bg-muted/30">
                  <td colSpan={tiers.length + 1} className="p-4 font-medium text-foreground text-sm uppercase tracking-wide">
                    {category.category}
                  </td>
                </tr>
                
                {/* Category Items */}
                {category.items.map((item, itemIndex) => (
                  <tr key={itemIndex} className="border-b hover:bg-muted/20 transition-colors">
                    <td className="p-4 text-sm text-muted-foreground">
                      {item.name}
                    </td>
                    {tiers.map((tier) => (
                      <td key={tier} className="p-4 text-center">
                        {renderFeatureValue(item[tier], tier)}
                      </td>
                    ))}
                  </tr>
                ))}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}