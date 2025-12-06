import React from 'react';
import { Check, X } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { cn } from '@/lib/utils';

interface FeatureComparisonProps {
  tiers: Array<{
    id: string;
    name: string;
    features: string[];
  }>;
  allFeatures: string[];
}

export const FeatureComparisonTable: React.FC<FeatureComparisonProps> = ({
  tiers,
  allFeatures
}) => {
  const hasFeature = (tierFeatures: string[], feature: string) => {
    return tierFeatures.some(tf => 
      tf.toLowerCase().includes(feature.toLowerCase()) ||
      feature.toLowerCase().includes(tf.toLowerCase())
    );
  };

  const getIconStyle = (tierId: string, hasFeature: boolean) => {
    if (!hasFeature) {
      return 'text-red-400';
    }
    
    // Gold for longevity tier
    if (tierId === 'longevity') {
      return 'text-amber-400';
    }
    
    return 'text-emerald-400';
  };

  return (
    <Card className="mt-12">
      <CardHeader>
        {/* Removed "Complete Feature Comparison" title as requested */}
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-4 font-semibold">Features</th>
                {tiers.map(tier => (
                  <th key={tier.id} className="text-center p-4 font-semibold min-w-[120px]">
                    {tier.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {allFeatures.map((feature, index) => (
                <tr key={index} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="p-4 text-sm font-medium">
                    {feature}
                  </td>
                  {tiers.map(tier => {
                    const included = hasFeature(tier.features, feature);
                    return (
                      <td key={`${tier.id}-${index}`} className="p-4 text-center">
                        {included ? (
                          <Check className={cn(
                            'w-5 h-5 mx-auto',
                            getIconStyle(tier.id, true)
                          )} />
                        ) : (
                          <X className={cn(
                            'w-5 h-5 mx-auto',
                            getIconStyle(tier.id, false)
                          )} />
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};