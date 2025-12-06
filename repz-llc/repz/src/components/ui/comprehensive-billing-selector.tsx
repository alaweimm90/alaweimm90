import React from 'react';
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Card, CardContent } from "@/ui/molecules/Card";
import { Check } from 'lucide-react';
import { BillingCycle, BILLING_CYCLE_CONFIG } from '@/constants/stripeConfig';

interface ComprehensiveBillingSelectorProps {
  selectedCycle: BillingCycle;
  onCycleChange: (cycle: BillingCycle) => void;
  monthlyPrice: number;
  className?: string;
  showAnchoringDefault?: boolean; // Psychology: default to annual
}

export const ComprehensiveBillingSelector: React.FC<ComprehensiveBillingSelectorProps> = ({
  selectedCycle,
  onCycleChange,
  monthlyPrice,
  className = "",
  showAnchoringDefault = true
}) => {
  
  const getDisplayPrice = (cycle: BillingCycle) => {
    const config = BILLING_CYCLE_CONFIG[cycle];
    const discountMultiplier = (100 - config.discount) / 100;
    return Math.round(monthlyPrice * discountMultiplier * 100) / 100;
  };

  const getTotalPrice = (cycle: BillingCycle) => {
    const config = BILLING_CYCLE_CONFIG[cycle];
    return getDisplayPrice(cycle) * config.multiplier;
  };

  const getSavingsAmount = (cycle: BillingCycle) => {
    if (cycle === 'monthly') return 0;
    const config = BILLING_CYCLE_CONFIG[cycle];
    const monthlyTotal = monthlyPrice * config.multiplier;
    const discountedTotal = getTotalPrice(cycle);
    return monthlyTotal - discountedTotal;
  };

  const isPopular = (cycle: BillingCycle) => cycle === 'annual';
  const isBestValue = (cycle: BillingCycle) => cycle === 'annual';

  return (
    <Card className={`w-full ${className}`}>
      <CardContent className="p-6">
        <div className="text-center mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Choose Your Billing Period</h3>
          <p className="text-sm text-gray-600">
            {showAnchoringDefault && (
              <><span className="font-medium text-green-600">Annual billing saves you ${getSavingsAmount('annual').toFixed(0)}/year</span> - Most athletes choose this</>
            )}
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {(Object.keys(BILLING_CYCLE_CONFIG) as BillingCycle[]).map((cycle) => {
            const config = BILLING_CYCLE_CONFIG[cycle];
            const displayPrice = getDisplayPrice(cycle);
            const totalPrice = getTotalPrice(cycle);
            const savings = getSavingsAmount(cycle);
            const isSelected = selectedCycle === cycle;
            
            return (
              <div 
                key={cycle}
                className={`
                  relative rounded-lg border-2 cursor-pointer transition-all duration-200 p-4
                  ${isSelected 
                    ? 'border-primary bg-primary/5 shadow-md md:scale-105' 
                    : 'border-gray-200 bg-white hover:border-gray-300 md:hover:scale-102'
                  }
                  ${isBestValue(cycle) ? 'ring-2 ring-green-200' : ''}
                `}
                onClick={() => onCycleChange(cycle)}
              >
                {/* Popular/Best Value Badges */}
                {isPopular(cycle) && (
                  <div className="absolute -top-2 left-1/2 transform -translate-x-1/2">
                    <Badge className="bg-orange-500 text-white text-xs px-2 py-1">
                      MOST POPULAR
                    </Badge>
                  </div>
                )}
                
                {isBestValue(cycle) && !isPopular(cycle) && (
                  <div className="absolute -top-2 right-2">
                    <Badge className="bg-green-500 text-white text-xs px-2 py-1">
                      BEST VALUE
                    </Badge>
                  </div>
                )}

                {/* Savings Badge */}
                {savings > 0 && (
                  <div className="absolute -top-2 right-2">
                    <Badge className="bg-green-500 text-white text-xs px-2 py-1">
                      {config.savings}
                    </Badge>
                  </div>
                )}
                
                <div className="text-center">
                  {/* Selection Indicator */}
                  <div className="flex items-center justify-center mb-3">
                    <div className={`
                      w-5 h-5 rounded-full border-2 mr-3 flex items-center justify-center
                      ${isSelected 
                        ? 'border-primary bg-primary' 
                        : 'border-gray-300'
                      }
                    `}>
                      {isSelected && (
                        <Check className="w-3 h-3 text-white" />
                      )}
                    </div>
                    <span className="font-semibold text-gray-900">{config.displayName}</span>
                  </div>
                  
                  {/* Price Display */}
                  <div className="text-2xl font-bold text-gray-900 mb-1">
                    ${displayPrice}
                    <span className="text-sm font-normal text-gray-600">/month</span>
                  </div>
                  
                  {/* Original Price (if discounted) */}
                  {config.discount > 0 && (
                    <div className="text-sm text-gray-500 mb-2">
                      <span className="line-through text-gray-400">${monthlyPrice}/month</span>
                      <span className="text-green-600 font-medium ml-2">{config.discount}% off</span>
                    </div>
                  )}
                  
                  {/* Billing Details */}
                  <div className="text-xs text-gray-500">
                    {cycle === 'monthly' ? (
                      'Billed monthly'
                    ) : (
                      <>
                        Billed every {config.multiplier} months
                        <br />
                        (${totalPrice.toFixed(0)} total)
                      </>
                    )}
                  </div>
                  
                  {/* Savings Amount */}
                  {savings > 0 && (
                    <div className="text-xs text-green-600 font-medium mt-1">
                      Save ${savings.toFixed(0)} total
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Summary */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="text-center text-sm text-gray-600">
            {selectedCycle === 'monthly' ? (
              <>You'll be charged <strong>${monthlyPrice}</strong> every month</>
            ) : (
              <>
                You'll be charged <strong>${getTotalPrice(selectedCycle).toFixed(0)}</strong> every {BILLING_CYCLE_CONFIG[selectedCycle].multiplier} months
                {getSavingsAmount(selectedCycle) > 0 && (
                  <>, saving <strong>${getSavingsAmount(selectedCycle).toFixed(0)}</strong> compared to monthly billing</>
                )}
              </>
            )}
          </div>
          
          {/* Psychology: Anchoring Effect */}
          {selectedCycle !== 'annual' && showAnchoringDefault && (
            <div className="text-center text-xs text-orange-600 mt-2">
              Switch to annual and save ${getSavingsAmount('annual').toFixed(0)} per year
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};