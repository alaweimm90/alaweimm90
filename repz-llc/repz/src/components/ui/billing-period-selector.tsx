import React from 'react';
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Card, CardContent } from "@/ui/molecules/Card";

interface BillingPeriodSelectorProps {
  selectedPeriod: 'monthly' | 'annual';
  onPeriodChange: (period: 'monthly' | 'annual') => void;
  monthlyPrice: number;
  annualPrice?: number; // If not provided, will calculate 20% discount
  className?: string;
  defaultToAnnual?: boolean; // Anchoring effect - default true
}

export const BillingPeriodSelector: React.FC<BillingPeriodSelectorProps> = ({
  selectedPeriod,
  onPeriodChange,
  monthlyPrice,
  annualPrice,
  className = "",
  defaultToAnnual = true
}) => {
  const calculatedAnnualPrice = annualPrice || Math.round(monthlyPrice * 0.8 * 100) / 100; // 20% discount
  const annualSavings = monthlyPrice - calculatedAnnualPrice;
  const yearlyTotal = calculatedAnnualPrice * 12;
  const monthlyTotal = monthlyPrice * 12;
  const totalSavings = monthlyTotal - yearlyTotal;

  return (
    <Card className={`w-full ${className}`}>
      <CardContent className="p-6">
        <div className="text-center mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Choose Your Billing Period</h3>
          <p className="text-sm text-gray-600">
            {defaultToAnnual ? (
              <><span className="font-medium text-green-600">Annual billing saves you ${totalSavings.toFixed(0)}/year</span> - Most athletes choose this</>
            ) : (
              'Save 20% with annual billing'
            )}
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Monthly Option */}
          <div 
            className={`
              relative rounded-lg border-2 cursor-pointer transition-all duration-200 p-4
              ${selectedPeriod === 'monthly' 
                ? 'border-blue-500 bg-blue-50 shadow-md' 
                : 'border-gray-200 bg-white hover:border-gray-300'
              }
            `}
            onClick={() => onPeriodChange('monthly')}
          >
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <div className={`
                  w-4 h-4 rounded-full border-2 mr-3
                  ${selectedPeriod === 'monthly' 
                    ? 'border-blue-500 bg-blue-500' 
                    : 'border-gray-300'
                  }
                `}>
                  {selectedPeriod === 'monthly' && (
                    <div className="w-2 h-2 bg-white rounded-full mx-auto mt-0.5"></div>
                  )}
                </div>
                <span className="font-semibold text-gray-900">Monthly</span>
              </div>
              <div className="text-2xl font-bold text-gray-900">
                ${monthlyPrice}
                <span className="text-sm font-normal text-gray-600">/month</span>
              </div>
              <div className="text-sm text-gray-500 mt-1">
                Billed monthly
              </div>
            </div>
          </div>

          {/* Annual Option */}
          <div 
            className={`
              relative rounded-lg border-2 cursor-pointer transition-all duration-200 p-4
              ${selectedPeriod === 'annual' 
                ? 'border-green-500 bg-green-50 shadow-md' 
                : 'border-gray-200 bg-white hover:border-gray-300'
              }
            `}
            onClick={() => onPeriodChange('annual')}
          >
            {/* Savings Badge */}
            <div className="absolute -top-2 right-2">
              <Badge className="bg-green-500 text-white text-xs px-2 py-1">
                Save ${totalSavings.toFixed(0)}/year
              </Badge>
            </div>
            
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <div className={`
                  w-4 h-4 rounded-full border-2 mr-3
                  ${selectedPeriod === 'annual' 
                    ? 'border-green-500 bg-green-500' 
                    : 'border-gray-300'
                  }
                `}>
                  {selectedPeriod === 'annual' && (
                    <div className="w-2 h-2 bg-white rounded-full mx-auto mt-0.5"></div>
                  )}
                </div>
                <span className="font-semibold text-gray-900">Annual</span>
              </div>
              <div className="text-2xl font-bold text-gray-900">
                ${calculatedAnnualPrice}
                <span className="text-sm font-normal text-gray-600">/month</span>
              </div>
              <div className="text-sm text-gray-500 mt-1">
                <span className="line-through text-gray-400">${monthlyPrice}/month</span>
                <span className="text-green-600 font-medium ml-2">20% off</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                Billed annually (${yearlyTotal}/year)
              </div>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="text-center text-sm text-gray-600">
            {selectedPeriod === 'monthly' ? (
              <>You'll be charged <strong>${monthlyPrice}</strong> every month</>
            ) : (
              <>You'll be charged <strong>${yearlyTotal}</strong> annually, saving <strong>${totalSavings}</strong> per year</>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};