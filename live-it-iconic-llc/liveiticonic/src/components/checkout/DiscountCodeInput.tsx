/**
 * Discount Code Input Component
 *
 * Allows customers to enter and apply discount codes during checkout.
 * Validates codes and displays discount amount.
 */

import React, { useState } from 'react';
import { Tag, Check, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { discountService } from '@/services/discountService';

interface DiscountCodeInputProps {
  orderAmount: number; // Amount in cents
  onDiscountApplied: (discountCode: string, discountAmount: number) => void;
  onDiscountRemoved: () => void;
  appliedCode?: string;
  disabled?: boolean;
}

export const DiscountCodeInput: React.FC<DiscountCodeInputProps> = ({
  orderAmount,
  onDiscountApplied,
  onDiscountRemoved,
  appliedCode,
  disabled = false,
}) => {
  const [code, setCode] = useState('');
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleApply = async () => {
    if (!code.trim()) {
      setError('Please enter a discount code');
      return;
    }

    setValidating(true);
    setError(null);
    setSuccess(null);

    try {
      const validation = await discountService.validateDiscount(code, orderAmount);

      if (validation.isValid && validation.discount) {
        const discountAmount = validation.discount.discountAmount;
        const formattedDiscount = (discountAmount / 100).toFixed(2);

        setSuccess(`Discount applied: -$${formattedDiscount}`);
        onDiscountApplied(code, discountAmount);
        setCode('');
      } else {
        setError(validation.error || 'Invalid discount code');
      }
    } catch (err) {
      setError('Failed to validate discount code. Please try again.');
    } finally {
      setValidating(false);
    }
  };

  const handleRemove = () => {
    setError(null);
    setSuccess(null);
    onDiscountRemoved();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleApply();
    }
  };

  if (appliedCode) {
    return (
      <div className="border border-green-200 bg-green-50 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-green-100 flex items-center justify-center">
              <Check className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <div className="font-semibold text-green-900">
                Discount Applied
              </div>
              <div className="text-sm text-green-700">
                Code: {appliedCode.toUpperCase()}
              </div>
            </div>
          </div>
          <Button
            onClick={handleRemove}
            variant="ghost"
            size="sm"
            className="text-green-700 hover:text-green-900"
            disabled={disabled}
          >
            Remove
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Tag className="h-4 w-4 text-gray-500" />
        <span className="text-sm font-medium text-gray-700">
          Have a discount code?
        </span>
      </div>

      <div className="flex gap-2">
        <div className="flex-1">
          <Input
            value={code}
            onChange={(e) => {
              setCode(e.target.value.toUpperCase());
              setError(null);
              setSuccess(null);
            }}
            onKeyPress={handleKeyPress}
            placeholder="Enter code"
            disabled={disabled || validating}
            className={`uppercase ${error ? 'border-red-300' : ''} ${
              success ? 'border-green-300' : ''
            }`}
            maxLength={50}
          />
        </div>
        <Button
          onClick={handleApply}
          disabled={disabled || validating || !code.trim()}
          variant="outline"
          className="min-w-[80px]"
        >
          {validating ? 'Validating...' : 'Apply'}
        </Button>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-600">
          <X className="h-4 w-4" />
          {error}
        </div>
      )}

      {success && (
        <div className="flex items-center gap-2 text-sm text-green-600">
          <Check className="h-4 w-4" />
          {success}
        </div>
      )}

      <div className="text-xs text-gray-500">
        Discount codes are applied at checkout and cannot be combined.
      </div>
    </div>
  );
};
