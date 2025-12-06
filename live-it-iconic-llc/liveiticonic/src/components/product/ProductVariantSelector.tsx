import React from 'react';
import { ProductVariant } from '@/types/product';
import { Label } from '@/components/ui/label';

interface ProductVariantSelectorProps {
  variants: ProductVariant[];
  selectedVariantId?: string;
  onVariantSelect: (variantId: string) => void;
}

/**
 * ProductVariantSelector component renders grouped variant options (size, color, etc.)
 *
 * Automatically groups product variants by attribute type and displays selectable buttons
 * for each option. Shows variant prices as surcharges if applicable. Supports single or
 * multi-attribute variants with visual feedback for selected variant.
 *
 * @component
 * @param {ProductVariantSelectorProps} props - Component props
 * @param {ProductVariant[]} props.variants - Array of product variants with their attributes and prices
 * @param {string} [props.selectedVariantId] - ID of currently selected variant for visual indication
 * @param {Function} props.onVariantSelect - Callback fired when variant option is clicked with variantId
 *
 * @example
 * <ProductVariantSelector
 *   variants={[
 *     { id: 'v1', values: { size: 'M', color: 'Black' }, price: 39.99 },
 *     { id: 'v2', values: { size: 'L', color: 'Black' }, price: 39.99 }
 *   ]}
 *   selectedVariantId="v1"
 *   onVariantSelect={(id) => handleVariantChange(id)}
 * />
 *
 * @remarks
 * - Returns null if no variants provided
 * - Deduplicates variant options by attribute value
 * - Attribute names are capitalized in the UI
 */
export const ProductVariantSelector: React.FC<ProductVariantSelectorProps> = ({
  variants,
  selectedVariantId,
  onVariantSelect,
}) => {
  if (!variants || variants.length === 0) {
    return null;
  }

  // Group variants by attribute (size, color, etc.)
  const groupedVariants = variants.reduce(
    (acc, variant) => {
      Object.entries(variant.values).forEach(([key, value]) => {
        if (!acc[key]) {
          acc[key] = [];
        }
        if (!acc[key].find(v => v.value === value && v.variantId === variant.id)) {
          acc[key].push({ value: value || '', variantId: variant.id, price: variant.price || 0 });
        }
      });
      return acc;
    },
    {} as Record<string, Array<{ value: string; variantId: string; price: number }>>
  );

  return (
    <div className="space-y-4">
      {Object.entries(groupedVariants).map(([attribute, options]) => (
        <div key={attribute}>
          <Label className="text-lii-cloud font-ui font-medium mb-2 block capitalize">
            {attribute}
          </Label>
          <div className="flex flex-wrap gap-2">
            {options.map((option, index) => {
              const variant = variants.find(v => v.id === option.variantId);
              const isSelected = selectedVariantId === option.variantId;

              return (
                <button
                  key={`${option.variantId}-${index}`}
                  onClick={() => onVariantSelect(option.variantId)}
                  className={`px-4 py-2 rounded-lg border transition-all ${
                    isSelected
                      ? 'border-lii-gold bg-lii-gold/10 text-lii-gold'
                      : 'border-lii-gold/20 text-lii-cloud hover:border-lii-gold/40'
                  }`}
                >
                  {option.value}
                  {variant && variant.price !== undefined && (
                    <span className="ml-2 text-xs text-lii-ash">
                      (+${(variant.price - variant.price * 0.9).toFixed(2)})
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ProductVariantSelector;
