import * as React from 'react';
import * as LabelPrimitive from '@radix-ui/react-label';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

const labelVariants = cva(
  'text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70'
);

/**
 * Label component renders an accessible form label element
 *
 * Semantic form label built with Radix UI primitives. Automatically associates with
 * form inputs via htmlFor attribute. Includes disabled state styling that inherits
 * from peer inputs.
 *
 * @component
 * @param {React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root>} props - Radix Label props
 * @param {string} [props.htmlFor] - ID of associated form input
 * @param {string} [props.className] - Additional CSS classes
 *
 * @example
 * <Label htmlFor="email">Email Address</Label>
 * <Input id="email" type="email" />
 *
 * @remarks
 * - Built on Radix UI Label primitive
 * - Disabled styling inherited from peer:disabled selector
 */
const Label = React.forwardRef<
  React.ElementRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root> & VariantProps<typeof labelVariants>
>(({ className, ...props }, ref) => (
  <LabelPrimitive.Root ref={ref} className={cn(labelVariants(), className)} {...props} />
));
Label.displayName = LabelPrimitive.Root.displayName;

export { Label };
