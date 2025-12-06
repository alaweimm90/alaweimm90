import { Slot } from '@radix-ui/react-slot';
import type { VariantProps } from 'class-variance-authority';
import * as React from 'react';

import { cn } from '@/lib/utils';
import buttonVariants from './buttonVariants';

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

/**
 * Button component provides a flexible, polymorphic button element with multiple style variants
 *
 * Renders a customizable button element that can be styled via variant system (primary, secondary, outline, ghost).
 * Supports size variants (sm, md, lg) and can be used as any element via asChild prop with Radix UI Slot.
 * Fully accessible with proper HTML button semantics.
 *
 * @component
 * @param {ButtonProps} props - Component props extending standard HTML button attributes
 * @param {VariantProps<typeof buttonVariants>} props.variant - Style variant (primary, secondary, outline, ghost, etc.)
 * @param {VariantProps<typeof buttonVariants>} props.size - Size variant (sm, md, lg)
 * @param {boolean} [props.asChild] - Render as child element (via Radix Slot)
 * @param {string} [props.className] - Additional CSS classes
 *
 * @example
 * <Button variant="primary" size="lg" onClick={() => handleClick()}>
 *   Click me
 * </Button>
 *
 * @remarks
 * - Variants defined in buttonVariants module
 * - Uses CVA (class-variance-authority) for variant management
 * - Supports forwarded refs for DOM access
 */
const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
    );
  }
);
Button.displayName = 'Button';

export { Button };
