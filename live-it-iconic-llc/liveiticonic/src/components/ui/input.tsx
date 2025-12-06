import * as React from 'react';

import { cn } from '@/lib/utils';

/**
 * Input component provides a styled text input field
 *
 * Wrapper around HTML input element with consistent styling, border, and placeholder text.
 * Supports all standard HTML input types and attributes with automatic responsive sizing.
 *
 * @component
 * @param {React.ComponentProps<'input'>} props - Standard HTML input attributes
 * @param {string} [props.type] - Input type (text, email, password, etc.)
 * @param {string} [props.className] - Additional CSS classes
 * @param {string} [props.placeholder] - Placeholder text
 *
 * @example
 * <Input type="email" placeholder="Enter your email" />
 *
 * @remarks
 * - Includes focus ring, border, and padding styling
 * - Disabled state styling included
 * - File input styling supported
 */
const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<'input'>>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          'flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-base ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export { Input };
