import * as React from 'react';

import { cn } from '@/lib/utils';

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  className?: string;
}

/**
 * Textarea component provides a styled multi-line text input field
 *
 * Wrapper around HTML textarea element with consistent styling, border, and focus states.
 * Supports all standard HTML textarea attributes with automatic responsive sizing.
 *
 * @component
 * @param {TextareaProps} props - Standard HTML textarea attributes
 * @param {string} [props.className] - Additional CSS classes
 * @param {string} [props.placeholder] - Placeholder text
 *
 * @example
 * <Textarea placeholder="Enter your message..." rows={5} />
 *
 * @remarks
 * - Minimum height: 80px
 * - Includes focus ring styling
 * - Disabled state styling included
 * - Supports custom className merging
 */
const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          'flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Textarea.displayName = 'Textarea';

export { Textarea };
