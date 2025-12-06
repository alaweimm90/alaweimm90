import * as React from 'react';

import { cn } from '@/lib/utils';

/**
 * Card component provides a container with rounded corners, border, background, and shadow
 *
 * Base wrapper component for grouped content with a styled card appearance. Accepts
 * standard HTML div attributes and custom className merging.
 *
 * @component
 * @param {React.HTMLAttributes<HTMLDivElement>} props - Standard HTML div attributes
 * @param {string} [props.className] - Additional CSS classes
 *
 * @example
 * <Card>
 *   <CardHeader>
 *     <CardTitle>Card Title</CardTitle>
 *   </CardHeader>
 *   <CardContent>Content here</CardContent>
 * </Card>
 */
const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'rounded-2xl border border-lii-graphite/20 bg-lii-ink text-lii-cloud shadow-[0_8px_24px_rgba(0,0,0,0.12)]',
        className
      )}
      {...props}
    />
  )
);
Card.displayName = 'Card';

/**
 * CardHeader component provides spacing and layout for card header content
 *
 * Container for card header with flex column layout and spacing between items.
 *
 * @component
 * @param {React.HTMLAttributes<HTMLDivElement>} props - Standard HTML div attributes
 */
const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('flex flex-col space-y-1.5 p-6', className)} {...props} />
  )
);
CardHeader.displayName = 'CardHeader';

/**
 * CardTitle component renders a semantic h3 heading for card titles
 *
 * Large, bold heading element typically used as the main title within a CardHeader.
 *
 * @component
 * @param {React.HTMLAttributes<HTMLHeadingElement>} props - Standard HTML heading attributes
 */
const CardTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn('text-2xl font-semibold leading-none tracking-tight', className)}
      {...props}
    />
  )
);
CardTitle.displayName = 'CardTitle';

/**
 * CardDescription component renders a descriptive paragraph for cards
 *
 * Small, muted text element for secondary content or descriptions within a card.
 *
 * @component
 * @param {React.HTMLAttributes<HTMLParagraphElement>} props - Standard HTML paragraph attributes
 */
const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
));
CardDescription.displayName = 'CardDescription';

/**
 * CardContent component provides padding and layout for card main content
 *
 * Primary content container with top padding offset for spacing below header.
 *
 * @component
 * @param {React.HTMLAttributes<HTMLDivElement>} props - Standard HTML div attributes
 */
const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('p-6 pt-0', className)} {...props} />
  )
);
CardContent.displayName = 'CardContent';

/**
 * CardFooter component provides spacing and layout for card footer content
 *
 * Container for card footer with flexbox layout for action buttons or links.
 *
 * @component
 * @param {React.HTMLAttributes<HTMLDivElement>} props - Standard HTML div attributes
 */
const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('flex items-center p-6 pt-0', className)} {...props} />
  )
);
CardFooter.displayName = 'CardFooter';

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent };
