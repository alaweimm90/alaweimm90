import type { VariantProps } from 'class-variance-authority';
import * as React from 'react';

import { cn } from '@/lib/utils';
import badgeVariants from './badgeVariants';

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

/**
 * Badge component renders a small label or status indicator with optional variants
 *
 * Flexible badge element with multiple color and style variants via CVA system.
 * Can display labels, tags, status indicators, or inventory states.
 *
 * @component
 * @param {BadgeProps} props - Component props
 * @param {VariantProps<typeof badgeVariants>} props.variant - Badge style variant
 * @param {string} [props.className] - Additional CSS classes
 *
 * @example
 * <Badge variant="outline">New</Badge>
 * <Badge variant="default">Premium</Badge>
 *
 * @remarks
 * - Variants defined in badgeVariants module
 * - Uses CVA for consistent styling
 * - Supports custom className merging
 */
function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge };
