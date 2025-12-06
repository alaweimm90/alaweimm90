// Component prop types - UI component interfaces
// These types ensure consistent component APIs across the application

import type { ReactNode, ComponentProps } from 'react';
import type { TierType, BillingCycle } from './business';

// Base component props that all components should extend
export interface BaseComponentProps {
  className?: string;
  children?: ReactNode;
  'data-testid'?: string;
}

// Tier-related component props
export interface TierCardProps extends BaseComponentProps {
  tier: TierType;
  variant?: 'default' | 'compact' | 'featured' | 'comparison';
  billingCycle?: BillingCycle;
  isSelected?: boolean;
  isPopular?: boolean;
  onSelect?: (tier: TierType) => void;
  onLearnMore?: (tier: TierType) => void;
  showFeatures?: boolean;
  showPricing?: boolean;
  disabled?: boolean;
}

export interface PricingToggleProps extends BaseComponentProps {
  value: BillingCycle;
  onChange: (cycle: BillingCycle) => void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export interface FeatureListProps extends BaseComponentProps {
  features: string[] | Record<string, boolean>;
  tier?: TierType;
  variant?: 'bullets' | 'checkmarks' | 'grid' | 'comparison';
  showTierColors?: boolean;
  compact?: boolean;
}

// Modal and dialog props
export interface ModalProps extends BaseComponentProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  description?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  showCloseButton?: boolean;
}

export interface ConfirmationModalProps extends ModalProps {
  onConfirm: () => void;
  onCancel?: () => void;
  confirmText?: string;
  cancelText?: string;
  variant?: 'default' | 'destructive' | 'warning';
}

export interface SubscriptionModalProps extends ModalProps {
  selectedTier?: TierType;
  onTierChange?: (tier: TierType) => void;
  onSubscriptionSuccess?: (subscriptionId: string) => void;
}

// Form component props
export interface FormProps extends BaseComponentProps {
  onSubmit: (data: Record<string, unknown>) => void | Promise<void>;
  isLoading?: boolean;
  disabled?: boolean;
  variant?: 'default' | 'card' | 'inline';
}

export interface FormFieldProps extends BaseComponentProps {
  label?: string;
  description?: string;
  error?: string;
  required?: boolean;
  disabled?: boolean;
}

export interface InputProps extends FormFieldProps, 
  Omit<ComponentProps<'input'>, 'className' | 'children' | 'size'> {
  variant?: 'default' | 'ghost' | 'filled';
  inputSize?: 'sm' | 'md' | 'lg';
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

export interface ButtonProps extends BaseComponentProps,
  Omit<ComponentProps<'button'>, 'className' | 'children'> {
  variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  isLoading?: boolean;
  loadingText?: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  fullWidth?: boolean;
}

// Navigation component props
export interface NavigationProps extends BaseComponentProps {
  variant?: 'horizontal' | 'vertical' | 'mobile';
  showLogo?: boolean;
  showUserMenu?: boolean;
  sticky?: boolean;
}

export interface NavigationItemProps extends BaseComponentProps {
  href?: string;
  isActive?: boolean;
  isDisabled?: boolean;
  icon?: ReactNode;
  badge?: string | number;
  onClick?: () => void;
}

// Layout component props
export interface LayoutProps extends BaseComponentProps {
  header?: ReactNode;
  sidebar?: ReactNode;
  footer?: ReactNode;
  variant?: 'default' | 'centered' | 'full-width';
}

export interface ContainerProps extends BaseComponentProps {
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  padding?: boolean;
  centered?: boolean;
}

// Card component props
export interface CardProps extends BaseComponentProps {
  variant?: 'default' | 'elevated' | 'outlined' | 'glass' | 'tier';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  clickable?: boolean;
  onClick?: () => void;
  tier?: TierType;
}

// Data display component props
export interface TableProps extends BaseComponentProps {
  data: Array<Record<string, unknown>>;
  columns: TableColumn[];
  isLoading?: boolean;
  emptyMessage?: string;
  onRowClick?: (row: Record<string, unknown>, index: number) => void;
  sortable?: boolean;
  pagination?: boolean;
}

export interface TableColumn {
  key: string;
  title: string;
  render?: (value: unknown, row: Record<string, unknown>, index: number) => ReactNode;
  sortable?: boolean;
  width?: string | number;
  align?: 'left' | 'center' | 'right';
}

// Loading and state component props
export interface LoadingProps extends BaseComponentProps {
  variant?: 'spinner' | 'skeleton' | 'pulse' | 'dots';
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  overlay?: boolean;
}

export interface SkeletonProps extends BaseComponentProps {
  width?: string | number;
  height?: string | number;
  circle?: boolean;
  count?: number;
}

// Toast and notification props
export interface ToastProps extends BaseComponentProps {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  title?: string;
  description?: string;
  action?: ReactNode;
  duration?: number;
  onClose?: () => void;
}

// Animation and transition props
export interface AnimatedSectionProps extends BaseComponentProps {
  animation?: 'fade' | 'slide' | 'scale' | 'bounce';
  direction?: 'up' | 'down' | 'left' | 'right';
  delay?: number;
  duration?: number;
  triggerOnce?: boolean;
}

// Search and filter props
export interface SearchProps extends BaseComponentProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  variant?: 'default' | 'contained' | 'borderless';
  clearable?: boolean;
  onClear?: () => void;
}

export interface FilterProps extends BaseComponentProps {
  filters: FilterOption[];
  selectedFilters: string[];
  onChange: (filters: string[]) => void;
  variant?: 'dropdown' | 'chips' | 'sidebar';
  multiple?: boolean;
}

export interface FilterOption {
  id: string;
  label: string;
  count?: number;
  disabled?: boolean;
}