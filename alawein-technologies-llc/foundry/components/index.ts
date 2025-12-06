/**
 * Components Index
 * Export all components from a single entry point
 */

// Core Components
export { default as Button, ButtonGroup, type ButtonProps, type ButtonGroupProps } from './Button';
export { default as Input, type InputProps } from './Input';
export { default as Select, type SelectProps, type SelectOption } from './Select';
export { default as Card, CardHeader, CardBody, CardFooter, type CardProps } from './Card';
export { default as Modal, ConfirmModal, type ModalProps, type ConfirmModalProps } from './Modal';
export { default as Table, type TableProps, type TableColumn } from './Table';
export { default as Badge, type BadgeProps } from './Badge';
export { default as Spinner, Skeleton, type SpinnerProps, type SkeletonProps } from './Spinner';

// Toast System
export {
  ToastProvider,
  useToast,
  toast,
  type ToastProps,
  type ToastData,
  type ToastVariant,
  type ToastPosition,
  type ToastProviderProps,
} from './Toast';

// Additional components that would be included in the full library
// These would be created similarly to the components above

// Layout Components
// export { default as Container } from './Container';
// export { default as Grid } from './Grid';
// export { default as Flex } from './Flex';
// export { default as Spacer } from './Spacer';
// export { default as Divider } from './Divider';

// Navigation Components
// export { default as Navbar } from './Navbar';
// export { default as Sidebar } from './Sidebar';
// export { default as Breadcrumb } from './Breadcrumb';
// export { default as Tabs } from './Tabs';
// export { default as Pagination } from './Pagination';
// export { default as Link } from './Link';

// Form Components
// export { default as Textarea } from './Textarea';
// export { default as Checkbox } from './Checkbox';
// export { default as Radio } from './Radio';
// export { default as Switch } from './Switch';
// export { default as DatePicker } from './DatePicker';
// export { default as FileUpload } from './FileUpload';
// export { default as Form } from './Form';
// export { default as FormField } from './FormField';

// Data Display Components
// export { default as List } from './List';
// export { default as Avatar } from './Avatar';
// export { default as Tag } from './Tag';
// export { default as Stat } from './Stat';
// export { default as EmptyState } from './EmptyState';
// export { default as Timeline } from './Timeline';

// Feedback Components
// export { default as Alert } from './Alert';
// export { default as Progress } from './Progress';
// export { default as Drawer } from './Drawer';
// export { default as Popover } from './Popover';
// export { default as Tooltip } from './Tooltip';
// export { default as LoadingDots } from './LoadingDots';

// Advanced Components
// export { default as Chart } from './Chart';
// export { default as DataGrid } from './DataGrid';
// export { default as SearchAutocomplete } from './SearchAutocomplete';
// export { default as FilterPanel } from './FilterPanel';
// export { default as Stepper } from './Stepper';
// export { default as Accordion } from './Accordion';
// export { default as Carousel } from './Carousel';

// Utility exports
export { cn, conditionalClass, twMerge } from '../utils/cn';

// Theme and animation exports
export { theme, lightTheme, darkTheme, themeManager, useTheme, type ThemeConfig, type ThemeColors } from '../theme';
export { animations, duration, easing, spring } from '../animations';