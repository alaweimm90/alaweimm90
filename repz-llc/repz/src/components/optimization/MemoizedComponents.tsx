import React, { memo, useMemo, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';

// Memoized card component for better performance
interface OptimizedCardProps {
  title: string;
  content: string;
  badges?: string[];
  onAction?: () => void;
  actionLabel?: string;
  variant?: 'default' | 'outline';
}

const OptimizedCard = memo<OptimizedCardProps>(({
  title,
  content,
  badges = [],
  onAction,
  actionLabel = 'Action',
  variant = 'default'
}) => {
  // Memoize badge rendering
  const renderedBadges = useMemo(() => (
    badges.map((badge, index) => (
      <Badge key={`${badge}-${index}`} variant="secondary" className="mr-1">
        {badge}
      </Badge>
    ))
  ), [badges]);

  // Memoize action handler
  const handleAction = useCallback(() => {
    onAction?.();
  }, [onAction]);

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          {title}
          {renderedBadges.length > 0 && (
            <div className="flex flex-wrap">
              {renderedBadges}
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground mb-4">{content}</p>
        {onAction && (
          <Button onClick={handleAction} variant="outline" size="sm">
            {actionLabel}
          </Button>
        )}
      </CardContent>
    </Card>
  );
});

OptimizedCard.displayName = 'OptimizedCard';

// Memoized list component with virtual scrolling support
interface OptimizedListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T, index: number) => string;
  loading?: boolean;
  empty?: React.ReactNode;
  className?: string;
}

const OptimizedList = memo(<T,>({
  items,
  renderItem,
  keyExtractor,
  loading = false,
  empty = <div className="text-center text-muted-foreground">No items found</div>,
  className = ''
}: OptimizedListProps<T>) => {
  // Memoize rendered items
  const renderedItems = useMemo(() => (
    items.map((item, index) => (
      <div key={keyExtractor(item, index)}>
        {renderItem(item, index)}
      </div>
    ))
  ), [items, renderItem, keyExtractor]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
        <span className="ml-2 text-muted-foreground">Loading...</span>
      </div>
    );
  }

  if (items.length === 0) {
    return <div className={className}>{empty}</div>;
  }

  return (
    <div className={`space-y-2 ${className}`} role="list">
      {renderedItems}
    </div>
  );
}) as <T>(props: OptimizedListProps<T>) => JSX.Element;

// Memoized form field component
interface OptimizedFormFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: 'text' | 'email' | 'password';
  placeholder?: string;
  error?: string;
  required?: boolean;
}

const OptimizedFormField = memo<OptimizedFormFieldProps>(({
  label,
  value,
  onChange,
  type = 'text',
  placeholder,
  error,
  required = false
}) => {
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
  }, [onChange]);

  const fieldId = useMemo(() => `field-${label.toLowerCase().replace(/\s+/g, '-')}`, [label]);

  return (
    <div className="space-y-2">
      <label htmlFor={fieldId} className="text-sm font-medium">
        {label}
        {required && <span className="text-destructive ml-1">*</span>}
      </label>
      <input
        id={fieldId}
        type={type}
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        required={required}
        className={`w-full px-3 py-2 border rounded-md ${
          error ? 'border-destructive' : 'border-input'
        } bg-background`}
        aria-invalid={!!error}
        aria-describedby={error ? `${fieldId}-error` : undefined}
      />
      {error && (
        <p id={`${fieldId}-error`} className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
    </div>
  );
});

OptimizedFormField.displayName = 'OptimizedFormField';

// Memoized table component for large datasets
interface OptimizedTableProps<T> {
  data: T[];
  columns: Array<{
    key: keyof T;
    label: string;
    render?: (value: T[keyof T], item: T) => React.ReactNode;
  }>;
  keyExtractor: (item: T, index: number) => string;
  sortable?: boolean;
  onSort?: (key: keyof T, direction: 'asc' | 'desc') => void;
}

const OptimizedTable = memo(<T extends Record<string, unknown>>({
  data,
  columns,
  keyExtractor,
  sortable = false,
  onSort
}: OptimizedTableProps<T>) => {
  const [sortKey, setSortKey] = React.useState<keyof T | null>(null);
  const [sortDirection, setSortDirection] = React.useState<'asc' | 'desc'>('asc');

  const handleSort = useCallback((key: keyof T) => {
    if (!sortable) return;

    const newDirection = sortKey === key && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortKey(key);
    setSortDirection(newDirection);
    onSort?.(key, newDirection);
  }, [sortable, sortKey, sortDirection, onSort]);

  const renderedHeaders = useMemo(() => (
    columns.map((column) => (
      <th
        key={String(column.key)}
        className={`px-4 py-2 text-left font-medium ${
          sortable ? 'cursor-pointer hover:bg-muted' : ''
        }`}
        onClick={() => handleSort(column.key)}
      >
        {column.label}
        {sortable && sortKey === column.key && (
          <span className="ml-1">
            {sortDirection === 'asc' ? '↑' : '↓'}
          </span>
        )}
      </th>
    ))
  ), [columns, sortable, sortKey, sortDirection, handleSort]);

  const renderedRows = useMemo(() => (
    data.map((item, index) => (
      <tr key={keyExtractor(item, index)} className="border-b hover:bg-muted/50">
        {columns.map((column) => (
          <td key={String(column.key)} className="px-4 py-2">
            {column.render ? 
              column.render(item[column.key], item) : 
              String(item[column.key])
            }
          </td>
        ))}
      </tr>
    ))
  ), [data, columns, keyExtractor]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead className="bg-muted">
          <tr>{renderedHeaders}</tr>
        </thead>
        <tbody>{renderedRows}</tbody>
      </table>
    </div>
  );
}) as <T extends Record<string, unknown>>(props: OptimizedTableProps<T>) => JSX.Element;

export {
  OptimizedCard,
  OptimizedList,
  OptimizedFormField,
  OptimizedTable
};