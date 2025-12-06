/**
 * Table Component
 * A comprehensive data table with sorting, filtering, and pagination
 */

import React, { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronUp,
  ChevronDown,
  ChevronsUpDown,
  Search,
  Filter,
  Download,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from 'lucide-react';
import { cn } from '../utils/cn';
import Input from './Input';
import Button from './Button';

export interface TableColumn<T = any> {
  /**
   * Column key (must match data property)
   */
  key: string;

  /**
   * Column header label
   */
  label: string;

  /**
   * Column width
   */
  width?: string | number;

  /**
   * Sortable column
   */
  sortable?: boolean;

  /**
   * Filterable column
   */
  filterable?: boolean;

  /**
   * Custom render function
   */
  render?: (value: any, row: T, index: number) => React.ReactNode;

  /**
   * Column alignment
   */
  align?: 'left' | 'center' | 'right';

  /**
   * Sticky column
   */
  sticky?: boolean;

  /**
   * Custom sort function
   */
  sortFn?: (a: T, b: T) => number;

  /**
   * Custom filter function
   */
  filterFn?: (value: any, filterValue: string) => boolean;
}

export interface TableProps<T = any> {
  /**
   * Table data
   */
  data: T[];

  /**
   * Table columns configuration
   */
  columns: TableColumn<T>[];

  /**
   * Table caption
   */
  caption?: string;

  /**
   * Striped rows
   */
  striped?: boolean;

  /**
   * Hover effect on rows
   */
  hoverable?: boolean;

  /**
   * Bordered table
   */
  bordered?: boolean;

  /**
   * Compact mode
   */
  compact?: boolean;

  /**
   * Loading state
   */
  loading?: boolean;

  /**
   * Empty state message
   */
  emptyMessage?: string;

  /**
   * Enable sorting
   */
  sortable?: boolean;

  /**
   * Enable filtering
   */
  filterable?: boolean;

  /**
   * Enable pagination
   */
  paginated?: boolean;

  /**
   * Page size for pagination
   */
  pageSize?: number;

  /**
   * Page size options
   */
  pageSizeOptions?: number[];

  /**
   * Enable row selection
   */
  selectable?: boolean;

  /**
   * Selected rows
   */
  selectedRows?: Set<number>;

  /**
   * On row selection change
   */
  onSelectionChange?: (selected: Set<number>) => void;

  /**
   * On row click
   */
  onRowClick?: (row: T, index: number) => void;

  /**
   * Sticky header
   */
  stickyHeader?: boolean;

  /**
   * Max height (for scrollable table)
   */
  maxHeight?: string | number;

  /**
   * Enable export
   */
  exportable?: boolean;

  /**
   * Export filename
   */
  exportFilename?: string;

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Additional wrapper class names
   */
  wrapperClassName?: string;
}

function Table<T extends Record<string, any>>({
  data,
  columns,
  caption,
  striped = false,
  hoverable = true,
  bordered = false,
  compact = false,
  loading = false,
  emptyMessage = 'No data available',
  sortable = true,
  filterable = false,
  paginated = false,
  pageSize: initialPageSize = 10,
  pageSizeOptions = [10, 25, 50, 100],
  selectable = false,
  selectedRows = new Set<number>(),
  onSelectionChange,
  onRowClick,
  stickyHeader = false,
  maxHeight,
  exportable = false,
  exportFilename = 'table-data',
  className,
  wrapperClassName,
}: TableProps<T>) {
  // State
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [filterValue, setFilterValue] = useState('');
  const [filterColumn, setFilterColumn] = useState<string>('');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialPageSize);
  const [localSelectedRows, setLocalSelectedRows] = useState(selectedRows);

  // Computed values
  const effectiveSelectedRows = onSelectionChange ? selectedRows : localSelectedRows;

  // Filter data
  const filteredData = useMemo(() => {
    if (!filterable || !filterValue || !filterColumn) return data;

    const column = columns.find(col => col.key === filterColumn);
    if (!column) return data;

    return data.filter(row => {
      const value = row[column.key];
      if (column.filterFn) {
        return column.filterFn(value, filterValue);
      }
      return String(value).toLowerCase().includes(filterValue.toLowerCase());
    });
  }, [data, filterValue, filterColumn, columns, filterable]);

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortable || !sortColumn) return filteredData;

    const column = columns.find(col => col.key === sortColumn);
    if (!column) return filteredData;

    const sorted = [...filteredData].sort((a, b) => {
      if (column.sortFn) {
        return column.sortFn(a, b);
      }

      const aValue = a[sortColumn];
      const bValue = b[sortColumn];

      if (aValue === bValue) return 0;
      if (aValue == null) return 1;
      if (bValue == null) return -1;

      const comparison = aValue < bValue ? -1 : 1;
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return sorted;
  }, [filteredData, sortColumn, sortDirection, columns, sortable]);

  // Paginate data
  const paginatedData = useMemo(() => {
    if (!paginated) return sortedData;

    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return sortedData.slice(startIndex, endIndex);
  }, [sortedData, currentPage, pageSize, paginated]);

  // Total pages
  const totalPages = Math.ceil(sortedData.length / pageSize);

  // Handlers
  const handleSort = useCallback((column: string) => {
    if (!sortable) return;

    if (sortColumn === column) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  }, [sortColumn, sortable]);

  const handleSelectAll = useCallback(() => {
    if (!selectable) return;

    const newSelection = new Set<number>();
    if (effectiveSelectedRows.size < paginatedData.length) {
      paginatedData.forEach((_, index) => {
        const globalIndex = paginated ? (currentPage - 1) * pageSize + index : index;
        newSelection.add(globalIndex);
      });
    }

    if (onSelectionChange) {
      onSelectionChange(newSelection);
    } else {
      setLocalSelectedRows(newSelection);
    }
  }, [selectable, effectiveSelectedRows, paginatedData, paginated, currentPage, pageSize, onSelectionChange]);

  const handleSelectRow = useCallback((index: number) => {
    if (!selectable) return;

    const globalIndex = paginated ? (currentPage - 1) * pageSize + index : index;
    const newSelection = new Set(effectiveSelectedRows);

    if (newSelection.has(globalIndex)) {
      newSelection.delete(globalIndex);
    } else {
      newSelection.add(globalIndex);
    }

    if (onSelectionChange) {
      onSelectionChange(newSelection);
    } else {
      setLocalSelectedRows(newSelection);
    }
  }, [selectable, paginated, currentPage, pageSize, effectiveSelectedRows, onSelectionChange]);

  const handleExport = useCallback(() => {
    const csvContent = [
      // Headers
      columns.map(col => col.label).join(','),
      // Data rows
      ...sortedData.map(row =>
        columns.map(col => {
          const value = row[col.key];
          // Escape values containing commas or quotes
          if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
            return `"${value.replace(/"/g, '""')}"`;
          }
          return value;
        }).join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${exportFilename}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }, [sortedData, columns, exportFilename]);

  // Render sort icon
  const renderSortIcon = (column: string) => {
    if (sortColumn !== column) {
      return <ChevronsUpDown className="w-4 h-4 text-gray-400" />;
    }
    return sortDirection === 'asc' ? (
      <ChevronUp className="w-4 h-4 text-primary-500" />
    ) : (
      <ChevronDown className="w-4 h-4 text-primary-500" />
    );
  };

  // Table wrapper classes
  const wrapperClasses = cn(
    'w-full',
    maxHeight && 'overflow-auto',
    wrapperClassName
  );

  // Table classes
  const tableClasses = cn(
    'min-w-full divide-y divide-gray-200 dark:divide-gray-700',
    bordered && 'border border-gray-200 dark:border-gray-700',
    className
  );

  return (
    <div className={wrapperClasses} style={{ maxHeight }}>
      {/* Toolbar */}
      {(filterable || exportable) && (
        <div className="mb-4 flex items-center justify-between gap-4">
          {/* Filter */}
          {filterable && (
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search..."
                  value={filterValue}
                  onChange={(e) => setFilterValue(e.target.value)}
                  className="pl-9 pr-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              {columns.some(col => col.filterable) && (
                <select
                  value={filterColumn}
                  onChange={(e) => setFilterColumn(e.target.value)}
                  className="px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="">All columns</option>
                  {columns.filter(col => col.filterable).map(col => (
                    <option key={col.key} value={col.key}>
                      {col.label}
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}

          {/* Export */}
          {exportable && (
            <Button
              variant="outline"
              size="sm"
              leftIcon={<Download className="w-4 h-4" />}
              onClick={handleExport}
            >
              Export CSV
            </Button>
          )}
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className={tableClasses}>
          {/* Caption */}
          {caption && (
            <caption className="py-2 text-sm text-gray-500 dark:text-gray-400">
              {caption}
            </caption>
          )}

          {/* Header */}
          <thead className={cn(
            'bg-gray-50 dark:bg-gray-900',
            stickyHeader && 'sticky top-0 z-10'
          )}>
            <tr>
              {/* Selection checkbox */}
              {selectable && (
                <th className="px-4 py-3">
                  <input
                    type="checkbox"
                    checked={effectiveSelectedRows.size === paginatedData.length && paginatedData.length > 0}
                    indeterminate={effectiveSelectedRows.size > 0 && effectiveSelectedRows.size < paginatedData.length}
                    onChange={handleSelectAll}
                    className="rounded border-gray-300 dark:border-gray-600"
                  />
                </th>
              )}

              {/* Column headers */}
              {columns.map((column) => (
                <th
                  key={column.key}
                  className={cn(
                    'px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider',
                    column.align === 'center' && 'text-center',
                    column.align === 'right' && 'text-right',
                    column.sortable && sortable && 'cursor-pointer select-none hover:text-gray-700 dark:hover:text-gray-200',
                    column.sticky && 'sticky left-0 z-10 bg-gray-50 dark:bg-gray-900',
                    compact && 'px-4 py-2'
                  )}
                  style={{ width: column.width }}
                  onClick={() => column.sortable && handleSort(column.key)}
                >
                  <div className="flex items-center gap-1">
                    <span>{column.label}</span>
                    {column.sortable && sortable && renderSortIcon(column.key)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>

          {/* Body */}
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {loading ? (
              // Loading state
              <tr>
                <td
                  colSpan={columns.length + (selectable ? 1 : 0)}
                  className="px-6 py-12 text-center"
                >
                  <div className="flex flex-col items-center gap-2">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full"
                    />
                    <span className="text-sm text-gray-500 dark:text-gray-400">Loading...</span>
                  </div>
                </td>
              </tr>
            ) : paginatedData.length === 0 ? (
              // Empty state
              <tr>
                <td
                  colSpan={columns.length + (selectable ? 1 : 0)}
                  className="px-6 py-12 text-center text-gray-500 dark:text-gray-400"
                >
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              // Data rows
              <AnimatePresence mode="wait">
                {paginatedData.map((row, rowIndex) => {
                  const globalIndex = paginated ? (currentPage - 1) * pageSize + rowIndex : rowIndex;
                  const isSelected = effectiveSelectedRows.has(globalIndex);

                  return (
                    <motion.tr
                      key={rowIndex}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className={cn(
                        striped && rowIndex % 2 === 1 && 'bg-gray-50 dark:bg-gray-900/50',
                        hoverable && 'hover:bg-gray-50 dark:hover:bg-gray-900/50',
                        isSelected && 'bg-primary-50 dark:bg-primary-950/30',
                        onRowClick && 'cursor-pointer'
                      )}
                      onClick={() => onRowClick?.(row, rowIndex)}
                    >
                      {/* Selection checkbox */}
                      {selectable && (
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => handleSelectRow(rowIndex)}
                            onClick={(e) => e.stopPropagation()}
                            className="rounded border-gray-300 dark:border-gray-600"
                          />
                        </td>
                      )}

                      {/* Data cells */}
                      {columns.map((column) => (
                        <td
                          key={column.key}
                          className={cn(
                            'px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100',
                            column.align === 'center' && 'text-center',
                            column.align === 'right' && 'text-right',
                            column.sticky && 'sticky left-0 z-10 bg-white dark:bg-gray-800',
                            compact && 'px-4 py-2'
                          )}
                        >
                          {column.render
                            ? column.render(row[column.key], row, rowIndex)
                            : row[column.key]}
                        </td>
                      ))}
                    </motion.tr>
                  );
                })}
              </AnimatePresence>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {paginated && !loading && sortedData.length > 0 && (
        <div className="mt-4 flex items-center justify-between">
          {/* Page size selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Show</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setCurrentPage(1);
              }}
              className="px-2 py-1 border border-gray-300 dark:border-gray-700 rounded-md text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
            >
              {pageSizeOptions.map(size => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
            <span className="text-sm text-gray-600 dark:text-gray-400">entries</span>
          </div>

          {/* Page info */}
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Showing {(currentPage - 1) * pageSize + 1} to{' '}
            {Math.min(currentPage * pageSize, sortedData.length)} of {sortedData.length} entries
          </div>

          {/* Page controls */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
              className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="First page"
            >
              <ChevronsLeft className="w-5 h-5" />
            </button>
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Previous page"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>

            {/* Page numbers */}
            <div className="flex items-center gap-1 px-2">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNumber: number;
                if (totalPages <= 5) {
                  pageNumber = i + 1;
                } else if (currentPage <= 3) {
                  pageNumber = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNumber = totalPages - 4 + i;
                } else {
                  pageNumber = currentPage - 2 + i;
                }

                return (
                  <button
                    key={i}
                    onClick={() => setCurrentPage(pageNumber)}
                    className={cn(
                      'w-8 h-8 rounded text-sm',
                      pageNumber === currentPage
                        ? 'bg-primary-500 text-white'
                        : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                    )}
                  >
                    {pageNumber}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
              className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Next page"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
            <button
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
              className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Last page"
            >
              <ChevronsRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

Table.displayName = 'Table';

export default Table;