/**
 * Select Component
 * A dropdown select component with search, multi-select, and custom rendering
 */

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, X, Check, Search } from 'lucide-react';
import { cn } from '../utils/cn';

export interface SelectOption {
  value: string | number;
  label: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  group?: string;
}

export interface SelectProps {
  /**
   * Options to display
   */
  options: SelectOption[];

  /**
   * Selected value(s)
   */
  value?: string | number | (string | number)[];

  /**
   * Default value
   */
  defaultValue?: string | number | (string | number)[];

  /**
   * On change handler
   */
  onChange?: (value: string | number | (string | number)[]) => void;

  /**
   * Placeholder text
   */
  placeholder?: string;

  /**
   * Multiple selection mode
   */
  multiple?: boolean;

  /**
   * Searchable select
   */
  searchable?: boolean;

  /**
   * Clearable select
   */
  clearable?: boolean;

  /**
   * Disabled state
   */
  disabled?: boolean;

  /**
   * Loading state
   */
  loading?: boolean;

  /**
   * Error state
   */
  error?: boolean | string;

  /**
   * Size variant
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';

  /**
   * Select variant
   */
  variant?: 'outline' | 'filled' | 'flushed';

  /**
   * Label
   */
  label?: string;

  /**
   * Helper text
   */
  helperText?: string;

  /**
   * Required field
   */
  required?: boolean;

  /**
   * Max height for dropdown
   */
  maxHeight?: number;

  /**
   * Custom render for option
   */
  renderOption?: (option: SelectOption, isSelected: boolean) => React.ReactNode;

  /**
   * Custom render for value
   */
  renderValue?: (value: string | number | (string | number)[], options: SelectOption[]) => React.ReactNode;

  /**
   * Full width
   */
  fullWidth?: boolean;

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Dropdown position
   */
  position?: 'bottom' | 'top' | 'auto';
}

const Select: React.FC<SelectProps> = ({
  options,
  value,
  defaultValue,
  onChange,
  placeholder = 'Select an option',
  multiple = false,
  searchable = false,
  clearable = false,
  disabled = false,
  loading = false,
  error,
  size = 'md',
  variant = 'outline',
  label,
  helperText,
  required = false,
  maxHeight = 300,
  renderOption,
  renderValue,
  fullWidth = false,
  className,
  position = 'auto',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [localValue, setLocalValue] = useState<string | number | (string | number)[]>(
    defaultValue || (multiple ? [] : '')
  );
  const [dropdownPosition, setDropdownPosition] = useState<'bottom' | 'top'>('bottom');

  const selectRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Use controlled or uncontrolled value
  const effectiveValue = value !== undefined ? value : localValue;

  // Filter options based on search
  const filteredOptions = useMemo(() => {
    if (!searchable || !searchTerm) return options;

    return options.filter(option =>
      option.label.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [options, searchTerm, searchable]);

  // Group options by group property
  const groupedOptions = useMemo(() => {
    const groups: Record<string, SelectOption[]> = {};
    const ungrouped: SelectOption[] = [];

    filteredOptions.forEach(option => {
      if (option.group) {
        if (!groups[option.group]) {
          groups[option.group] = [];
        }
        groups[option.group].push(option);
      } else {
        ungrouped.push(option);
      }
    });

    return { groups, ungrouped };
  }, [filteredOptions]);

  // Get selected options
  const selectedOptions = useMemo(() => {
    if (multiple) {
      const values = Array.isArray(effectiveValue) ? effectiveValue : [];
      return options.filter(opt => values.includes(opt.value));
    } else {
      return options.filter(opt => opt.value === effectiveValue);
    }
  }, [effectiveValue, options, multiple]);

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Auto-position dropdown
  useEffect(() => {
    if (!isOpen || position !== 'auto') return;

    const updatePosition = () => {
      if (!selectRef.current) return;

      const rect = selectRef.current.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;

      setDropdownPosition(spaceBelow < maxHeight && spaceAbove > spaceBelow ? 'top' : 'bottom');
    };

    updatePosition();
    window.addEventListener('scroll', updatePosition);
    window.addEventListener('resize', updatePosition);

    return () => {
      window.removeEventListener('scroll', updatePosition);
      window.removeEventListener('resize', updatePosition);
    };
  }, [isOpen, position, maxHeight]);

  // Focus search input when opened
  useEffect(() => {
    if (isOpen && searchable && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen, searchable]);

  // Handle select
  const handleSelect = (option: SelectOption) => {
    if (option.disabled) return;

    let newValue: string | number | (string | number)[];

    if (multiple) {
      const currentValues = Array.isArray(effectiveValue) ? effectiveValue : [];
      if (currentValues.includes(option.value)) {
        newValue = currentValues.filter(v => v !== option.value);
      } else {
        newValue = [...currentValues, option.value];
      }
    } else {
      newValue = option.value;
      setIsOpen(false);
    }

    if (onChange) {
      onChange(newValue);
    } else {
      setLocalValue(newValue);
    }

    setSearchTerm('');
  };

  // Handle clear
  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    const newValue = multiple ? [] : '';

    if (onChange) {
      onChange(newValue);
    } else {
      setLocalValue(newValue);
    }
  };

  // Check if value is selected
  const isSelected = (optionValue: string | number) => {
    if (multiple) {
      const values = Array.isArray(effectiveValue) ? effectiveValue : [];
      return values.includes(optionValue);
    }
    return effectiveValue === optionValue;
  };

  // Size styles
  const sizes = {
    xs: 'px-2.5 py-1.5 text-xs',
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2.5 text-base',
    lg: 'px-5 py-3 text-lg',
    xl: 'px-6 py-3.5 text-xl',
  };

  // Variant styles
  const variants = {
    outline: `
      bg-white dark:bg-gray-900
      border border-gray-300 dark:border-gray-700
      hover:border-gray-400 dark:hover:border-gray-600
    `,
    filled: `
      bg-gray-100 dark:bg-gray-800
      border border-transparent
      hover:bg-gray-200 dark:hover:bg-gray-700
    `,
    flushed: `
      bg-transparent
      border-0 border-b-2 border-gray-300 dark:border-gray-700
      hover:border-gray-400 dark:hover:border-gray-600
      rounded-none px-0
    `,
  };

  // Base classes
  const baseClasses = `
    relative w-full
    text-gray-900 dark:text-gray-100
    transition-all duration-200
    cursor-pointer
    ${disabled ? 'cursor-not-allowed opacity-50' : ''}
  `;

  const selectClasses = cn(
    baseClasses,
    variants[variant],
    sizes[size],
    error && 'border-danger-500 dark:border-danger-400',
    isOpen && 'ring-2 ring-primary-500 ring-offset-0',
    'rounded-lg flex items-center justify-between',
    className
  );

  const wrapperClasses = cn(
    'relative',
    fullWidth ? 'w-full' : 'inline-flex'
  );

  // Render option content
  const renderOptionContent = (option: SelectOption, selected: boolean) => {
    if (renderOption) {
      return renderOption(option, selected);
    }

    return (
      <>
        {option.icon && <span className="mr-2">{option.icon}</span>}
        <span className="flex-1">{option.label}</span>
        {multiple && selected && (
          <Check className="w-4 h-4 text-primary-500 ml-2" />
        )}
      </>
    );
  };

  // Render selected value
  const renderSelectedValue = () => {
    if (renderValue) {
      return renderValue(effectiveValue, options);
    }

    if (multiple) {
      if (selectedOptions.length === 0) {
        return <span className="text-gray-400">{placeholder}</span>;
      }
      return (
        <div className="flex flex-wrap gap-1">
          {selectedOptions.map(option => (
            <span
              key={option.value}
              className="inline-flex items-center px-2 py-1 rounded-md bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 text-xs"
            >
              {option.label}
              {!disabled && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSelect(option);
                  }}
                  className="ml-1 hover:text-primary-900 dark:hover:text-primary-100"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </span>
          ))}
        </div>
      );
    }

    const selected = selectedOptions[0];
    if (!selected) {
      return <span className="text-gray-400">{placeholder}</span>;
    }

    return (
      <div className="flex items-center">
        {selected.icon && <span className="mr-2">{selected.icon}</span>}
        <span>{selected.label}</span>
      </div>
    );
  };

  const actualPosition = position === 'auto' ? dropdownPosition : position;

  return (
    <div className={wrapperClasses}>
      {/* Label */}
      {label && (
        <label className="block mb-1.5 text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
          {required && <span className="ml-1 text-danger-500">*</span>}
        </label>
      )}

      {/* Select */}
      <div ref={selectRef}>
        <div
          className={selectClasses}
          onClick={() => !disabled && setIsOpen(!isOpen)}
        >
          {/* Selected value */}
          <div className="flex-1 min-w-0">
            {renderSelectedValue()}
          </div>

          {/* Icons */}
          <div className="flex items-center gap-1">
            {/* Clear button */}
            {clearable && selectedOptions.length > 0 && !disabled && (
              <button
                onClick={handleClear}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            )}

            {/* Loading spinner */}
            {loading && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              </motion.div>
            )}

            {/* Dropdown arrow */}
            <motion.div
              animate={{ rotate: isOpen ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronDown className="w-4 h-4 text-gray-400" />
            </motion.div>
          </div>
        </div>

        {/* Dropdown */}
        <AnimatePresence>
          {isOpen && !disabled && (
            <motion.div
              ref={dropdownRef}
              initial={{ opacity: 0, y: actualPosition === 'top' ? 10 : -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: actualPosition === 'top' ? 10 : -10 }}
              transition={{ duration: 0.15 }}
              className={cn(
                'absolute z-50 w-full mt-1 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden',
                actualPosition === 'top' && 'bottom-full mb-1 mt-0'
              )}
              style={{ maxHeight }}
            >
              {/* Search input */}
              {searchable && (
                <div className="p-2 border-b border-gray-200 dark:border-gray-700">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                      ref={searchInputRef}
                      type="text"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      placeholder="Search..."
                      className="w-full pl-9 pr-3 py-2 text-sm bg-gray-50 dark:bg-gray-900 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                      onClick={(e) => e.stopPropagation()}
                    />
                  </div>
                </div>
              )}

              {/* Options list */}
              <div className="overflow-y-auto" style={{ maxHeight: maxHeight - (searchable ? 60 : 0) }}>
                {filteredOptions.length === 0 ? (
                  <div className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400 text-center">
                    No options found
                  </div>
                ) : (
                  <>
                    {/* Ungrouped options */}
                    {groupedOptions.ungrouped.map(option => (
                      <div
                        key={option.value}
                        onClick={() => handleSelect(option)}
                        className={cn(
                          'px-4 py-2.5 cursor-pointer transition-colors flex items-center',
                          option.disabled && 'opacity-50 cursor-not-allowed',
                          isSelected(option.value)
                            ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                            : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                        )}
                      >
                        {renderOptionContent(option, isSelected(option.value))}
                      </div>
                    ))}

                    {/* Grouped options */}
                    {Object.entries(groupedOptions.groups).map(([groupName, groupOptions]) => (
                      <div key={groupName}>
                        <div className="px-4 py-2 text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase">
                          {groupName}
                        </div>
                        {groupOptions.map(option => (
                          <div
                            key={option.value}
                            onClick={() => handleSelect(option)}
                            className={cn(
                              'px-4 py-2.5 cursor-pointer transition-colors flex items-center',
                              option.disabled && 'opacity-50 cursor-not-allowed',
                              isSelected(option.value)
                                ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                                : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                            )}
                          >
                            {renderOptionContent(option, isSelected(option.value))}
                          </div>
                        ))}
                      </div>
                    ))}
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Helper text / Error message */}
      {(error || helperText) && (
        <p className={cn(
          'mt-1.5 text-sm',
          error ? 'text-danger-500 dark:text-danger-400' : 'text-gray-500 dark:text-gray-400'
        )}>
          {typeof error === 'string' ? error : helperText}
        </p>
      )}
    </div>
  );
};

Select.displayName = 'Select';

export default Select;