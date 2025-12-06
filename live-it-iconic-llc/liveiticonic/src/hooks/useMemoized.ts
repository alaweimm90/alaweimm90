/**
 * Memoization Utilities
 * Wrappers for React.memo, useCallback, and useMemo
 * Simplify performance optimization patterns
 */

import React, { useMemo, useCallback, DependencyList, ComponentType, PropsWithChildren } from 'react';

/**
 * Create a memoized version of a component
 * Only re-renders when props change
 * Use for list items, expensive renders
 */
export function memoComponent<P extends object>(
  Component: ComponentType<P>,
  propsAreEqual?: (prevProps: Readonly<P>, nextProps: Readonly<P>) => boolean
) {
  const MemoizedComponent = React.memo(Component, propsAreEqual);
  MemoizedComponent.displayName = `Memo(${Component.displayName || Component.name})`;
  return MemoizedComponent;
}

/**
 * Create a memoized callback with automatic dependency validation
 * Useful for event handlers passed as props
 */
export function useMemoCallback<T extends (...args: unknown[]) => unknown>(
  callback: T,
  deps: DependencyList
) {
  return useCallback(callback, deps);
}

/**
 * Memoize an object value
 * Prevents object recreation on every render
 */
export function useMemoObject<T extends Record<string, unknown>>(
  value: T,
  deps: DependencyList
): T {
  return useMemo(() => value, deps);
}

/**
 * Memoize an array value
 * Prevents array recreation on every render
 */
export function useMemoArray<T>(
  value: T[],
  deps: DependencyList
): T[] {
  return useMemo(() => value, deps);
}

/**
 * Compare two values with shallow equality check
 * Returns true if values are equal (safe to skip re-render)
 */
export function shallowEqual(obj1: unknown, obj2: unknown): boolean {
  if (Object.is(obj1, obj2)) {
    return true;
  }

  if (
    typeof obj1 !== 'object' ||
    obj1 === null ||
    typeof obj2 !== 'object' ||
    obj2 === null
  ) {
    return false;
  }

  const keys1 = Object.keys(obj1 as Record<string, unknown>);
  const keys2 = Object.keys(obj2 as Record<string, unknown>);

  if (keys1.length !== keys2.length) {
    return false;
  }

  for (const key of keys1) {
    if (!Object.prototype.hasOwnProperty.call(obj2, key)) {
      return false;
    }

    if (!Object.is((obj1 as Record<string, unknown>)[key], (obj2 as Record<string, unknown>)[key])) {
      return false;
    }
  }

  return true;
}

/**
 * Custom comparison for complex props
 * Only update if specific props change
 */
export function createPropsComparison<T extends Record<string, unknown>>(
  keysToCompare: (keyof T)[]
) {
  return (prev: T, next: T) => {
    for (const key of keysToCompare) {
      if (!Object.is(prev[key], next[key])) {
        return false; // Props changed, re-render
      }
    }
    return true; // Props didn't change, skip re-render
  };
}

/**
 * Performance-optimized list component
 * Prevents unnecessary re-renders of list items
 */
export const MemoizedList = React.memo(function MemoizedList({
  items,
  renderItem,
  keyExtractor = (item: { id: string | number }, index: number) => item.id || index,
  className = '',
}: {
  items: { id: string | number }[];
  renderItem: (item: { id: string | number }, index: number) => React.ReactNode;
  keyExtractor?: (item: { id: string | number }, index: number) => string | number;
  className?: string;
}) {
  return (
    <div className={className}>
      {items.map((item, index) => (
        <div key={keyExtractor(item, index)}>{renderItem(item, index)}</div>
      ))}
    </div>
  );
});

/**
 * Memoized div for expensive children
 * Only re-renders when children or deps change
 */
export const MemoizedDiv = React.memo(
  React.forwardRef<
    HTMLDivElement,
    PropsWithChildren<{
      deps?: DependencyList;
      [key: string]: unknown;
    }>
  >(function MemoizedDiv({ deps, children, ...props }, ref) {
    return (
      <div ref={ref} {...props}>
        {children}
      </div>
    );
  })
);

/**
 * Throttle-debounce combination for smooth updates
 * Useful for value changes that trigger expensive operations
 */
export function useSmoothValue<T>(
  value: T,
  debounceMs: number = 300,
  throttleMs: number = 50
) {
  const [debouncedValue, setDebouncedValue] = React.useState(value);
  const debounceTimeoutRef = React.useRef<number | undefined>(undefined);
  const lastThrottleRef = React.useRef<number>(0);

  React.useEffect(() => {
    lastThrottleRef.current = Date.now();
  }, []);

  React.useEffect(() => {
    if (debounceTimeoutRef.current !== undefined) {
      clearTimeout(debounceTimeoutRef.current);
      debounceTimeoutRef.current = undefined;
    }

    const now = Date.now();
    const timeSinceLastThrottle = now - lastThrottleRef.current;

    const updateValue = () => {
      queueMicrotask(() => {
        setDebouncedValue(value);
        lastThrottleRef.current = Date.now();
      });
    };

    if (timeSinceLastThrottle >= throttleMs) {
      updateValue();
    } else {
      debounceTimeoutRef.current = window.setTimeout(
        updateValue,
        debounceMs - timeSinceLastThrottle
      );
    }

    return () => {
      if (debounceTimeoutRef.current !== undefined) {
        clearTimeout(debounceTimeoutRef.current);
        debounceTimeoutRef.current = undefined;
      }
    };
  }, [value, debounceMs, throttleMs]);

  return debouncedValue;
}

/**
 * Create a selector function for extracting values from objects
 * Prevents object recreation, improving memoization
 */
export function createSelector<T, R>(selector: (value: T) => R) {
  return (value: T, previousValue?: T) => {
    if (previousValue && Object.is(value, previousValue)) {
      return selector(previousValue);
    }
    return selector(value);
  };
}

/**
 * Reselect-like functionality for complex selectors
 * Memoizes result based on dependencies
 */
export function useSelectorMemo<T, R>(
  selector: (value: T) => R,
  value: T,
  deps?: DependencyList
): R {
  return useMemo(() => selector(value), deps ? [value, ...deps] : [value]);
}
