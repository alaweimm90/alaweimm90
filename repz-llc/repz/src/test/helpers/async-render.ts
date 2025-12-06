/**
 * Async Rendering Utilities
 * Helpers for testing components with async state updates
 */

import { render, waitFor, type RenderResult } from '@testing-library/react';
import { act } from 'react';
import type { ReactElement } from 'react';

/**
 * Renders a component and waits for all async effects to complete
 * Eliminates React act() warnings for components with useEffect hooks that update state
 *
 * @example
 * ```typescript
 * import { renderAsync } from '@/test/helpers/async-render';
 *
 * it('renders dashboard', async () => {
 *   const { getByText } = await renderAsync(<Dashboard />);
 *   expect(getByText('Welcome')).toBeInTheDocument();
 * });
 * ```
 */
export const renderAsync = async (
  component: ReactElement
): Promise<RenderResult> => {
  let result: RenderResult;

  await act(async () => {
    result = render(component);
    // Allow all pending promises and timers to resolve
    await new Promise(resolve => setTimeout(resolve, 0));
  });

  return result!;
};

/**
 * Renders a component and waits for a specific condition
 * Useful for components with complex async initialization
 *
 * @example
 * ```typescript
 * const { getByText } = await renderAndWaitFor(
 *   <Dashboard />,
 *   () => screen.getByText('Dashboard loaded')
 * );
 * ```
 */
export const renderAndWaitFor = async <T = void>(
  component: ReactElement,
  condition: () => T,
  options?: {
    timeout?: number;
    interval?: number;
  }
): Promise<RenderResult> => {
  const result = await renderAsync(component);

  await waitFor(condition, {
    timeout: options?.timeout ?? 3000,
    interval: options?.interval ?? 50,
  });

  return result;
};

/**
 * Waits for all async state updates to complete
 * Use this after triggering async actions in tests
 *
 * @example
 * ```typescript
 * userEvent.click(submitButton);
 * await waitForAsyncUpdates();
 * expect(successMessage).toBeInTheDocument();
 * ```
 */
export const waitForAsyncUpdates = async (timeout: number = 1000) => {
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 0));
  });

  await waitFor(() => {}, { timeout });
};

/**
 * Wraps a component render with error boundary for testing error scenarios
 *
 * @example
 * ```typescript
 * const { error } = await renderWithErrorBoundary(<ComponentThatThrows />);
 * expect(error).toBeDefined();
 * ```
 */
export const renderWithErrorBoundary = async (
  component: ReactElement
): Promise<RenderResult & { error?: Error }> => {
  let caughtError: Error | undefined;

  const ErrorBoundary = ({ children }: { children: React.ReactNode }) => {
    const [error, setError] = React.useState<Error>();

    React.useEffect(() => {
      const handleError = (event: ErrorEvent) => {
        setError(event.error);
        caughtError = event.error;
        event.preventDefault();
      };

      window.addEventListener('error', handleError);
      return () => window.removeEventListener('error', handleError);
    }, []);

    if (error) {
      return <div role="alert">Error: {error.message}</div>;
    }

    return <>{children}</>;
  };

  const result = await renderAsync(
    <ErrorBoundary>{component}</ErrorBoundary>
  );

  return { ...result, error: caughtError };
};
