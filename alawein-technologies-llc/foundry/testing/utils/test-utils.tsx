import React, { ReactElement } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

// Create a custom render function that includes providers
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialEntries?: string[];
  route?: string;
  queryClient?: QueryClient;
}

const createTestQueryClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
    logger: {
      log: console.log,
      warn: console.warn,
      error: () => {},
    },
  });

export const AllTheProviders: React.FC<{
  children: React.ReactNode;
  queryClient?: QueryClient;
}> = ({ children, queryClient = createTestQueryClient() }) => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
};

export const customRender = (
  ui: ReactElement,
  {
    route = '/',
    initialEntries = [route],
    queryClient,
    ...options
  }: CustomRenderOptions = {}
): RenderResult & { user: ReturnType<typeof userEvent.setup> } => {
  window.history.pushState({}, 'Test page', route);

  const user = userEvent.setup();

  const rendered = render(ui, {
    wrapper: ({ children }) => (
      <AllTheProviders queryClient={queryClient}>{children}</AllTheProviders>
    ),
    ...options,
  });

  return {
    ...rendered,
    user,
  };
};

// Mock data generators
export const mockUser = (overrides = {}) => ({
  id: '1',
  email: 'test@example.com',
  name: 'Test User',
  role: 'researcher',
  organization: 'Test Org',
  expertise: ['Physics', 'Computer Science'],
  avatar: 'https://example.com/avatar.jpg',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  ...overrides,
});

export const mockProject = (overrides = {}) => ({
  id: '1',
  title: 'Test Project',
  description: 'A test project description',
  status: 'active',
  owner: mockUser(),
  collaborators: [],
  tags: ['test', 'project'],
  visibility: 'public',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  ...overrides,
});

export const mockIdea = (overrides = {}) => ({
  id: '1',
  title: 'Revolutionary Idea',
  description: 'An amazing idea that combines multiple domains',
  domains: ['AI', 'Healthcare'],
  score: 95,
  potential: 'high',
  feasibility: 'medium',
  impact: 'transformative',
  author: mockUser(),
  votes: 42,
  comments: [],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  ...overrides,
});

// API mock utilities
export const mockApiResponse = <T>(data: T, delay = 0): Promise<T> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(data), delay);
  });
};

export const mockApiError = (message: string, status = 400, delay = 0): Promise<never> => {
  return new Promise((_, reject) => {
    setTimeout(() => {
      reject({
        response: {
          status,
          data: { message },
        },
      });
    }, delay);
  });
};

// WebSocket mock
export class MockWebSocket {
  url: string;
  readyState: number;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    this.readyState = WebSocket.CONNECTING;
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  send(data: string): void {
    console.log('MockWebSocket sending:', data);
  }

  close(): void {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  simulateMessage(data: any): void {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }

  simulateError(): void {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }
}

// Accessibility testing utilities
export const checkAccessibility = async (container: HTMLElement): Promise<void> => {
  const axe = await import('axe-core');
  const results = await axe.run(container);

  if (results.violations.length > 0) {
    const violations = results.violations.map(v => ({
      id: v.id,
      impact: v.impact,
      description: v.description,
      nodes: v.nodes.length,
    }));

    console.error('Accessibility violations:', violations);
    throw new Error(`Found ${results.violations.length} accessibility violations`);
  }
};

// Performance testing utilities
export const measureRenderTime = async (
  component: ReactElement
): Promise<number> => {
  const startTime = performance.now();
  render(component);
  const endTime = performance.now();
  return endTime - startTime;
};

// Snapshot testing utilities
export const createSnapshot = (component: ReactElement, name: string): void => {
  const { container } = render(component);
  expect(container.firstChild).toMatchSnapshot(name);
};

// Form testing utilities
export const fillForm = async (
  user: ReturnType<typeof userEvent.setup>,
  formData: Record<string, string>
): Promise<void> => {
  for (const [fieldName, value] of Object.entries(formData)) {
    const field = document.querySelector(`[name="${fieldName}"]`) as HTMLElement;
    if (field) {
      await user.type(field, value);
    }
  }
};

// Wait utilities
export const waitForLoadingToFinish = async (): Promise<void> => {
  const { waitFor } = await import('@testing-library/react');
  await waitFor(() => {
    expect(document.querySelector('[data-testid="loading"]')).not.toBeInTheDocument();
  });
};

// Local storage mock
export const mockLocalStorage = (): void => {
  const localStorageMock = {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
  };

  global.localStorage = localStorageMock as any;
};

// Export everything
export * from '@testing-library/react';
export { userEvent };