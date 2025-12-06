/**
 * REPZ API Client Utilities
 * Type-safe API client with automatic validation and error handling
 */

import type { components, paths } from '../types/generated';

// Extract operation types from OpenAPI paths
type ApiPaths = paths;
type ApiComponents = components;

// Helper types for extracting request/response types
type GetOperationRequestBody<T> = T extends { requestBody: { content: { 'application/json': infer R } } } ? R : never;
type GetOperationResponse<T> = T extends { responses: { 200: { content: { 'application/json': infer R } } } } ? R : never;

/**
 * Configuration for the API client
 */
export interface ApiClientConfig {
  /** Base URL for the API */
  baseUrl: string;
  
  /** Default headers to include with requests */
  defaultHeaders?: Record<string, string>;
  
  /** Authentication token */
  token?: string;
  
  /** Request timeout in milliseconds */
  timeout?: number;
  
  /** Enable request/response logging */
  debug?: boolean;
  
  /** Custom fetch implementation */
  fetch?: typeof fetch;
  
  /** Request interceptor */
  beforeRequest?: (request: RequestInit) => RequestInit | Promise<RequestInit>;
  
  /** Response interceptor */
  afterResponse?: (response: Response) => Response | Promise<Response>;
  
  /** Error handler */
  onError?: (error: ApiError) => void;
}

/**
 * API Error class with structured error information
 */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: any,
    public path?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }

  static async fromResponse(response: Response, path?: string): Promise<ApiError> {
    let errorData: any = {};
    
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        errorData = await response.json();
      } else {
        errorData.message = await response.text();
      }
    } catch {
      errorData.message = `HTTP ${response.status} ${response.statusText}`;
    }

    return new ApiError(
      response.status,
      errorData.error || 'UNKNOWN_ERROR',
      errorData.message || `HTTP ${response.status} ${response.statusText}`,
      errorData.details,
      path
    );
  }
}

/**
 * Type-safe API client for REPZ platform
 */
export class RepzApiClient {
  private config: Required<ApiClientConfig>;
  private fetchImpl: typeof fetch;

  constructor(config: ApiClientConfig) {
    this.config = {
      baseUrl: config.baseUrl.replace(/\/$/, ''), // Remove trailing slash
      defaultHeaders: {
        'Content-Type': 'application/json',
        ...config.defaultHeaders
      },
      token: config.token || '',
      timeout: config.timeout || 30000,
      debug: config.debug || false,
      fetch: config.fetch || globalThis.fetch,
      beforeRequest: config.beforeRequest || ((req) => req),
      afterResponse: config.afterResponse || ((res) => res),
      onError: config.onError || (() => {})
    };

    this.fetchImpl = this.config.fetch;
    
    // Add authorization header if token provided
    if (this.config.token) {
      this.config.defaultHeaders['Authorization'] = `Bearer ${this.config.token}`;
    }
  }

  /**
   * Update the authentication token
   */
  setToken(token: string): void {
    this.config.token = token;
    this.config.defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  /**
   * Clear the authentication token
   */
  clearToken(): void {
    this.config.token = '';
    delete this.config.defaultHeaders['Authorization'];
  }

  /**
   * Generic request method with type safety and error handling
   */
  async request<TResponse = any>(
    method: string,
    path: string,
    options: {
      params?: Record<string, any>;
      body?: any;
      headers?: Record<string, string>;
      timeout?: number;
    } = {}
  ): Promise<TResponse> {
    const url = this.buildUrl(path, options.params);
    
    let requestInit: RequestInit = {
      method: method.toUpperCase(),
      headers: {
        ...this.config.defaultHeaders,
        ...options.headers
      }
    };

    // Add request body for non-GET requests
    if (options.body && method.toUpperCase() !== 'GET') {
      requestInit.body = JSON.stringify(options.body);
    }

    // Apply request timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      options.timeout || this.config.timeout
    );
    requestInit.signal = controller.signal;

    try {
      // Apply request interceptor
      requestInit = await this.config.beforeRequest(requestInit);

      // Debug logging
      if (this.config.debug) {
        console.log(`API Request: ${method.toUpperCase()} ${url}`, {
          headers: requestInit.headers,
          body: requestInit.body
        });
      }

      // Make the request
      let response = await this.fetchImpl(url, requestInit);
      
      // Apply response interceptor
      response = await this.config.afterResponse(response);

      // Clear timeout
      clearTimeout(timeoutId);

      // Handle non-2xx responses
      if (!response.ok) {
        const error = await ApiError.fromResponse(response, path);
        this.config.onError(error);
        throw error;
      }

      // Parse response
      const contentType = response.headers.get('content-type');
      let data: TResponse;
      
      if (contentType?.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text() as any;
      }

      // Debug logging
      if (this.config.debug) {
        console.log(`API Response: ${method.toUpperCase()} ${url}`, data);
      }

      return data;

    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof ApiError) {
        throw error;
      }
      
      // Handle network errors, timeouts, etc.
      const apiError = new ApiError(
        0,
        'NETWORK_ERROR',
        error instanceof Error ? error.message : 'Network request failed',
        error,
        path
      );
      
      this.config.onError(apiError);
      throw apiError;
    }
  }

  /**
   * GET request
   */
  async get<TResponse = any>(
    path: string,
    params?: Record<string, any>,
    options?: { headers?: Record<string, string>; timeout?: number }
  ): Promise<TResponse> {
    return this.request<TResponse>('GET', path, { params, ...options });
  }

  /**
   * POST request
   */
  async post<TResponse = any, TBody = any>(
    path: string,
    body?: TBody,
    options?: { params?: Record<string, any>; headers?: Record<string, string>; timeout?: number }
  ): Promise<TResponse> {
    return this.request<TResponse>('POST', path, { body, ...options });
  }

  /**
   * PATCH request
   */
  async patch<TResponse = any, TBody = any>(
    path: string,
    body?: TBody,
    options?: { params?: Record<string, any>; headers?: Record<string, string>; timeout?: number }
  ): Promise<TResponse> {
    return this.request<TResponse>('PATCH', path, { body, ...options });
  }

  /**
   * PUT request
   */
  async put<TResponse = any, TBody = any>(
    path: string,
    body?: TBody,
    options?: { params?: Record<string, any>; headers?: Record<string, string>; timeout?: number }
  ): Promise<TResponse> {
    return this.request<TResponse>('PUT', path, { body, ...options });
  }

  /**
   * DELETE request
   */
  async delete<TResponse = any>(
    path: string,
    options?: { params?: Record<string, any>; headers?: Record<string, string>; timeout?: number }
  ): Promise<TResponse> {
    return this.request<TResponse>('DELETE', path, options);
  }

  // Type-safe endpoint methods
  
  /**
   * Authentication endpoints
   */
  auth = {
    login: (credentials: GetOperationRequestBody<ApiPaths['/auth/login']['post']>) =>
      this.post<GetOperationResponse<ApiPaths['/auth/login']['post']>>('/auth/login', credentials),
      
    register: (userData: GetOperationRequestBody<ApiPaths['/auth/register']['post']>) =>
      this.post<GetOperationResponse<ApiPaths['/auth/register']['post']>>('/auth/register', userData),
      
    refresh: (refreshData: GetOperationRequestBody<ApiPaths['/auth/refresh']['post']>) =>
      this.post<GetOperationResponse<ApiPaths['/auth/refresh']['post']>>('/auth/refresh', refreshData)
  };

  /**
   * User management endpoints
   */
  users = {
    me: () =>
      this.get<GetOperationResponse<ApiPaths['/users/me']['get']>>('/users/me'),
      
    updateMe: (updates: GetOperationRequestBody<ApiPaths['/users/me']['patch']>) =>
      this.patch<GetOperationResponse<ApiPaths['/users/me']['patch']>>('/users/me', updates)
  };

  /**
   * Subscription endpoints
   */
  subscriptions = {
    list: () =>
      this.get<GetOperationResponse<ApiPaths['/subscriptions']['get']>>('/subscriptions'),
      
    create: (subscription: GetOperationRequestBody<ApiPaths['/subscriptions']['post']>) =>
      this.post<GetOperationResponse<ApiPaths['/subscriptions']['post']>>('/subscriptions', subscription),
      
    cancel: (subscriptionId: string) =>
      this.post<GetOperationResponse<ApiPaths['/subscriptions/{subscriptionId}/cancel']['post']>>(
        `/subscriptions/${subscriptionId}/cancel`
      )
  };

  /**
   * Client data endpoints
   */
  clientData = {
    get: (params?: { include?: string[]; startDate?: string; endDate?: string }) =>
      this.get<GetOperationResponse<ApiPaths['/clients/data']['get']>>('/clients/data', params),
      
    update: (updates: GetOperationRequestBody<ApiPaths['/clients/data']['patch']>) =>
      this.patch<GetOperationResponse<ApiPaths['/clients/data']['patch']>>('/clients/data', updates)
  };

  /**
   * Workout endpoints
   */
  workouts = {
    list: (params?: { status?: string; startDate?: string; endDate?: string; limit?: number; offset?: number }) =>
      this.get<GetOperationResponse<ApiPaths['/workouts']['get']>>('/workouts', params),
      
    create: (workout: GetOperationRequestBody<ApiPaths['/workouts']['post']>) =>
      this.post<GetOperationResponse<ApiPaths['/workouts']['post']>>('/workouts', workout),
      
    get: (workoutId: string) =>
      this.get<GetOperationResponse<ApiPaths['/workouts/{workoutId}']['get']>>(`/workouts/${workoutId}`),
      
    update: (workoutId: string, updates: GetOperationRequestBody<ApiPaths['/workouts/{workoutId}']['patch']>) =>
      this.patch<GetOperationResponse<ApiPaths['/workouts/{workoutId}']['patch']>>(
        `/workouts/${workoutId}`, updates
      )
  };

  /**
   * Analytics endpoints
   */
  analytics = {
    dashboard: (params?: { period?: string; metrics?: string[] }) =>
      this.get<GetOperationResponse<ApiPaths['/analytics/dashboard']['get']>>('/analytics/dashboard', params)
  };

  /**
   * Feature flags endpoints
   */
  featureFlags = {
    list: () =>
      this.get<GetOperationResponse<ApiPaths['/feature-flags']['get']>>('/feature-flags')
  };

  // Private helper methods

  private buildUrl(path: string, params?: Record<string, any>): string {
    const url = new URL(path, this.config.baseUrl);
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          if (Array.isArray(value)) {
            value.forEach(v => url.searchParams.append(key, String(v)));
          } else {
            url.searchParams.append(key, String(value));
          }
        }
      });
    }
    
    return url.toString();
  }
}

/**
 * Create a pre-configured API client for REPZ
 */
export function createRepzApiClient(config: Partial<ApiClientConfig> = {}): RepzApiClient {
  const defaultConfig: ApiClientConfig = {
    baseUrl: import.meta.env.VITE_API_BASE_URL || 'https://api.repzcoach.com/v1',
    defaultHeaders: {
      'Content-Type': 'application/json',
      'User-Agent': 'REPZ-WebApp/1.0.0'
    },
    timeout: 30000,
    debug: !import.meta.env.PROD,
    onError: (error) => {
      console.error('API Error:', error);
      
      // Send to error reporting service (e.g., Sentry)
      if (window.Sentry) {
        window.Sentry.captureException(error);
      }
    }
  };

  return new RepzApiClient({ ...defaultConfig, ...config });
}