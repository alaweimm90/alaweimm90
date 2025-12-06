// API types - HTTP request/response interfaces
// These types ensure consistent API communication across the application

// Base API response structure
export interface ApiResponse<T = unknown> {
  data: T;
  success: boolean;
  message?: string;
  timestamp: string;
  requestId: string;
}

// Error response structure
export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
    stack?: string; // Only in development
  };
  success: false;
  timestamp: string;
  requestId: string;
}

// Success response structure
export interface SuccessResponse<T = unknown> extends ApiResponse<T> {
  success: true;
}

// Paginated response structure
export interface PaginatedResponse<T = unknown> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

// API request options
export interface ApiRequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  headers?: Record<string, string>;
  body?: unknown;
  timeout?: number;
  retries?: number;
  cache?: boolean;
}

// Authentication API types
export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface LoginResponse {
  user: {
    id: string;
    email: string;
    role: string;
    tier?: string;
  };
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  tier?: string;
}

export interface RegisterResponse extends LoginResponse {
  isNewUser: true;
}

// User management API types
export interface UserUpdateRequest {
  firstName?: string;
  lastName?: string;
  email?: string;
  tier?: string;
  preferences?: Record<string, unknown>;
}

export interface UserResponse {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: string;
  tier?: string;
  isActive: boolean;
  lastLogin?: string;
  createdAt: string;
  updatedAt: string;
}

// Subscription API types
export interface SubscriptionCreateRequest {
  tier: string;
  billingCycle: string;
  paymentMethodId: string;
  couponCode?: string;
}

export interface SubscriptionResponse {
  id: string;
  userId: string;
  tier: string;
  billingCycle: string;
  status: string;
  currentPeriodStart: string;
  currentPeriodEnd: string;
  stripeSubscriptionId?: string;
  amount: number;
  currency: string;
  createdAt: string;
  updatedAt: string;
}

export interface SubscriptionUpdateRequest {
  tier?: string;
  billingCycle?: string;
  pauseCollection?: boolean;
}

// Payment API types
export interface PaymentMethodResponse {
  id: string;
  type: string;
  last4?: string;
  brand?: string;
  expiryMonth?: number;
  expiryYear?: number;
  isDefault: boolean;
}

export interface InvoiceResponse {
  id: string;
  subscriptionId: string;
  amount: number;
  currency: string;
  status: string;
  dueDate: string;
  paidAt?: string;
  downloadUrl: string;
  items: InvoiceItem[];
}

export interface InvoiceItem {
  description: string;
  amount: number;
  quantity: number;
  period: {
    start: string;
    end: string;
  };
}

// Analytics API types
export interface AnalyticsEventRequest {
  event: string;
  properties?: Record<string, unknown>;
  userId?: string;
  sessionId?: string;
  timestamp?: string;
}

export interface MetricsResponse {
  metrics: Record<string, number | string>;
  period: {
    start: string;
    end: string;
  };
  granularity: 'hour' | 'day' | 'week' | 'month';
}

// File upload API types
export interface FileUploadRequest {
  file: File;
  category: string;
  metadata?: Record<string, unknown>;
}

export interface FileUploadResponse {
  id: string;
  url: string;
  filename: string;
  size: number;
  mimeType: string;
  category: string;
  uploadedAt: string;
}

// Webhook types
export interface WebhookEvent {
  id: string;
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
  source: string;
}

export interface StripeWebhookEvent extends WebhookEvent {
  source: 'stripe';
  data: {
    object: Record<string, unknown>;
    previousAttributes?: Record<string, unknown>;
  };
}

// Rate limiting and quota types
export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
  retryAfter?: number;
}

export interface QuotaInfo {
  used: number;
  limit: number;
  period: string;
  resetsAt: string;
}

// API client configuration
export interface ApiClientConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  headers: Record<string, string>;
  interceptors?: {
    request?: <T = unknown>(config: T) => T;
    response?: <T = unknown>(response: T) => T;
    error?: (error: Error) => Error | Promise<never>;
  };
}

// Validation error types
export interface ValidationError {
  field: string;
  message: string;
  code: string;
  value?: unknown;
}

export interface ValidationErrorResponse extends ErrorResponse {
  error: {
    code: 'VALIDATION_ERROR';
    message: string;
    details: {
      errors: ValidationError[];
    };
  };
}

// Health check types
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  timestamp: string;
  checks: {
    [service: string]: {
      status: 'up' | 'down';
      responseTime?: number;
      error?: string;
    };
  };
}