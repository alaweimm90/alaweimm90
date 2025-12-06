import React, { createContext, useContext, useEffect, useState } from 'react';
import * as Sentry from '@sentry/react';

interface SecurityConfig {
  csp: string;
  rateLimiting: boolean;
  encryptionEnabled: boolean;
  auditLogging: boolean;
  sessionTimeout: number;
}

interface SecurityContextType {
  config: SecurityConfig;
  reportSecurityEvent: (event: string, details: Record<string, unknown>) => void;
  checkRateLimit: (identifier: string) => boolean;
  validateSession: () => boolean;
  encryptData: (data: unknown) => string;
  decryptData: (encryptedData: string) => unknown;
}

const SecurityContext = createContext<SecurityContextType | null>(null);

export const useSecurityContext = () => {
  const context = useContext(SecurityContext);
  if (!context) {
    throw new Error('useSecurityContext must be used within SecurityProvider');
  }
  return context;
};

interface SecurityProviderProps {
  children: React.ReactNode;
}

// Rate limiting store (in production, use Redis)
const rateLimitStore = new Map<string, { count: number; resetTime: number }>();

export const SecurityProvider: React.FC<SecurityProviderProps> = ({ children }) => {
  const [config] = useState<SecurityConfig>({
    csp: "default-src 'self'; script-src 'self' 'unsafe-inline' *.stripe.com *.supabase.co; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' *.supabase.co *.stripe.com *.openai.com; frame-src 'self' *.stripe.com",
    rateLimiting: true,
    encryptionEnabled: true,
    auditLogging: true,
    sessionTimeout: 3600000, // 1 hour
  });

  useEffect(() => {
    // Initialize Sentry for error monitoring
    Sentry.init({
      dsn: import.meta.env.VITE_SENTRY_DSN || undefined,
      environment: import.meta.env.VITE_ENVIRONMENT || import.meta.env.MODE || 'development',
      tracesSampleRate: import.meta.env.MODE === 'production' ? 0.1 : 1.0,
      profilesSampleRate: import.meta.env.MODE === 'production' ? 0.1 : 1.0,
    });

    // Set CSP header (in production, this should be done server-side)
    if (typeof document !== 'undefined') {
      const metaCsp = document.createElement('meta');
      metaCsp.httpEquiv = 'Content-Security-Policy';
      metaCsp.content = config.csp;
      document.head.appendChild(metaCsp);
    }

    // Session timeout handling
    const timeoutWarning = setTimeout(() => {
      console.warn('Session approaching timeout');
    }, config.sessionTimeout - 300000); // 5 minutes before timeout

    return () => clearTimeout(timeoutWarning);
  }, [config]);

  const reportSecurityEvent = (event: string, details: Record<string, unknown>) => {
    const securityEvent = {
      event,
      details,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
    };

    // Log to Sentry
    Sentry.addBreadcrumb({
      category: 'security',
      message: event,
      level: 'warning',
      data: details,
    });

    // In production, send to security monitoring service
    console.warn('Security Event:', securityEvent);
  };

  const checkRateLimit = (identifier: string): boolean => {
    if (!config.rateLimiting) return true;

    const now = Date.now();
    const windowMs = 15 * 60 * 1000; // 15 minutes
    const maxRequests = 100;

    const record = rateLimitStore.get(identifier);
    
    if (!record || now > record.resetTime) {
      rateLimitStore.set(identifier, { count: 1, resetTime: now + windowMs });
      return true;
    }

    if (record.count >= maxRequests) {
      reportSecurityEvent('rate_limit_exceeded', { identifier, count: record.count });
      return false;
    }

    record.count++;
    return true;
  };

  const validateSession = (): boolean => {
    const sessionStart = localStorage.getItem('session_start');
    if (!sessionStart) return false;

    const sessionAge = Date.now() - parseInt(sessionStart);
    if (sessionAge > config.sessionTimeout) {
      reportSecurityEvent('session_timeout', { sessionAge });
      localStorage.removeItem('session_start');
      return false;
    }

    return true;
  };

  const encryptData = (data: unknown): string => {
    if (!config.encryptionEnabled) return JSON.stringify(data);
    
    // Simple encryption for demo (use proper encryption in production)
    const jsonData = JSON.stringify(data);
    return btoa(jsonData);
  };

  const decryptData = (encryptedData: string): unknown => {
    if (!config.encryptionEnabled) return JSON.parse(encryptedData);
    
    try {
      const jsonData = atob(encryptedData);
      return JSON.parse(jsonData);
    } catch (error) {
      reportSecurityEvent('decryption_failed', { error: error instanceof Error ? error.message : 'Unknown error' });
      return null;
    }
  };

  const value: SecurityContextType = {
    config,
    reportSecurityEvent,
    checkRateLimit,
    validateSession,
    encryptData,
    decryptData,
  };

  return (
    <SecurityContext.Provider value={value}>
      {children}
    </SecurityContext.Provider>
  );
};
