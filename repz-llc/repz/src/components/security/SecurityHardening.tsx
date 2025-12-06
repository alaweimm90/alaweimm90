import React, { useCallback, useMemo, useState, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import DOMPurify from 'dompurify';
import { trackError } from '@/lib/monitoring';

// Security validation schemas
const emailSchema = z.string()
  .email('Invalid email format')
  .min(5, 'Email too short')
  .max(254, 'Email too long')
  .refine(
    (email) => !email.includes('<script>') && !email.includes('javascript:'),
    'Invalid characters in email'
  );

const passwordSchema = z.string()
  .min(8, 'Password must be at least 8 characters')
  .max(128, 'Password too long')
  .regex(/[A-Z]/, 'Password must contain uppercase letter')
  .regex(/[a-z]/, 'Password must contain lowercase letter')
  .regex(/[0-9]/, 'Password must contain number')
  .regex(/[^A-Za-z0-9]/, 'Password must contain special character');

const userInputSchema = z.string()
  .min(1, 'Input required')
  .max(1000, 'Input too long')
  .refine(
    (input) => !input.includes('<script>') && !input.includes('javascript:'),
    'Invalid characters detected'
  );

// Enhanced input sanitization
const sanitizeInput = (input: string): string => {
  if (typeof window !== 'undefined' && DOMPurify) {
    return DOMPurify.sanitize(input, { 
      ALLOWED_TAGS: [],
      ALLOWED_ATTR: [],
      ALLOW_DATA_ATTR: false 
    });
  }
  
  // Fallback sanitization
  return input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+\s*=/gi, '')
    .replace(/[<>]/g, '');
};

// CSRF protection
const useCSRFToken = () => {
  const [token, setToken] = useState<string>('');
  
  React.useEffect(() => {
    // Generate CSRF token
    const generateToken = () => {
      const array = new Uint8Array(32);
      crypto.getRandomValues(array);
      return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    };
    
    const csrfToken = generateToken();
    setToken(csrfToken);
    
    // Store in session storage
    sessionStorage.setItem('csrf_token', csrfToken);
  }, []);
  
  return token;
};

// Rate limiting hook
const useRateLimit = (maxAttempts = 5, windowMs = 60000) => {
  const attempts = useRef<number[]>([]);
  
  const checkRateLimit = useCallback(() => {
    const now = Date.now();
    const windowStart = now - windowMs;
    
    // Remove old attempts
    attempts.current = attempts.current.filter(time => time > windowStart);
    
    if (attempts.current.length >= maxAttempts) {
      return false; // Rate limited
    }
    
    attempts.current.push(now);
    return true; // Allowed
  }, [maxAttempts, windowMs]);
  
  const getRemainingCooldown = useCallback(() => {
    if (attempts.current.length < maxAttempts) return 0;
    
    const oldestAttempt = Math.min(...attempts.current);
    const cooldownEnd = oldestAttempt + windowMs;
    return Math.max(0, cooldownEnd - Date.now());
  }, [maxAttempts, windowMs]);
  
  return { checkRateLimit, getRemainingCooldown };
};

// Secure form component
interface SecureFormProps {
  onSubmit: (data: Record<string, unknown>) => Promise<void>;
  children: React.ReactNode;
  requireCSRF?: boolean;
  maxAttempts?: number;
}

const SecureForm: React.FC<SecureFormProps> = ({
  onSubmit,
  children,
  requireCSRF = true,
  maxAttempts = 5
}) => {
  const csrfToken = useCSRFToken();
  const { checkRateLimit, getRemainingCooldown } = useRateLimit(maxAttempts);
  const [isBlocked, setIsBlocked] = useState(false);
  const [error, setError] = useState<string>('');

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    // Check rate limiting
    if (!checkRateLimit()) {
      const cooldown = getRemainingCooldown();
      setIsBlocked(true);
      setError(`Too many attempts. Try again in ${Math.ceil(cooldown / 1000)} seconds.`);
      
      setTimeout(() => setIsBlocked(false), cooldown);
      return;
    }
    
    // CSRF validation
    if (requireCSRF) {
      const storedToken = sessionStorage.getItem('csrf_token');
      if (!storedToken || storedToken !== csrfToken) {
        setError('Security validation failed. Please refresh the page.');
        return;
      }
    }
    
    try {
      // Get form data
      const formData = new FormData(e.target as HTMLFormElement);
      const data = Object.fromEntries(formData.entries());
      
      // Add CSRF token to form data
      const secureData = {
        ...data,
        _csrf: csrfToken
      };
      
      await onSubmit(secureData);
    } catch (error) {
      setError('An error occurred. Please try again.');
      trackError(error as Error, { component: 'SecureForm' });
    }
  }, [onSubmit, csrfToken, checkRateLimit, getRemainingCooldown, requireCSRF]);

  return (
    <form onSubmit={handleSubmit}>
      {requireCSRF && (
        <input type="hidden" name="_csrf" value={csrfToken} />
      )}
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}
      
      <fieldset disabled={isBlocked}>
        {children}
      </fieldset>
    </form>
  );
};

// Secure input component with validation
interface SecureInputProps {
  type: 'email' | 'password' | 'text';
  name: string;
  placeholder?: string;
  required?: boolean;
  className?: string;
  maxLength?: number;
}

const SecureInput: React.FC<SecureInputProps> = ({
  type,
  name,
  placeholder,
  required = false,
  className = '',
  maxLength = 1000
}) => {
  const [value, setValue] = useState('');
  const [error, setError] = useState<string>('');
  
  const validateInput = useCallback((input: string) => {
    let schema;
    
    switch (type) {
      case 'email':
        schema = emailSchema;
        break;
      case 'password':
        schema = passwordSchema;
        break;
      default:
        schema = userInputSchema;
    }
    
    const result = schema.safeParse(input);
    if (!result.success) {
      setError(result.error.errors[0].message);
      return false;
    }
    
    setError('');
    return true;
  }, [type]);
  
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value;
    const sanitizedValue = sanitizeInput(rawValue);
    
    setValue(sanitizedValue);
    
    if (sanitizedValue !== rawValue) {
      setError('Invalid characters were removed from input');
    } else {
      validateInput(sanitizedValue);
    }
  }, [validateInput]);
  
  return (
    <div className="mb-4">
      <input
        type={type}
        name={name}
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        required={required}
        maxLength={maxLength}
        className={`w-full px-3 py-2 border rounded-md ${
          error ? 'border-red-500' : 'border-gray-300'
        } ${className}`}
        autoComplete={type === 'password' ? 'current-password' : 'on'}
      />
      {error && (
        <p className="mt-1 text-xs text-red-600">{error}</p>
      )}
    </div>
  );
};

// Security headers component for CSP
const SecurityHeaders = () => {
  React.useEffect(() => {
    // Set security headers via meta tags (backup - should be set server-side)
    const metaCSP = document.createElement('meta');
    metaCSP.httpEquiv = 'Content-Security-Policy';
    metaCSP.content = `
      default-src 'self';
      script-src 'self' 'unsafe-inline' 'unsafe-eval' https://js.stripe.com;
      style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
      font-src 'self' https://fonts.gstatic.com;
      img-src 'self' data: https:;
      connect-src 'self' https://*.supabase.co https://api.stripe.com;
      frame-src https://js.stripe.com;
    `.replace(/\s+/g, ' ').trim();
    
    document.head.appendChild(metaCSP);
    
    return () => {
      document.head.removeChild(metaCSP);
    };
  }, []);
  
  return null;
};

// Session timeout hook
const useSessionTimeout = (timeoutMs = 30 * 60 * 1000) => {
  const [isExpired, setIsExpired] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();
  
  const resetTimeout = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(() => {
      setIsExpired(true);
      // Clear session storage
      sessionStorage.clear();
      
      // Redirect to login
      window.location.href = '/login';
    }, timeoutMs);
  }, [timeoutMs]);
  
  React.useEffect(() => {
    // Track user activity
    const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
    
    const handleActivity = () => {
      if (!isExpired) {
        resetTimeout();
      }
    };
    
    events.forEach(event => {
      document.addEventListener(event, handleActivity, true);
    });
    
    resetTimeout(); // Start timeout
    
    return () => {
      events.forEach(event => {
        document.removeEventListener(event, handleActivity, true);
      });
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [resetTimeout, isExpired]);
  
  return { isExpired, resetTimeout };
};

// Security monitor component
const SecurityMonitor = () => {
  const [alerts, setAlerts] = useState<string[]>([]);
  
  React.useEffect(() => {
    // Monitor for security events
    const checkSecurity = () => {
      const newAlerts: string[] = [];
      
      // Check for XSS attempts
      if (document.querySelectorAll('script[src*="evil"]').length > 0) {
        newAlerts.push('Suspicious script detected');
      }
      
      // Check for iframe injection
      if (document.querySelectorAll('iframe[src*="javascript:"]').length > 0) {
        newAlerts.push('Suspicious iframe detected');
      }
      
      // Check for form tampering
      const forms = document.querySelectorAll('form');
      forms.forEach(form => {
        if (form.action && !form.action.startsWith(window.location.origin)) {
          newAlerts.push('Form action points to external domain');
        }
      });
      
      setAlerts(newAlerts);
    };
    
    // Check immediately and then every 10 seconds
    checkSecurity();
    const interval = setInterval(checkSecurity, 10000);
    
    return () => clearInterval(interval);
  }, []);
  
  if (alerts.length === 0) return null;
  
  return (
    <div className="fixed top-4 right-4 bg-red-600 text-white p-4 rounded-lg shadow-lg z-50">
      <h4 className="font-bold">Security Alert</h4>
      <ul className="mt-2">
        {alerts.map((alert, index) => (
          <li key={index} className="text-sm">{alert}</li>
        ))}
      </ul>
    </div>
  );
};

export {
  SecureForm,
  SecureInput,
  SecurityHeaders,
  useCSRFToken,
  useRateLimit,
  useSessionTimeout,
  SecurityMonitor,
  sanitizeInput,
  emailSchema,
  passwordSchema,
  userInputSchema
};