import { supabase } from '@/integrations/supabase/client';
import { AuthError, AuthResponse } from '@supabase/supabase-js';
import { SecurityValidator, rateLimiters, SecurityAudit, SecureErrorHandler } from '@/lib/security';

interface SecureAuthResult {
  success: boolean;
  data?: unknown;
  error?: string;
  rateLimited?: boolean;
  shouldRetry?: boolean;
  retryAfter?: number;
}

/**
 * Security wrapper for authentication operations
 * Implements rate limiting, input validation, and security logging
 */
export class SecureAuthWrapper {
  
  /**
   * Get client identifier for rate limiting
   */
  private static getClientIdentifier(): string {
    // Use a combination of IP-like identifier and session
    const userAgent = navigator.userAgent;
    const screen = `${window.screen.width}x${window.screen.height}`;
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    
    // Create a consistent but anonymous identifier
    const identifier = btoa(`${userAgent.slice(0, 50)}-${screen}-${timezone}`).slice(0, 32);
    return identifier;
  }

  /**
   * Secure sign up with comprehensive validation
   */
  static async secureSignUp(email: string, password: string): Promise<SecureAuthResult> {
    try {
      const clientId = this.getClientIdentifier();
      
      // Rate limiting check
      const rateLimitResult = rateLimiters.auth.isAllowed(clientId);
      if (!rateLimitResult.allowed) {
        SecurityAudit.logSecurityEvent({
          type: 'rate_limit_exceeded',
          details: `Sign up rate limit exceeded for client: ${clientId}`,
          severity: 'medium',
          metadata: { email: email.substring(0, 3) + '***' }
        });
        
        return {
          success: false,
          error: rateLimitResult.reason || 'Too many attempts. Please try again later.',
          rateLimited: true,
          retryAfter: rateLimitResult.resetTime ? Math.ceil((rateLimitResult.resetTime - Date.now()) / 1000) : 900
        };
      }

      // Validate email
      const emailValidation = SecurityValidator.validateEmail(email);
      if (!emailValidation.isValid) {
        return {
          success: false,
          error: emailValidation.error || 'Invalid email format'
        };
      }

      // Validate password
      const passwordValidation = SecurityValidator.validatePassword(password);
      if (!passwordValidation.isValid) {
        return {
          success: false,
          error: passwordValidation.error || 'Password does not meet security requirements'
        };
      }

      // Sanitize email
      const sanitizedEmail = email.toLowerCase().trim();

      // Attempt registration with Supabase
      const { data, error }: AuthResponse = await supabase.auth.signUp({
        email: sanitizedEmail,
        password,
        options: {
          emailRedirectTo: `${window.location.origin}/`,
          data: {
            email_confirmed: false,
            created_at: new Date().toISOString()
          }
        }
      });

      if (error) {
        SecurityAudit.logSecurityEvent({
          type: 'auth_attempt',
          details: `Registration failed: ${error.message}`,
          severity: 'low',
          metadata: { 
            email: sanitizedEmail.substring(0, 3) + '***',
            errorCode: error.status 
          }
        });

        const { message } = SecureErrorHandler.handleError(error, 'registration');
        return {
          success: false,
          error: message
        };
      }

      // Success - clear rate limit and log
      rateLimiters.auth.clear(clientId);
      
      SecurityAudit.logSecurityEvent({
        type: 'auth_attempt',
        details: `Successful registration`,
        severity: 'low',
        userId: data.user?.id,
        metadata: { email: sanitizedEmail.substring(0, 3) + '***' }
      });

      return {
        success: true,
        data
      };

    } catch (error) {
      const { message } = SecureErrorHandler.handleError(error, 'registration');
      return {
        success: false,
        error: message
      };
    }
  }

  /**
   * Secure sign in with rate limiting and validation
   */
  static async secureSignIn(email: string, password: string): Promise<SecureAuthResult> {
    try {
      const clientId = this.getClientIdentifier();
      
      // Rate limiting check
      const rateLimitResult = rateLimiters.auth.isAllowed(clientId);
      if (!rateLimitResult.allowed) {
        SecurityAudit.logSecurityEvent({
          type: 'rate_limit_exceeded',
          details: `Sign in rate limit exceeded for client: ${clientId}`,
          severity: 'high',
          metadata: { email: email.substring(0, 3) + '***' }
        });
        
        return {
          success: false,
          error: rateLimitResult.reason || 'Too many login attempts. Please try again later.',
          rateLimited: true,
          retryAfter: rateLimitResult.resetTime ? Math.ceil((rateLimitResult.resetTime - Date.now()) / 1000) : 900
        };
      }

      // Validate email format
      const emailValidation = SecurityValidator.validateEmail(email);
      if (!emailValidation.isValid) {
        return {
          success: false,
          error: 'Invalid email format'
        };
      }

      // Basic password validation (don't reveal strength requirements for existing accounts)
      if (!password || password.length < 1) {
        return {
          success: false,
          error: 'Password is required'
        };
      }

      const sanitizedEmail = email.toLowerCase().trim();

      // Attempt authentication
      const { data, error }: AuthResponse = await supabase.auth.signInWithPassword({
        email: sanitizedEmail,
        password
      });

      if (error) {
        SecurityAudit.logSecurityEvent({
          type: 'auth_attempt',
          details: `Login failed: ${error.message}`,
          severity: 'medium',
          metadata: { 
            email: sanitizedEmail.substring(0, 3) + '***',
            errorCode: error.status,
            clientId
          }
        });

        const { message } = SecureErrorHandler.handleError(error, 'login');
        return {
          success: false,
          error: message
        };
      }

      // Success - clear rate limit and log
      rateLimiters.auth.clear(clientId);
      
      SecurityAudit.logSecurityEvent({
        type: 'auth_attempt',
        details: `Successful login`,
        severity: 'low',
        userId: data.user?.id,
        metadata: { email: sanitizedEmail.substring(0, 3) + '***' }
      });

      return {
        success: true,
        data
      };

    } catch (error) {
      const { message } = SecureErrorHandler.handleError(error, 'login');
      return {
        success: false,
        error: message
      };
    }
  }

  /**
   * Secure password reset with rate limiting
   */
  static async securePasswordReset(email: string): Promise<SecureAuthResult> {
    try {
      const clientId = this.getClientIdentifier();
      
      // Rate limiting check
      const rateLimitResult = rateLimiters.passwordReset.isAllowed(clientId);
      if (!rateLimitResult.allowed) {
        SecurityAudit.logSecurityEvent({
          type: 'rate_limit_exceeded',
          details: `Password reset rate limit exceeded for client: ${clientId}`,
          severity: 'medium',
          metadata: { email: email.substring(0, 3) + '***' }
        });
        
        return {
          success: false,
          error: 'Too many password reset attempts. Please try again later.',
          rateLimited: true,
          retryAfter: rateLimitResult.resetTime ? Math.ceil((rateLimitResult.resetTime - Date.now()) / 1000) : 3600
        };
      }

      // Validate email
      const emailValidation = SecurityValidator.validateEmail(email);
      if (!emailValidation.isValid) {
        return {
          success: false,
          error: 'Invalid email format'
        };
      }

      const sanitizedEmail = email.toLowerCase().trim();

      // Attempt password reset
      const { error } = await supabase.auth.resetPasswordForEmail(sanitizedEmail, {
        redirectTo: `${window.location.origin}/auth?mode=reset`
      });

      if (error) {
        SecurityAudit.logSecurityEvent({
          type: 'auth_attempt',
          details: `Password reset failed: ${error.message}`,
          severity: 'low',
          metadata: { 
            email: sanitizedEmail.substring(0, 3) + '***',
            errorCode: error.status 
          }
        });

        const { message } = SecureErrorHandler.handleError(error, 'password-reset');
        return {
          success: false,
          error: message
        };
      }

      // Log successful password reset request
      SecurityAudit.logSecurityEvent({
        type: 'auth_attempt',
        details: `Password reset email sent`,
        severity: 'low',
        metadata: { email: sanitizedEmail.substring(0, 3) + '***' }
      });

      return {
        success: true
      };

    } catch (error) {
      const { message } = SecureErrorHandler.handleError(error, 'password-reset');
      return {
        success: false,
        error: message
      };
    }
  }

  /**
   * Secure sign out with cleanup
   */
  static async secureSignOut(): Promise<SecureAuthResult> {
    try {
      const { error } = await supabase.auth.signOut();

      if (error) {
        const { message } = SecureErrorHandler.handleError(error, 'logout');
        return {
          success: false,
          error: message
        };
      }

      // Clear any sensitive data from localStorage
      try {
        const keys = Object.keys(localStorage);
        keys.forEach(key => {
          if (key.includes('auth') || key.includes('session') || key.includes('token')) {
            localStorage.removeItem(key);
          }
        });
      } catch (e) {
        console.warn('Failed to clear localStorage:', e);
      }

      SecurityAudit.logSecurityEvent({
        type: 'auth_attempt',
        details: 'User signed out successfully',
        severity: 'low'
      });

      return {
        success: true
      };

    } catch (error) {
      const { message } = SecureErrorHandler.handleError(error, 'logout');
      return {
        success: false,
        error: message
      };
    }
  }
}