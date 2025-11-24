// Security Configuration for E-commerce Application
// Implements enterprise security best practices

class SecurityConfig {
    constructor() {
        this.csrfToken = this.generateCSRFToken();
        this.sessionTimeout = 30 * 60 * 1000; // 30 minutes
        this.maxLoginAttempts = 5;
        this.rateLimitWindow = 15 * 60 * 1000; // 15 minutes
    }

    // CSRF Protection
    generateCSRFToken() {
        const array = new Uint8Array(32);
        crypto.getRandomValues(array);
        return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    }

    validateCSRFToken(token) {
        return token === this.csrfToken && token.length === 64;
    }

    // Input Sanitization
    static sanitizeHTML(input) {
        const div = document.createElement('div');
        div.textContent = input;
        return div.innerHTML;
    }

    static sanitizeURL(url) {
        try {
            const parsed = new URL(url);
            // Only allow http/https protocols
            if (!['http:', 'https:'].includes(parsed.protocol)) {
                throw new Error('Invalid protocol');
            }
            return parsed.toString();
        } catch {
            return '';
        }
    }

    static validateInput(input, type, maxLength = 255) {
        if (typeof input !== 'string') return false;
        if (input.length > maxLength) return false;

        const patterns = {
            email: /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/,
            phone: /^\+?[\d\s\-\(\)]{10,15}$/,
            alphanumeric: /^[a-zA-Z0-9\s]+$/,
            numeric: /^\d+$/,
            currency: /^\d+(\.\d{2})?$/
        };

        return patterns[type] ? patterns[type].test(input) : true;
    }

    // Rate Limiting
    static createRateLimiter(maxRequests = 100, windowMs = 15 * 60 * 1000) {
        const requests = new Map();

        return function(identifier) {
            const now = Date.now();
            const windowStart = now - windowMs;

            // Clean old entries
            for (const [key, timestamps] of requests.entries()) {
                requests.set(key, timestamps.filter(time => time > windowStart));
                if (requests.get(key).length === 0) {
                    requests.delete(key);
                }
            }

            // Check current requests
            const userRequests = requests.get(identifier) || [];
            if (userRequests.length >= maxRequests) {
                return false; // Rate limit exceeded
            }

            // Add current request
            userRequests.push(now);
            requests.set(identifier, userRequests);
            return true;
        };
    }

    // Content Security Policy
    static getCSPHeader() {
        return {
            'Content-Security-Policy': [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline'", // Note: Remove unsafe-inline in production
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: https:",
                "font-src 'self'",
                "connect-src 'self'",
                "frame-ancestors 'none'",
                "base-uri 'self'",
                "form-action 'self'"
            ].join('; ')
        };
    }

    // Security Headers
    static getSecurityHeaders() {
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            ...this.getCSPHeader()
        };
    }

    // Session Management
    static createSecureSession() {
        return {
            id: crypto.getRandomValues(new Uint8Array(32)).join(''),
            created: Date.now(),
            lastAccess: Date.now(),
            data: {}
        };
    }

    static validateSession(session, maxAge = 30 * 60 * 1000) {
        if (!session || !session.id || !session.created) return false;
        
        const now = Date.now();
        const age = now - session.created;
        const idle = now - session.lastAccess;

        return age < maxAge && idle < maxAge;
    }

    // Password Security
    static validatePasswordStrength(password) {
        const checks = {
            length: password.length >= 8,
            uppercase: /[A-Z]/.test(password),
            lowercase: /[a-z]/.test(password),
            numbers: /\d/.test(password),
            special: /[!@#$%^&*(),.?":{}|<>]/.test(password),
            noCommon: !this.isCommonPassword(password)
        };

        const score = Object.values(checks).filter(Boolean).length;
        return {
            score,
            checks,
            isStrong: score >= 5
        };
    }

    static isCommonPassword(password) {
        const common = [
            'password', '123456', '123456789', 'qwerty', 'abc123',
            'password123', 'admin', 'letmein', 'welcome', 'monkey'
        ];
        return common.includes(password.toLowerCase());
    }

    // API Security
    static createAPIKey() {
        const array = new Uint8Array(32);
        crypto.getRandomValues(array);
        return 'ak_' + Array.from(array, byte => 
            byte.toString(16).padStart(2, '0')
        ).join('');
    }

    static validateAPIKey(key) {
        return typeof key === 'string' && 
               key.startsWith('ak_') && 
               key.length === 67;
    }

    // Audit Logging
    static logSecurityEvent(event, details = {}) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            event,
            details,
            userAgent: navigator.userAgent,
            ip: 'client-side', // Would be populated server-side
            sessionId: sessionStorage.getItem('sessionId')
        };

        // In production, send to security monitoring system
        console.log('Security Event:', logEntry);
        
        // Store locally for debugging
        const logs = JSON.parse(localStorage.getItem('securityLogs') || '[]');
        logs.push(logEntry);
        
        // Keep only last 100 entries
        if (logs.length > 100) {
            logs.splice(0, logs.length - 100);
        }
        
        localStorage.setItem('securityLogs', JSON.stringify(logs));
    }

    // Encryption Utilities (for client-side data)
    static async encryptData(data, password) {
        const encoder = new TextEncoder();
        const dataBuffer = encoder.encode(JSON.stringify(data));
        
        const passwordBuffer = encoder.encode(password);
        const key = await crypto.subtle.importKey(
            'raw',
            passwordBuffer,
            { name: 'PBKDF2' },
            false,
            ['deriveKey']
        );

        const derivedKey = await crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: encoder.encode('salt'), // Use random salt in production
                iterations: 100000,
                hash: 'SHA-256'
            },
            key,
            { name: 'AES-GCM', length: 256 },
            false,
            ['encrypt']
        );

        const iv = crypto.getRandomValues(new Uint8Array(12));
        const encrypted = await crypto.subtle.encrypt(
            { name: 'AES-GCM', iv },
            derivedKey,
            dataBuffer
        );

        return {
            encrypted: Array.from(new Uint8Array(encrypted)),
            iv: Array.from(iv)
        };
    }
}

// Initialize security configuration
const security = new SecurityConfig();

// Add CSRF token to all forms
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const csrfInput = document.createElement('input');
        csrfInput.type = 'hidden';
        csrfInput.name = 'csrf_token';
        csrfInput.value = security.csrfToken;
        form.appendChild(csrfInput);
    });

    // Log page load
    SecurityConfig.logSecurityEvent('page_load', {
        url: window.location.href,
        referrer: document.referrer
    });
});

// Rate limiter for API calls
const apiRateLimiter = SecurityConfig.createRateLimiter(50, 15 * 60 * 1000);

// Export for use in other scripts
window.SecurityConfig = SecurityConfig;
window.security = security;