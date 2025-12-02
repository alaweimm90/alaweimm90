---
name: 'Security & Cybersecurity Superprompt'
version: '1.0'
category: 'project'
tags: ['security', 'cybersecurity', 'appsec', 'devsecops', 'vulnerability']
created: '2024-11-30'
---

# Security & Cybersecurity Superprompt

## Purpose

Comprehensive framework for application security, infrastructure security, DevSecOps practices, and cybersecurity implementation across all projects.

---

## System Prompt

```text
You are a Senior Security Engineer and Cybersecurity Architect with expertise in:
- Application Security (OWASP Top 10, SANS Top 25)
- Infrastructure Security (Zero Trust, Defense in Depth)
- DevSecOps and Security Automation
- Threat Modeling and Risk Assessment
- Penetration Testing and Vulnerability Management
- Incident Response and Security Operations

Your mission is to implement security that:
1. Prevents vulnerabilities before deployment
2. Detects and responds to threats in real-time
3. Maintains compliance with security standards
4. Enables secure development practices
5. Protects data and infrastructure assets
```

---

## Security Architecture

### Defense in Depth Model

```yaml
security_layers:
  perimeter:
    controls:
      - Web Application Firewall (WAF)
      - DDoS Protection
      - Rate Limiting
      - IP Reputation Filtering
      - Bot Detection

  network:
    controls:
      - Network Segmentation
      - Micro-segmentation
      - VPN/Zero Trust Network Access
      - Intrusion Detection/Prevention
      - Network Access Control

  application:
    controls:
      - Input Validation
      - Output Encoding
      - Authentication/Authorization
      - Session Management
      - API Security

  data:
    controls:
      - Encryption at Rest
      - Encryption in Transit
      - Data Masking
      - Access Controls
      - Data Loss Prevention

  endpoint:
    controls:
      - Endpoint Detection and Response
      - Anti-malware
      - Device Management
      - Patch Management
      - Application Whitelisting
```

---

## OWASP Top 10 Mitigations

### Security Implementation Guide

```typescript
// security/middleware/security.ts
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { Express } from 'express';

export function configureSecurityMiddleware(app: Express) {
  // A01:2021 - Broken Access Control
  // Implement proper authorization checks

  // A02:2021 - Cryptographic Failures
  app.use(
    helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'strict-dynamic'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          imgSrc: ["'self'", 'data:', 'https:'],
          connectSrc: ["'self'"],
          fontSrc: ["'self'"],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          frameSrc: ["'none'"],
          upgradeInsecureRequests: [],
        },
      },
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true,
      },
      referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
      noSniff: true,
      xssFilter: true,
      frameguard: { action: 'deny' },
    })
  );

  // A03:2021 - Injection
  // Use parameterized queries (see database section)

  // A04:2021 - Insecure Design
  // Implement threat modeling (see threat model section)

  // A05:2021 - Security Misconfiguration
  app.disable('x-powered-by');

  // A06:2021 - Vulnerable Components
  // Use dependency scanning (see CI/CD section)

  // A07:2021 - Authentication Failures
  const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 attempts
    message: 'Too many login attempts, please try again later',
    standardHeaders: true,
    legacyHeaders: false,
  });
  app.use('/api/auth/login', authLimiter);

  // A08:2021 - Software and Data Integrity Failures
  // Implement SRI and signed artifacts

  // A09:2021 - Security Logging and Monitoring Failures
  // See logging section

  // A10:2021 - Server-Side Request Forgery
  // Validate and sanitize URLs
}
```

### Input Validation

```typescript
// security/validation/input.ts
import { z } from 'zod';
import DOMPurify from 'isomorphic-dompurify';

// User input schema with strict validation
export const userInputSchema = z.object({
  email: z
    .string()
    .email('Invalid email format')
    .max(255, 'Email too long')
    .transform((val) => val.toLowerCase().trim()),

  password: z
    .string()
    .min(12, 'Password must be at least 12 characters')
    .max(128, 'Password too long')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
      'Password must contain uppercase, lowercase, number, and special character'
    ),

  name: z
    .string()
    .min(1, 'Name is required')
    .max(100, 'Name too long')
    .transform((val) => DOMPurify.sanitize(val.trim())),

  bio: z
    .string()
    .max(1000, 'Bio too long')
    .optional()
    .transform((val) => (val ? DOMPurify.sanitize(val) : undefined)),
});

// SQL Injection prevention - use parameterized queries
export async function findUserByEmail(email: string) {
  // NEVER do this:
  // const query = `SELECT * FROM users WHERE email = '${email}'`;

  // ALWAYS use parameterized queries:
  const result = await db.query('SELECT * FROM users WHERE email = $1', [email]);
  return result.rows[0];
}

// XSS Prevention
export function sanitizeHtml(input: string): string {
  return DOMPurify.sanitize(input, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href', 'target', 'rel'],
  });
}

// Path Traversal Prevention
export function sanitizePath(userPath: string, baseDir: string): string {
  const path = require('path');
  const resolvedPath = path.resolve(baseDir, userPath);

  if (!resolvedPath.startsWith(baseDir)) {
    throw new Error('Path traversal attempt detected');
  }

  return resolvedPath;
}
```

### Authentication Security

```typescript
// security/auth/authentication.ts
import bcrypt from 'bcrypt';
import { randomBytes, createHash } from 'crypto';
import jwt from 'jsonwebtoken';

const SALT_ROUNDS = 12;
const TOKEN_EXPIRY = '15m';
const REFRESH_TOKEN_EXPIRY = '7d';

// Password hashing
export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, SALT_ROUNDS);
}

export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

// Secure token generation
export function generateSecureToken(length: number = 32): string {
  return randomBytes(length).toString('hex');
}

// JWT with proper configuration
export function generateAccessToken(userId: string, roles: string[]): string {
  return jwt.sign(
    {
      sub: userId,
      roles,
      type: 'access',
    },
    process.env.JWT_SECRET!,
    {
      expiresIn: TOKEN_EXPIRY,
      algorithm: 'RS256',
      issuer: 'your-app',
      audience: 'your-app-users',
    }
  );
}

export function generateRefreshToken(userId: string): string {
  const token = generateSecureToken(64);
  const hashedToken = createHash('sha256').update(token).digest('hex');

  // Store hashed token in database
  // await db.refreshTokens.create({ userId, token: hashedToken, expiresAt });

  return token;
}

// Session management
export const sessionConfig = {
  name: '__Host-session', // Secure cookie prefix
  secret: process.env.SESSION_SECRET!,
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: true, // HTTPS only
    httpOnly: true, // No JavaScript access
    sameSite: 'strict' as const,
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    domain: process.env.COOKIE_DOMAIN,
    path: '/',
  },
  rolling: true, // Reset expiry on activity
};

// MFA Implementation
export async function generateTOTP(secret: string): Promise<string> {
  const { authenticator } = await import('otplib');
  return authenticator.generate(secret);
}

export async function verifyTOTP(token: string, secret: string): Promise<boolean> {
  const { authenticator } = await import('otplib');
  return authenticator.verify({ token, secret });
}
```

---

## Security Scanning Pipeline

### DevSecOps Workflow

```yaml
# .github/workflows/security.yml
name: Security Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 0 * * *' # Daily scan

jobs:
  # Static Application Security Testing
  sast:
    name: SAST
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: javascript, typescript
          queries: security-extended

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

      - name: Semgrep Scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/owasp-top-ten
            p/typescript

      - name: SonarCloud Scan
        uses: SonarSource/sonarcloud-github-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  # Software Composition Analysis
  sca:
    name: SCA
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Snyk Security Scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high --fail-on=all

      - name: OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: '${{ github.repository }}'
          path: '.'
          format: 'ALL'
          args: >-
            --failOnCVSS 7
            --enableRetired

      - name: License Compliance
        run: |
          npx license-checker --production --failOn "GPL;AGPL;LGPL"

  # Secret Detection
  secrets:
    name: Secret Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD

  # Container Security
  container:
    name: Container Security
    runs-on: ubuntu-latest
    if: contains(github.event.pull_request.changed_files, 'Dockerfile')
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t test-image .

      - name: Trivy Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'test-image'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Dockle Lint
        uses: erzz/dockle-action@v1
        with:
          image: 'test-image'
          failure-threshold: high

  # Infrastructure Security
  iac:
    name: IaC Security
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Checkov Scan
        uses: bridgecrewio/checkov-action@master
        with:
          directory: infrastructure/
          framework: terraform

      - name: tfsec Scan
        uses: aquasecurity/tfsec-action@v1.0.0
        with:
          working_directory: infrastructure/

  # Dynamic Application Security Testing
  dast:
    name: DAST
    runs-on: ubuntu-latest
    needs: [sast, sca]
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4

      - name: ZAP Scan
        uses: zaproxy/action-full-scan@v0.7.0
        with:
          target: 'https://staging.example.com'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'
```

---

## Threat Modeling

### STRIDE Analysis Template

```yaml
threat_model:
  system: 'Web Application'
  version: '1.0'
  date: '2024-11-30'

  assets:
    - name: 'User Data'
      sensitivity: 'High'
      description: 'PII including email, name, preferences'

    - name: 'Authentication Tokens'
      sensitivity: 'Critical'
      description: 'JWT tokens and session data'

    - name: 'API Keys'
      sensitivity: 'Critical'
      description: 'Third-party service credentials'

  threats:
    spoofing:
      - threat: 'Session hijacking'
        asset: 'Authentication Tokens'
        likelihood: 'Medium'
        impact: 'High'
        mitigations:
          - 'Secure cookie flags (HttpOnly, Secure, SameSite)'
          - 'Token rotation on privilege change'
          - 'IP binding for sensitive operations'

      - threat: 'Credential stuffing'
        asset: 'User Data'
        likelihood: 'High'
        impact: 'High'
        mitigations:
          - 'Rate limiting on auth endpoints'
          - 'CAPTCHA after failed attempts'
          - 'Breach password checking'
          - 'MFA enforcement'

    tampering:
      - threat: 'Parameter manipulation'
        asset: 'API Endpoints'
        likelihood: 'Medium'
        impact: 'Medium'
        mitigations:
          - 'Input validation'
          - 'Authorization checks'
          - 'Request signing'

      - threat: 'SQL injection'
        asset: 'Database'
        likelihood: 'Medium'
        impact: 'Critical'
        mitigations:
          - 'Parameterized queries'
          - 'ORM usage'
          - 'Input sanitization'
          - 'WAF rules'

    repudiation:
      - threat: 'Action denial'
        asset: 'Audit Logs'
        likelihood: 'Low'
        impact: 'Medium'
        mitigations:
          - 'Comprehensive logging'
          - 'Log integrity protection'
          - 'Timestamp verification'

    information_disclosure:
      - threat: 'Data breach'
        asset: 'User Data'
        likelihood: 'Medium'
        impact: 'Critical'
        mitigations:
          - 'Encryption at rest'
          - 'Encryption in transit'
          - 'Access controls'
          - 'Data minimization'

    denial_of_service:
      - threat: 'Resource exhaustion'
        asset: 'Application'
        likelihood: 'High'
        impact: 'High'
        mitigations:
          - 'Rate limiting'
          - 'Auto-scaling'
          - 'CDN/DDoS protection'
          - 'Resource quotas'

    elevation_of_privilege:
      - threat: 'Privilege escalation'
        asset: 'Authorization System'
        likelihood: 'Low'
        impact: 'Critical'
        mitigations:
          - 'Principle of least privilege'
          - 'Role-based access control'
          - 'Regular access reviews'
          - 'Separation of duties'
```

---

## Security Monitoring

### Logging Configuration

```typescript
// security/logging/security-logger.ts
import winston from 'winston';
import { v4 as uuidv4 } from 'uuid';

const securityLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  defaultMeta: { service: 'security' },
  transports: [
    new winston.transports.File({ filename: 'logs/security.log' }),
    new winston.transports.Console(),
  ],
});

interface SecurityEvent {
  eventType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  userId?: string;
  ipAddress?: string;
  userAgent?: string;
  resource?: string;
  action?: string;
  outcome: 'success' | 'failure';
  details?: Record<string, unknown>;
}

export function logSecurityEvent(event: SecurityEvent) {
  securityLogger.log({
    level: event.severity === 'critical' ? 'error' : 'info',
    eventId: uuidv4(),
    timestamp: new Date().toISOString(),
    ...event,
  });
}

// Authentication events
export function logAuthEvent(
  type: 'login' | 'logout' | 'mfa' | 'password_reset',
  userId: string,
  outcome: 'success' | 'failure',
  ipAddress: string,
  details?: Record<string, unknown>
) {
  logSecurityEvent({
    eventType: `auth.${type}`,
    severity: outcome === 'failure' ? 'medium' : 'low',
    userId,
    ipAddress,
    action: type,
    outcome,
    details,
  });
}

// Authorization events
export function logAuthzEvent(
  userId: string,
  resource: string,
  action: string,
  outcome: 'success' | 'failure',
  ipAddress: string
) {
  logSecurityEvent({
    eventType: 'authorization',
    severity: outcome === 'failure' ? 'high' : 'low',
    userId,
    ipAddress,
    resource,
    action,
    outcome,
  });
}

// Suspicious activity
export function logSuspiciousActivity(
  type: string,
  details: Record<string, unknown>,
  ipAddress: string,
  userId?: string
) {
  logSecurityEvent({
    eventType: `suspicious.${type}`,
    severity: 'high',
    userId,
    ipAddress,
    outcome: 'failure',
    details,
  });
}
```

---

## Incident Response

```yaml
incident_response:
  severity_levels:
    critical:
      description: 'Active breach, data exfiltration, system compromise'
      response_time: '15 minutes'
      escalation: 'Immediate to CISO and executive team'

    high:
      description: 'Vulnerability exploitation, unauthorized access attempt'
      response_time: '1 hour'
      escalation: 'Security team lead'

    medium:
      description: 'Policy violation, suspicious activity'
      response_time: '4 hours'
      escalation: 'Security team'

    low:
      description: 'Minor security event, informational'
      response_time: '24 hours'
      escalation: 'Logged for review'

  response_phases:
    1_detection:
      - Identify the incident
      - Classify severity
      - Document initial findings

    2_containment:
      - Isolate affected systems
      - Block malicious IPs/users
      - Preserve evidence

    3_eradication:
      - Remove threat
      - Patch vulnerabilities
      - Reset compromised credentials

    4_recovery:
      - Restore systems
      - Verify integrity
      - Monitor for recurrence

    5_lessons_learned:
      - Post-incident review
      - Update procedures
      - Implement improvements
```

---

## Execution Phases

### Phase 1: Foundation

- [ ] Implement security headers
- [ ] Configure authentication
- [ ] Set up input validation
- [ ] Enable security logging

### Phase 2: Scanning

- [ ] Integrate SAST tools
- [ ] Add dependency scanning
- [ ] Configure secret detection
- [ ] Set up container scanning

### Phase 3: Monitoring

- [ ] Deploy security logging
- [ ] Configure alerting
- [ ] Set up SIEM integration
- [ ] Implement threat detection

### Phase 4: Response

- [ ] Create incident response plan
- [ ] Set up communication channels
- [ ] Document procedures
- [ ] Conduct tabletop exercises

---

_Last updated: 2024-11-30_
