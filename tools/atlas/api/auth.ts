/**
 * ATLAS Authentication & Authorization Module
 * JWT authentication with Role-Based Access Control (RBAC)
 */

import * as crypto from 'crypto';

// ============================================================================
// Types
// ============================================================================

export type Role = 'admin' | 'operator' | 'user' | 'readonly';

export interface Permission {
  resource: string;
  actions: ('read' | 'write' | 'execute' | 'admin')[];
}

export interface User {
  id: string;
  username: string;
  role: Role;
  permissions?: Permission[];
  createdAt: string;
  lastLogin?: string;
}

export interface JWTPayload {
  sub: string; // User ID
  username: string;
  role: Role;
  permissions?: Permission[];
  iat: number; // Issued at
  exp: number; // Expiration
  iss: string; // Issuer
}

export interface AuthConfig {
  jwtSecret: string;
  jwtExpiry: number; // seconds
  apiKey?: string;
  issuer: string;
}

export interface AuthResult {
  valid: boolean;
  user?: User;
  error?: string;
  method?: 'jwt' | 'api-key';
}

// ============================================================================
// Default Role Permissions
// ============================================================================

export const ROLE_PERMISSIONS: Record<Role, Permission[]> = {
  admin: [{ resource: '*', actions: ['read', 'write', 'execute', 'admin'] }],
  operator: [
    { resource: 'agents', actions: ['read', 'write', 'execute'] },
    { resource: 'tasks', actions: ['read', 'write', 'execute'] },
    { resource: 'health', actions: ['read'] },
    { resource: 'metrics', actions: ['read'] },
    { resource: 'users', actions: ['read'] },
  ],
  user: [
    { resource: 'agents', actions: ['read', 'execute'] },
    { resource: 'tasks', actions: ['read', 'write', 'execute'] },
    { resource: 'health', actions: ['read'] },
  ],
  readonly: [
    { resource: 'agents', actions: ['read'] },
    { resource: 'tasks', actions: ['read'] },
    { resource: 'health', actions: ['read'] },
  ],
};

// ============================================================================
// JWT Implementation (without external dependencies)
// ============================================================================

function base64UrlEncode(data: string | Buffer): string {
  const base64 = Buffer.from(data).toString('base64');
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function base64UrlDecode(data: string): string {
  const base64 = data.replace(/-/g, '+').replace(/_/g, '/');
  const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4);
  return Buffer.from(padded, 'base64').toString('utf8');
}

function createSignature(data: string, secret: string): string {
  const hmac = crypto.createHmac('sha256', secret);
  hmac.update(data);
  return base64UrlEncode(hmac.digest());
}

/**
 * Create a JWT token
 */
export function createJWT(
  payload: Omit<JWTPayload, 'iat' | 'exp' | 'iss'>,
  config: AuthConfig
): string {
  const header = {
    alg: 'HS256',
    typ: 'JWT',
  };

  const now = Math.floor(Date.now() / 1000);
  const fullPayload: JWTPayload = {
    ...payload,
    iat: now,
    exp: now + config.jwtExpiry,
    iss: config.issuer,
  };

  const headerEncoded = base64UrlEncode(JSON.stringify(header));
  const payloadEncoded = base64UrlEncode(JSON.stringify(fullPayload));
  const signature = createSignature(`${headerEncoded}.${payloadEncoded}`, config.jwtSecret);

  return `${headerEncoded}.${payloadEncoded}.${signature}`;
}

/**
 * Verify and decode a JWT token
 */
export function verifyJWT(
  token: string,
  config: AuthConfig
): { valid: boolean; payload?: JWTPayload; error?: string } {
  const parts = token.split('.');
  if (parts.length !== 3) {
    return { valid: false, error: 'Invalid token format' };
  }

  const [headerEncoded, payloadEncoded, signature] = parts;

  // Verify signature
  const expectedSignature = createSignature(`${headerEncoded}.${payloadEncoded}`, config.jwtSecret);
  if (signature !== expectedSignature) {
    return { valid: false, error: 'Invalid signature' };
  }

  // Decode payload
  let payload: JWTPayload;
  try {
    payload = JSON.parse(base64UrlDecode(payloadEncoded));
  } catch {
    return { valid: false, error: 'Invalid payload' };
  }

  // Check expiration
  const now = Math.floor(Date.now() / 1000);
  if (payload.exp && payload.exp < now) {
    return { valid: false, error: 'Token expired' };
  }

  // Check issuer
  if (payload.iss !== config.issuer) {
    return { valid: false, error: 'Invalid issuer' };
  }

  return { valid: true, payload };
}

// ============================================================================
// Authentication
// ============================================================================

/**
 * Authenticate a request using JWT or API key
 */
export function authenticate(
  headers: Record<string, string | string[] | undefined>,
  config: AuthConfig
): AuthResult {
  const authHeader = headers['authorization'] as string | undefined;
  const apiKeyHeader = headers['x-api-key'] as string | undefined;

  // Try JWT authentication first
  if (authHeader?.startsWith('Bearer ')) {
    const token = authHeader.slice(7);

    // Check if it's an API key (for backward compatibility)
    if (config.apiKey && token === config.apiKey) {
      return {
        valid: true,
        method: 'api-key',
        user: {
          id: 'api-key-user',
          username: 'api-key',
          role: 'admin',
          createdAt: new Date().toISOString(),
        },
      };
    }

    // Try JWT verification
    const result = verifyJWT(token, config);
    if (result.valid && result.payload) {
      return {
        valid: true,
        method: 'jwt',
        user: {
          id: result.payload.sub,
          username: result.payload.username,
          role: result.payload.role,
          permissions: result.payload.permissions,
          createdAt: new Date().toISOString(),
        },
      };
    }

    return { valid: false, error: result.error || 'Invalid token' };
  }

  // Try X-API-Key header
  if (apiKeyHeader && config.apiKey && apiKeyHeader === config.apiKey) {
    return {
      valid: true,
      method: 'api-key',
      user: {
        id: 'api-key-user',
        username: 'api-key',
        role: 'admin',
        createdAt: new Date().toISOString(),
      },
    };
  }

  // No authentication provided
  if (!config.apiKey && !config.jwtSecret) {
    return { valid: true }; // No auth required
  }

  return { valid: false, error: 'Authentication required' };
}

// ============================================================================
// Authorization (RBAC)
// ============================================================================

/**
 * Check if a user has permission to perform an action on a resource
 */
export function authorize(
  user: User | undefined,
  resource: string,
  action: 'read' | 'write' | 'execute' | 'admin'
): boolean {
  if (!user) {
    return false;
  }

  // Get permissions from user or role defaults
  const permissions = user.permissions || ROLE_PERMISSIONS[user.role] || [];

  for (const perm of permissions) {
    // Wildcard resource matches all
    if (perm.resource === '*' && perm.actions.includes(action)) {
      return true;
    }

    // Exact resource match
    if (perm.resource === resource && perm.actions.includes(action)) {
      return true;
    }

    // Resource prefix match (e.g., 'agents' matches 'agents/claude')
    if (resource.startsWith(perm.resource + '/') && perm.actions.includes(action)) {
      return true;
    }
  }

  return false;
}

/**
 * Map API paths to resources for authorization
 */
export function pathToResource(path: string): string {
  // Remove leading slash and extract base resource
  const normalized = path.replace(/^\//, '');
  const parts = normalized.split('/');

  // Map common paths
  const resourceMap: Record<string, string> = {
    health: 'health',
    status: 'health',
    agents: 'agents',
    execute: 'tasks',
    generate: 'tasks',
    review: 'tasks',
    explain: 'tasks',
    chat: 'tasks',
    users: 'users',
    auth: 'auth',
    metrics: 'metrics',
  };

  return resourceMap[parts[0]] || parts[0];
}

/**
 * Map HTTP methods to actions
 */
export function methodToAction(method: string): 'read' | 'write' | 'execute' | 'admin' {
  switch (method.toUpperCase()) {
    case 'GET':
    case 'HEAD':
    case 'OPTIONS':
      return 'read';
    case 'POST':
      return 'execute';
    case 'PUT':
    case 'PATCH':
      return 'write';
    case 'DELETE':
      return 'admin';
    default:
      return 'read';
  }
}

// ============================================================================
// User Management (In-Memory Store)
// ============================================================================

const userStore = new Map<string, User & { passwordHash: string }>();

/**
 * Hash a password using SHA-256
 */
function hashPassword(password: string, salt: string): string {
  return crypto
    .createHash('sha256')
    .update(password + salt)
    .digest('hex');
}

/**
 * Create a new user
 */
export function createUser(username: string, password: string, role: Role = 'user'): User {
  const id = crypto.randomUUID();
  const salt = crypto.randomBytes(16).toString('hex');
  const passwordHash = hashPassword(password, salt);

  const user: User & { passwordHash: string } = {
    id,
    username,
    role,
    passwordHash: `${salt}:${passwordHash}`,
    createdAt: new Date().toISOString(),
  };

  userStore.set(id, user);
  userStore.set(username, user);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { passwordHash: _, ...publicUser } = user;
  return publicUser;
}

/**
 * Verify user credentials
 */
export function verifyCredentials(username: string, password: string): User | null {
  const user = userStore.get(username);
  if (!user) {
    return null;
  }

  const [salt, storedHash] = user.passwordHash.split(':');
  const hash = hashPassword(password, salt);

  if (hash !== storedHash) {
    return null;
  }

  // Update last login
  user.lastLogin = new Date().toISOString();

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { passwordHash: _, ...publicUser } = user;
  return publicUser;
}

/**
 * Get user by ID
 */
export function getUser(id: string): User | null {
  const user = userStore.get(id);
  if (!user) {
    return null;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { passwordHash: _, ...publicUser } = user;
  return publicUser;
}

/**
 * List all users
 */
export function listUsers(): User[] {
  const users: User[] = [];
  const seen = new Set<string>();

  userStore.forEach((user) => {
    if (!seen.has(user.id)) {
      seen.add(user.id);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { passwordHash: _, ...publicUser } = user;
      users.push(publicUser);
    }
  });

  return users;
}

/**
 * Delete a user
 */
export function deleteUser(id: string): boolean {
  const user = userStore.get(id);
  if (!user) {
    return false;
  }

  userStore.delete(id);
  userStore.delete(user.username);
  return true;
}

// ============================================================================
// Auth Configuration Helper
// ============================================================================

/**
 * Get auth configuration from environment
 */
export function getAuthConfig(): AuthConfig {
  return {
    jwtSecret: process.env.ATLAS_JWT_SECRET || crypto.randomBytes(32).toString('hex'),
    jwtExpiry: parseInt(process.env.ATLAS_JWT_EXPIRY || '86400', 10), // 24 hours
    apiKey: process.env.ATLAS_API_KEY,
    issuer: process.env.ATLAS_JWT_ISSUER || 'atlas-api',
  };
}

// ============================================================================
// Audit Logging
// ============================================================================

export interface AuditEntry {
  timestamp: string;
  userId?: string;
  username?: string;
  action: string;
  resource: string;
  method: string;
  path: string;
  ip?: string;
  success: boolean;
  error?: string;
}

const auditLog: AuditEntry[] = [];

/**
 * Log an authentication/authorization event
 */
export function logAuditEntry(entry: Omit<AuditEntry, 'timestamp'>): void {
  const fullEntry: AuditEntry = {
    ...entry,
    timestamp: new Date().toISOString(),
  };

  auditLog.push(fullEntry);

  // Keep only last 1000 entries in memory
  if (auditLog.length > 1000) {
    auditLog.shift();
  }

  // Log to console in development
  if (process.env.NODE_ENV !== 'production') {
    const status = entry.success ? '✓' : '✗';
    console.log(
      `[AUDIT] ${status} ${entry.method} ${entry.path} - ${entry.username || 'anonymous'}`
    );
  }
}

/**
 * Get recent audit entries
 */
export function getAuditLog(limit: number = 100): AuditEntry[] {
  return auditLog.slice(-limit);
}
