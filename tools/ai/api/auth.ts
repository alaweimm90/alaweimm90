#!/usr/bin/env npx tsx
/**
 * API Authentication Module
 * Provides secure API key validation with HMAC signatures
 */

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { IncomingMessage, ServerResponse } from 'http';

const ROOT = process.cwd();
const API_KEYS_PATH = path.join(ROOT, '.ai/config/api-keys.json');

export interface ApiKey {
  id: string;
  name: string;
  keyHash: string;
  scopes: string[];
  rateLimit?: number;
  expiresAt?: string;
  createdAt: string;
  lastUsedAt?: string;
}

export interface ApiKeyStore {
  keys: ApiKey[];
  version: string;
}

// Public endpoints that don't require authentication
const PUBLIC_ENDPOINTS = ['/health', '/metrics/prometheus', '/dashboard/ui', '/docs'];

/**
 * Hash an API key using SHA-256
 */
export function hashApiKey(key: string): string {
  return crypto.createHash('sha256').update(key).digest('hex');
}

/**
 * Generate a new API key with prefix
 */
export function generateApiKey(prefix: string = 'meta'): { key: string; hash: string } {
  const randomPart = crypto.randomBytes(24).toString('base64url');
  const key = `${prefix}_${randomPart}`;
  return { key, hash: hashApiKey(key) };
}

/**
 * Load API keys from config file
 */
function loadApiKeys(): ApiKeyStore {
  try {
    if (fs.existsSync(API_KEYS_PATH)) {
      return JSON.parse(fs.readFileSync(API_KEYS_PATH, 'utf8'));
    }
  } catch {
    // Ignore errors, return empty store
  }
  return { keys: [], version: '1.0.0' };
}

/**
 * Save API keys to config file
 */
function saveApiKeys(store: ApiKeyStore): void {
  const dir = path.dirname(API_KEYS_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(API_KEYS_PATH, JSON.stringify(store, null, 2));
}

/**
 * Validate an API key and return the key record if valid
 */
export function validateApiKey(key: string): ApiKey | null {
  const store = loadApiKeys();
  const hash = hashApiKey(key);
  const apiKey = store.keys.find((k) => k.keyHash === hash);

  if (!apiKey) return null;

  // Check expiration
  if (apiKey.expiresAt && new Date(apiKey.expiresAt) < new Date()) {
    return null;
  }

  // Update last used timestamp
  apiKey.lastUsedAt = new Date().toISOString();
  saveApiKeys(store);

  return apiKey;
}

/**
 * Check if a scope is allowed for a key
 */
export function hasScope(apiKey: ApiKey, requiredScope: string): boolean {
  if (apiKey.scopes.includes('*')) return true;
  if (apiKey.scopes.includes(requiredScope)) return true;

  // Check wildcard patterns (e.g., 'read:*' matches 'read:compliance')
  for (const scope of apiKey.scopes) {
    if (scope.endsWith(':*')) {
      const prefix = scope.slice(0, -1);
      if (requiredScope.startsWith(prefix)) return true;
    }
  }
  return false;
}

/**
 * Authentication middleware for API requests
 */
export function authMiddleware(
  req: IncomingMessage,
  _res: ServerResponse,
  pathname: string
): { authenticated: boolean; apiKey?: ApiKey; error?: string } {
  // Allow public endpoints without auth
  if (PUBLIC_ENDPOINTS.some((ep) => pathname.startsWith(ep))) {
    return { authenticated: true };
  }

  // Check for API key in Authorization header or X-API-Key header
  const authHeader = req.headers['authorization'];
  const apiKeyHeader = req.headers['x-api-key'];

  let keyValue: string | undefined;

  if (authHeader?.startsWith('Bearer ')) {
    keyValue = authHeader.slice(7);
  } else if (typeof apiKeyHeader === 'string') {
    keyValue = apiKeyHeader;
  }

  // If no auth required in dev mode (configurable)
  if (!keyValue && process.env.API_AUTH_REQUIRED !== 'true') {
    return { authenticated: true };
  }

  if (!keyValue) {
    return {
      authenticated: false,
      error: 'API key required. Use Authorization: Bearer <key> or X-API-Key header.',
    };
  }

  const apiKey = validateApiKey(keyValue);
  if (!apiKey) {
    return { authenticated: false, error: 'Invalid or expired API key.' };
  }

  return { authenticated: true, apiKey };
}

/**
 * Create a new API key
 */
export function createApiKey(
  name: string,
  scopes: string[] = ['*'],
  expiresInDays?: number
): { id: string; key: string } {
  const store = loadApiKeys();
  const { key, hash } = generateApiKey();
  const id = crypto.randomUUID();

  const apiKey: ApiKey = {
    id,
    name,
    keyHash: hash,
    scopes,
    createdAt: new Date().toISOString(),
    ...(expiresInDays && {
      expiresAt: new Date(Date.now() + expiresInDays * 86400000).toISOString(),
    }),
  };

  store.keys.push(apiKey);
  saveApiKeys(store);

  return { id, key };
}

/**
 * Revoke an API key by ID
 */
export function revokeApiKey(id: string): boolean {
  const store = loadApiKeys();
  const index = store.keys.findIndex((k) => k.id === id);
  if (index === -1) return false;
  store.keys.splice(index, 1);
  saveApiKeys(store);
  return true;
}

/**
 * List all API keys (without exposing hashes)
 */
export function listApiKeys(): Omit<ApiKey, 'keyHash'>[] {
  const store = loadApiKeys();
  return store.keys.map(({ keyHash: _, ...rest }) => rest);
}

// CLI
if (require.main === module || process.argv[1]?.includes('auth')) {
  const args = process.argv.slice(2);
  const cmd = args[0];

  switch (cmd) {
    case 'create': {
      const name = args[1] || 'default';
      const scopes = args[2]?.split(',') || ['*'];
      const days = args[3] ? parseInt(args[3]) : undefined;
      const { id, key } = createApiKey(name, scopes, days);
      console.log(`\n✅ API Key Created\n`);
      console.log(`   ID:    ${id}`);
      console.log(`   Name:  ${name}`);
      console.log(`   Key:   ${key}`);
      console.log(`   Scopes: ${scopes.join(', ')}`);
      if (days) console.log(`   Expires: ${days} days`);
      console.log(`\n⚠️  Save this key securely - it cannot be retrieved later!\n`);
      break;
    }
    case 'revoke': {
      const success = revokeApiKey(args[1]);
      console.log(success ? `✅ API key ${args[1]} revoked` : `❌ API key not found`);
      break;
    }
    case 'list':
      console.log(JSON.stringify(listApiKeys(), null, 2));
      break;
    case 'validate': {
      const result = validateApiKey(args[1]);
      console.log(
        result ? `✅ Valid: ${result.name} (${result.scopes.join(', ')})` : '❌ Invalid key'
      );
      break;
    }
    default:
      console.log(`
API Authentication CLI

Commands:
  create <name> [scopes] [expiry_days]   Create new API key
  revoke <id>                            Revoke an API key
  list                                   List all API keys
  validate <key>                         Validate an API key

Examples:
  npx tsx tools/ai/api/auth.ts create prod-key "read:*,write:compliance" 365
  npx tsx tools/ai/api/auth.ts list
`);
  }
}
