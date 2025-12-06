#!/usr/bin/env tsx
/**
 * Universal AI Proxy - Core Module
 * OpenAI-compatible proxy with automatic model tiering
 */

import { IncomingMessage } from 'http';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import * as yaml from 'js-yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..', '..', '..');

export const PORT = parseInt(process.env.AI_PROXY_PORT || '4000', 10);
export const LOG_FILE = join(ROOT_DIR, '.config', 'ai', 'logs', 'token-usage.jsonl');
const TIERING_CONFIG = join(ROOT_DIR, '.config', 'ai', 'model-tiering.yaml');

// Provider endpoints
export const PROVIDERS: Record<string, { baseUrl: string; envKey: string }> = {
  openai: { baseUrl: 'https://api.openai.com/v1', envKey: 'OPENAI_API_KEY' },
  anthropic: { baseUrl: 'https://api.anthropic.com/v1', envKey: 'ANTHROPIC_API_KEY' },
  ollama: { baseUrl: 'http://localhost:11434/v1', envKey: '' },
};

// Model to provider mapping
export const MODEL_PROVIDERS: Record<string, string> = {
  'gpt-4o': 'openai',
  'gpt-4o-mini': 'openai',
  'gpt-4-turbo': 'openai',
  'gpt-3.5-turbo': 'openai',
  'claude-3-opus': 'anthropic',
  'claude-3-sonnet': 'anthropic',
  'claude-3-haiku': 'anthropic',
  llama3: 'ollama',
  codellama: 'ollama',
  mistral: 'ollama',
};

// Tier to model mapping
export const TIER_MODELS: Record<string, string[]> = {
  lightweight: ['gpt-4o-mini', 'claude-3-haiku', 'codellama'],
  standard: ['gpt-4o', 'claude-3-sonnet', 'llama3'],
  heavyweight: ['gpt-4-turbo', 'claude-3-opus', 'mistral'],
};

interface TieringConfig {
  tiers: { [key: string]: { triggers: { query_patterns: string[] } } };
}

export interface ChatRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
}

// Load tiering configuration
export function loadTieringConfig(): TieringConfig | null {
  try {
    if (existsSync(TIERING_CONFIG)) {
      return yaml.load(readFileSync(TIERING_CONFIG, 'utf-8')) as TieringConfig;
    }
  } catch (e) {
    console.error('Failed to load tiering config:', e);
  }
  return null;
}

// Estimate tokens (rough approximation: 4 chars = 1 token)
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// Select tier based on query content
export function selectTier(messages: Array<{ role: string; content: string }>): string {
  const config = loadTieringConfig();
  const fullText = messages
    .map((m) => m.content)
    .join(' ')
    .toLowerCase();
  const estimatedTokens = estimateTokens(fullText);

  const heavyPatterns = config?.tiers?.heavyweight?.triggers?.query_patterns || [
    'architect',
    'design system',
    'security audit',
    'enterprise',
    'analyze codebase',
  ];
  if (heavyPatterns.some((p) => fullText.includes(p)) || estimatedTokens > 12000) {
    return 'heavyweight';
  }

  const lightPatterns = config?.tiers?.lightweight?.triggers?.query_patterns || [
    'fix typo',
    'add import',
    'rename',
    'format',
    'complete this',
  ];
  if (lightPatterns.some((p) => fullText.includes(p)) && estimatedTokens < 2000) {
    return 'lightweight';
  }

  return 'standard';
}

// Select best available model for tier
export function selectModel(tier: string, requestedModel?: string): string {
  const tierModels = TIER_MODELS[tier] || TIER_MODELS.standard;
  if (requestedModel && tierModels.includes(requestedModel)) return requestedModel;

  for (const model of tierModels) {
    const provider = MODEL_PROVIDERS[model];
    if (provider === 'ollama') return model;
    const envKey = PROVIDERS[provider]?.envKey;
    if (envKey && process.env[envKey]) return model;
  }
  return requestedModel || tierModels[0];
}

// Log token usage
export function logUsage(
  tier: string,
  model: string,
  inputTokens: number,
  outputTokens: number,
  cost: number
): void {
  const logDir = dirname(LOG_FILE);
  if (!existsSync(logDir)) mkdirSync(logDir, { recursive: true });
  const entry = {
    timestamp: new Date().toISOString(),
    tier,
    model,
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    estimated_cost: cost,
    task_type: 'proxy',
  };
  writeFileSync(LOG_FILE, JSON.stringify(entry) + '\n', { flag: 'a' });
}

// Forward request to provider
export async function forwardRequest(
  model: string,
  body: ChatRequest,
  headers: Record<string, string>
): Promise<Response> {
  const provider = MODEL_PROVIDERS[model] || 'openai';
  const config = PROVIDERS[provider];
  const apiKey = process.env[config.envKey];

  const reqHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
  if (provider === 'openai') {
    reqHeaders['Authorization'] =
      `Bearer ${apiKey || headers['authorization']?.replace('Bearer ', '')}`;
  } else if (provider === 'anthropic') {
    reqHeaders['x-api-key'] = apiKey || headers['x-api-key'] || '';
    reqHeaders['anthropic-version'] = '2023-06-01';
  }

  return fetch(`${config.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: reqHeaders,
    body: JSON.stringify({ ...body, model }),
  });
}

// Parse request body
export async function parseBody(req: IncomingMessage): Promise<ChatRequest> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => (data += chunk));
    req.on('end', () => {
      try {
        resolve(JSON.parse(data));
      } catch {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}
