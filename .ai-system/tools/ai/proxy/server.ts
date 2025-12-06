#!/usr/bin/env tsx
/**
 * Universal AI Proxy Server
 * OpenAI-compatible endpoints with automatic tier routing
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import {
  PORT,
  TIER_MODELS,
  MODEL_PROVIDERS,
  PROVIDERS,
  selectTier,
  selectModel,
  logUsage,
  forwardRequest,
  parseBody,
  ChatRequest,
} from './core.js';

// Main chat completions handler
async function handleChatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    const body: ChatRequest = await parseBody(req);
    const tier = selectTier(body.messages);
    const selectedModel = selectModel(tier, body.model);

    console.log(
      `[PROXY] Tier: ${tier.toUpperCase()} | Model: ${selectedModel} | Requested: ${body.model}`
    );

    // Get headers as record
    const headers: Record<string, string> = {};
    for (const [key, value] of Object.entries(req.headers)) {
      if (typeof value === 'string') headers[key] = value;
    }

    // Forward to provider
    const providerResponse = await forwardRequest(selectedModel, body, headers);
    const responseData = await providerResponse.json();

    // Log usage
    const usage = responseData.usage || { prompt_tokens: 0, completion_tokens: 0 };
    const costPerK = tier === 'lightweight' ? 0.0005 : tier === 'standard' ? 0.01 : 0.03;
    const cost = ((usage.prompt_tokens + usage.completion_tokens) / 1000) * costPerK;
    logUsage(tier, selectedModel, usage.prompt_tokens, usage.completion_tokens, cost);

    // Return response with tier headers
    res.writeHead(providerResponse.status, {
      'Content-Type': 'application/json',
      'X-AI-Tier': tier,
      'X-AI-Model': selectedModel,
    });
    res.end(JSON.stringify(responseData));
  } catch (error) {
    console.error('[PROXY] Error:', error);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: String(error) } }));
  }
}

// Health check
function handleHealth(_req: IncomingMessage, res: ServerResponse): void {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(
    JSON.stringify({
      status: 'ok',
      version: '1.0.0',
      tiers: Object.keys(TIER_MODELS),
      providers: Object.keys(PROVIDERS),
    })
  );
}

// Models list (OpenAI-compatible)
function handleModels(_req: IncomingMessage, res: ServerResponse): void {
  const models = Object.keys(MODEL_PROVIDERS).map((id) => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: MODEL_PROVIDERS[id],
  }));
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ object: 'list', data: models }));
}

// Create server
const server = createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url || '/', `http://localhost:${PORT}`);
  const path = url.pathname;

  if (path === '/health' || path === '/') {
    handleHealth(req, res);
  } else if (path === '/v1/models' || path === '/models') {
    handleModels(req, res);
  } else if (
    (path === '/v1/chat/completions' || path === '/chat/completions') &&
    req.method === 'POST'
  ) {
    await handleChatCompletions(req, res);
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'Not found' } }));
  }
});

// CLI
function main(): void {
  const command = process.argv[2];

  if (command === 'start') {
    server.listen(PORT, () => {
      console.log(`
╔══════════════════════════════════════════════════════════════╗
║          🔀 UNIVERSAL AI PROXY SERVER                        ║
╠══════════════════════════════════════════════════════════════╣
║  Server: http://localhost:${PORT.toString().padEnd(35)}║
║                                                              ║
║  OpenAI-Compatible Endpoints:                                ║
║    POST /v1/chat/completions  - Chat (with tier routing)     ║
║    GET  /v1/models            - List available models        ║
║    GET  /health               - Health check                 ║
║                                                              ║
║  Features:                                                   ║
║    ✓ Automatic tier selection (lightweight/standard/heavy)   ║
║    ✓ Token usage logging to .config/ai/logs/                 ║
║    ✓ Multi-provider routing (OpenAI, Anthropic, Ollama)      ║
║    ✓ Response headers: X-AI-Tier, X-AI-Model                 ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Configure Your Tools:                                       ║
║                                                              ║
║  Continue.dev (~/.continue/config.json):                     ║
║    "apiBase": "http://localhost:${PORT}/v1"                    ║
║                                                              ║
║  Cursor (Settings → Models → OpenAI Base URL):               ║
║    http://localhost:${PORT}/v1                                 ║
║                                                              ║
║  CLI: OPENAI_BASE_URL=http://localhost:${PORT}/v1              ║
╚══════════════════════════════════════════════════════════════╝
      `);
    });
  } else {
    console.log(`
Universal AI Proxy - OpenAI-compatible proxy with model tiering

Usage:
  npm run ai:proxy start   Start the proxy server

Environment:
  AI_PROXY_PORT     Port (default: 4000)
  OPENAI_API_KEY    OpenAI API key
  ANTHROPIC_API_KEY Anthropic API key
    `);
  }
}

main();
