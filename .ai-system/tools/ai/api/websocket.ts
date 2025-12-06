#!/usr/bin/env npx tsx
/**
 * WebSocket Server for Real-time Dashboard Updates
 * Pushes metrics to connected clients every 5 seconds
 */

import { WebSocketServer, WebSocket } from 'ws';
import * as fs from 'fs';
import * as path from 'path';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');
const WS_PORT = parseInt(process.env.WS_PORT || '3201', 10);
const BROADCAST_INTERVAL = 5000; // 5 seconds

interface Client {
  ws: WebSocket;
  subscriptions: Set<string>;
  connectedAt: Date;
}

interface MetricsPayload {
  type: 'metrics' | 'health' | 'compliance' | 'errors' | 'ping';
  data: unknown;
  timestamp: string;
}

const clients = new Map<string, Client>();
let broadcastInterval: NodeJS.Timeout | null = null;

// Read JSON file safely
function readJsonFile(filePath: string): unknown {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }
  } catch {
    // Ignore errors
  }
  return null;
}

// Gather all metrics for broadcast
function gatherMetrics(): MetricsPayload {
  const metrics = readJsonFile(path.join(AI_DIR, 'metrics.json')) || {};
  const compliance = readJsonFile(path.join(AI_DIR, 'compliance-report.json')) || {};
  const errors = readJsonFile(path.join(ROOT, '.metaHub/reports/error-tracking.json')) || {
    errors: [],
  };
  const issues = readJsonFile(path.join(ROOT, '.metaHub/reports/issues.json')) || { issues: [] };

  return {
    type: 'metrics',
    data: {
      ...metrics,
      compliance,
      errors: (errors as { errors?: unknown[] }).errors?.length || 0,
      issues: (issues as { issues?: unknown[] }).issues?.length || 0,
      connectedClients: clients.size,
      uptime: process.uptime(),
    },
    timestamp: new Date().toISOString(),
  };
}

// Broadcast to all subscribed clients
function broadcast(payload: MetricsPayload): void {
  const message = JSON.stringify(payload);

  for (const [id, client] of clients.entries()) {
    if (client.ws.readyState === WebSocket.OPEN) {
      if (client.subscriptions.has('*') || client.subscriptions.has(payload.type)) {
        try {
          client.ws.send(message);
        } catch {
          clients.delete(id);
        }
      }
    } else {
      clients.delete(id);
    }
  }
}

// Start WebSocket server
export function startWebSocketServer(): WebSocketServer {
  const wss = new WebSocketServer({ port: WS_PORT });

  console.log(`🔌 WebSocket server listening on ws://localhost:${WS_PORT}`);

  wss.on('connection', (ws, req) => {
    const clientId = `${req.socket.remoteAddress}-${Date.now()}`;
    const client: Client = {
      ws,
      subscriptions: new Set(['*']), // Subscribe to all by default
      connectedAt: new Date(),
    };

    clients.set(clientId, client);
    console.log(`✅ Client connected: ${clientId} (total: ${clients.size})`);

    // Send initial metrics
    ws.send(JSON.stringify(gatherMetrics()));

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());

        // Handle subscription changes
        if (msg.action === 'subscribe' && Array.isArray(msg.topics)) {
          client.subscriptions = new Set(msg.topics);
          ws.send(JSON.stringify({ type: 'subscribed', topics: msg.topics }));
        }

        // Handle ping
        if (msg.action === 'ping') {
          ws.send(JSON.stringify({ type: 'pong', timestamp: new Date().toISOString() }));
        }
      } catch {
        // Ignore invalid messages
      }
    });

    ws.on('close', () => {
      clients.delete(clientId);
      console.log(`❌ Client disconnected: ${clientId} (total: ${clients.size})`);
    });

    ws.on('error', () => {
      clients.delete(clientId);
    });
  });

  // Start broadcast interval
  broadcastInterval = setInterval(() => {
    if (clients.size > 0) {
      broadcast(gatherMetrics());
    }
  }, BROADCAST_INTERVAL);

  return wss;
}

// Stop WebSocket server
export function stopWebSocketServer(wss: WebSocketServer): void {
  if (broadcastInterval) {
    clearInterval(broadcastInterval);
    broadcastInterval = null;
  }

  for (const [, client] of clients.entries()) {
    client.ws.close();
  }
  clients.clear();

  wss.close();
}

// CLI
if (require.main === module || process.argv[1]?.includes('websocket')) {
  const wss = startWebSocketServer();

  process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down WebSocket server...');
    stopWebSocketServer(wss);
    process.exit(0);
  });
}
