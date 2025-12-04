#!/usr/bin/env npx tsx
/**
 * Agent Handoff Persistence
 * SQLite-based context caching for cross-agent sessions
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const ROOT = process.cwd();
const DB_PATH = path.join(ROOT, '.ai/handoff/sessions.json');
const CONTEXT_DIR = path.join(ROOT, '.ai/handoff/contexts');

export interface HandoffContext {
  id: string;
  sourceAgent: string;
  targetAgent: string;
  taskType: string;
  context: Record<string, unknown>;
  files: string[];
  metadata: Record<string, unknown>;
  createdAt: string;
  expiresAt: string;
  status: 'pending' | 'accepted' | 'completed' | 'expired';
}

export interface SessionStore {
  sessions: Record<string, HandoffContext>;
  lastUpdated: string;
}

class HandoffPersistence {
  private store: SessionStore;

  constructor() {
    this.ensureDirectories();
    this.store = this.loadStore();
    this.cleanupExpired();
  }

  private ensureDirectories(): void {
    const dbDir = path.dirname(DB_PATH);
    if (!fs.existsSync(dbDir)) fs.mkdirSync(dbDir, { recursive: true });
    if (!fs.existsSync(CONTEXT_DIR)) fs.mkdirSync(CONTEXT_DIR, { recursive: true });
  }

  private loadStore(): SessionStore {
    if (fs.existsSync(DB_PATH)) {
      return JSON.parse(fs.readFileSync(DB_PATH, 'utf8'));
    }
    return { sessions: {}, lastUpdated: new Date().toISOString() };
  }

  private saveStore(): void {
    this.store.lastUpdated = new Date().toISOString();
    fs.writeFileSync(DB_PATH, JSON.stringify(this.store, null, 2));
  }

  private cleanupExpired(): void {
    const now = new Date();
    for (const [id, session] of Object.entries(this.store.sessions)) {
      if (new Date(session.expiresAt) < now) {
        session.status = 'expired';
        const contextFile = path.join(CONTEXT_DIR, `${id}.json`);
        if (fs.existsSync(contextFile)) fs.unlinkSync(contextFile);
      }
    }
    this.saveStore();
  }

  createSession(
    sourceAgent: string,
    targetAgent: string,
    taskType: string,
    context: Record<string, unknown>,
    files: string[] = [],
    ttlHours: number = 24
  ): HandoffContext {
    const id = crypto.randomBytes(16).toString('hex');
    const now = new Date();
    const expiresAt = new Date(now.getTime() + ttlHours * 60 * 60 * 1000);

    const session: HandoffContext = {
      id,
      sourceAgent,
      targetAgent,
      taskType,
      context,
      files,
      metadata: { ttlHours, contextSize: JSON.stringify(context).length },
      createdAt: now.toISOString(),
      expiresAt: expiresAt.toISOString(),
      status: 'pending',
    };

    // Store full context separately for large payloads
    fs.writeFileSync(
      path.join(CONTEXT_DIR, `${id}.json`),
      JSON.stringify({ context, files }, null, 2)
    );

    this.store.sessions[id] = session;
    this.saveStore();
    return session;
  }

  getSession(id: string): HandoffContext | null {
    const session = this.store.sessions[id];
    if (!session || session.status === 'expired') return null;

    // Load full context
    const contextFile = path.join(CONTEXT_DIR, `${id}.json`);
    if (fs.existsSync(contextFile)) {
      const fullContext = JSON.parse(fs.readFileSync(contextFile, 'utf8'));
      return { ...session, context: fullContext.context, files: fullContext.files };
    }
    return session;
  }

  acceptSession(id: string, targetAgent: string): boolean {
    const session = this.store.sessions[id];
    if (!session || session.status !== 'pending') return false;
    if (session.targetAgent !== targetAgent) return false;

    session.status = 'accepted';
    this.saveStore();
    return true;
  }

  completeSession(id: string): boolean {
    const session = this.store.sessions[id];
    if (!session || session.status !== 'accepted') return false;

    session.status = 'completed';
    this.saveStore();
    return true;
  }

  listSessions(filter?: { agent?: string; status?: string }): HandoffContext[] {
    return Object.values(this.store.sessions).filter((s) => {
      if (filter?.agent && s.sourceAgent !== filter.agent && s.targetAgent !== filter.agent)
        return false;
      if (filter?.status && s.status !== filter.status) return false;
      return true;
    });
  }

  getStats(): Record<string, number> {
    const sessions = Object.values(this.store.sessions);
    return {
      total: sessions.length,
      pending: sessions.filter((s) => s.status === 'pending').length,
      accepted: sessions.filter((s) => s.status === 'accepted').length,
      completed: sessions.filter((s) => s.status === 'completed').length,
      expired: sessions.filter((s) => s.status === 'expired').length,
    };
  }
}

export const handoffPersistence = new HandoffPersistence();
export default HandoffPersistence;

// CLI
if (require.main === module || process.argv[1]?.includes('persistence')) {
  const args = process.argv.slice(2);
  const cmd = args[0];

  switch (cmd) {
    case 'create': {
      const [, source, target, taskType, contextJson] = args;
      if (!source || !target || !taskType) {
        console.log('Usage: persistence create <source> <target> <taskType> [contextJson]');
        process.exit(1);
      }
      const context = contextJson ? JSON.parse(contextJson) : {};
      const session = handoffPersistence.createSession(source, target, taskType, context);
      console.log(`✅ Session created: ${session.id}`);
      console.log(JSON.stringify(session, null, 2));
      break;
    }
    case 'get': {
      const session = handoffPersistence.getSession(args[1]);
      if (session) console.log(JSON.stringify(session, null, 2));
      else console.log('❌ Session not found or expired');
      break;
    }
    case 'accept': {
      const success = handoffPersistence.acceptSession(args[1], args[2]);
      console.log(success ? '✅ Session accepted' : '❌ Failed to accept session');
      break;
    }
    case 'complete': {
      const success = handoffPersistence.completeSession(args[1]);
      console.log(success ? '✅ Session completed' : '❌ Failed to complete session');
      break;
    }
    case 'list': {
      const filter: { agent?: string; status?: string } = {};
      if (args[1]) filter.agent = args[1];
      if (args[2]) filter.status = args[2];
      const sessions = handoffPersistence.listSessions(filter);
      console.log(JSON.stringify(sessions, null, 2));
      break;
    }
    case 'stats':
      console.log(JSON.stringify(handoffPersistence.getStats(), null, 2));
      break;
    default:
      console.log(`
Agent Handoff Persistence CLI

Commands:
  create <source> <target> <taskType> [contextJson]  Create new handoff session
  get <sessionId>                                    Get session by ID
  accept <sessionId> <targetAgent>                   Accept a pending session
  complete <sessionId>                               Mark session as completed
  list [agent] [status]                              List sessions with optional filters
  stats                                              Show session statistics
`);
  }
}
