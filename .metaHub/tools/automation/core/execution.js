let WebSocketServer;
const { join } = require('path');
const { writeFile, mkdir } = require('fs').promises;

class AuditTrail {
  constructor(logDir) {
    this.logDir = logDir;
  }

  async record(event) {
    const dir = join(this.logDir, 'audit');
    await mkdir(dir, { recursive: true });
    const file = join(dir, `event-${Date.now()}.json`);
    await writeFile(file, JSON.stringify(event));
  }
}

class ExecutionTracker {
  constructor({ logger, audit, broadcast }) {
    this.logger = logger;
    this.audit = audit;
    this.broadcast = broadcast;
  }

  async emit(type, payload) {
    const event = { type, ts: new Date().toISOString(), payload };
    await this.audit.record(event);
    if (this.broadcast) this.broadcast(event);
    this.logger.info('execution_event', event);
  }
}

class StatusServer {
  constructor({ port = 7070, logger }) {
    this.port = port;
    this.logger = logger;
    this.wss = null;
  }

  start() {
    try {
      if (!WebSocketServer) {
        const mod = require('ws');
        WebSocketServer = mod.Server || mod.WebSocketServer;
      }
      this.wss = new WebSocketServer({ port: this.port });
      this.logger.info(`status_server_started:${this.port}`);
    } catch (e) {
      this.logger.warn('status_server_unavailable', { error: e.message });
      this.wss = null;
    }
  }

  broadcast(event) {
    if (!this.wss) return;
    const data = JSON.stringify(event);
    for (const client of this.wss.clients) {
      if (client.readyState === 1) client.send(data);
    }
  }

  stop() {
    if (this.wss) this.wss.close();
  }
}

module.exports = { AuditTrail, ExecutionTracker, StatusServer };
