const { EventEmitter } = require('events');
const path = require('path');
const fs = require('fs');

class AuditSystem extends EventEmitter {
  constructor({ rootPath }) {
    super();
    this.rootPath = rootPath;
    this.eventTypes = {
      AUTOMATION_RUN: 'automation_run',
      ERROR: 'error',
    };
    this.logDir = path.join(this.rootPath, '.governance', 'reports');
  }

  async logAudit(eventType, payload = {}) {
    const entry = {
      ts: new Date().toISOString(),
      eventType,
      details: payload,
    };
    try {
      fs.mkdirSync(this.logDir, { recursive: true });
      const file = path.join(this.logDir, `audit-${Date.now()}.json`);
      fs.writeFileSync(file, JSON.stringify(entry, null, 2));
      if (eventType === this.eventTypes.ERROR) {
        this.emit('alert', entry);
      }
    } catch (err) {
      // Swallow audit write failures to avoid blocking validation.
      console.error('AuditSystem write failed:', err.message);
    }
    return entry;
  }
}

module.exports = AuditSystem;
