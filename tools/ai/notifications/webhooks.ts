#!/usr/bin/env npx tsx
/**
 * Webhook Notifications System
 * Send notifications for governance violations and security alerts
 */

import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import * as http from 'http';

const ROOT = process.cwd();
const CONFIG_PATH = path.join(ROOT, '.ai/webhooks/config.json');

export interface WebhookConfig {
  url: string;
  name: string;
  events: string[];
  secret?: string;
  enabled: boolean;
}

export interface WebhooksStore {
  webhooks: WebhookConfig[];
  lastUpdated: string;
}

export interface WebhookPayload {
  event: string;
  timestamp: string;
  data: Record<string, unknown>;
  source: string;
}

class WebhookNotifier {
  private config: WebhooksStore;

  constructor() {
    this.ensureDirectory();
    this.config = this.loadConfig();
  }

  private ensureDirectory(): void {
    const dir = path.dirname(CONFIG_PATH);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }

  private loadConfig(): WebhooksStore {
    if (fs.existsSync(CONFIG_PATH)) {
      return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
    }
    const defaultConfig: WebhooksStore = {
      webhooks: [],
      lastUpdated: new Date().toISOString(),
    };
    this.saveConfig(defaultConfig);
    return defaultConfig;
  }

  private saveConfig(config: WebhooksStore): void {
    config.lastUpdated = new Date().toISOString();
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
  }

  addWebhook(webhook: Omit<WebhookConfig, 'enabled'>): void {
    this.config.webhooks.push({ ...webhook, enabled: true });
    this.saveConfig(this.config);
  }

  removeWebhook(name: string): boolean {
    const idx = this.config.webhooks.findIndex((w) => w.name === name);
    if (idx === -1) return false;
    this.config.webhooks.splice(idx, 1);
    this.saveConfig(this.config);
    return true;
  }

  toggleWebhook(name: string, enabled: boolean): boolean {
    const webhook = this.config.webhooks.find((w) => w.name === name);
    if (!webhook) return false;
    webhook.enabled = enabled;
    this.saveConfig(this.config);
    return true;
  }

  listWebhooks(): WebhookConfig[] {
    return this.config.webhooks;
  }

  async send(
    event: string,
    data: Record<string, unknown>
  ): Promise<{ name: string; success: boolean; error?: string }[]> {
    const payload: WebhookPayload = {
      event,
      timestamp: new Date().toISOString(),
      data,
      source: 'meta-orchestration',
    };

    const results: { name: string; success: boolean; error?: string }[] = [];

    for (const webhook of this.config.webhooks) {
      if (!webhook.enabled || (!webhook.events.includes(event) && !webhook.events.includes('*'))) {
        continue;
      }

      try {
        await this.sendToWebhook(webhook, payload);
        results.push({ name: webhook.name, success: true });
      } catch (err) {
        results.push({
          name: webhook.name,
          success: false,
          error: err instanceof Error ? err.message : 'Unknown error',
        });
      }
    }

    return results;
  }

  private sendToWebhook(webhook: WebhookConfig, payload: WebhookPayload): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = new URL(webhook.url);
      const data = JSON.stringify(payload);
      const client = url.protocol === 'https:' ? https : http;

      const req = client.request(
        url,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(data),
            'User-Agent': 'MetaOrchestration-Webhook/1.0',
            ...(webhook.secret && { 'X-Webhook-Secret': webhook.secret }),
          },
        },
        (res) => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) resolve();
          else reject(new Error(`HTTP ${res.statusCode}`));
        }
      );

      req.on('error', reject);
      req.setTimeout(10000, () => {
        req.destroy();
        reject(new Error('Timeout'));
      });
      req.write(data);
      req.end();
    });
  }

  // Convenience methods for common events
  async notifyGovernanceViolation(details: Record<string, unknown>): Promise<void> {
    await this.send('governance.violation', details);
  }

  async notifySecurityAlert(details: Record<string, unknown>): Promise<void> {
    await this.send('security.alert', details);
  }

  async notifyComplianceFailure(details: Record<string, unknown>): Promise<void> {
    await this.send('compliance.failure', details);
  }

  async notifyTaskComplete(details: Record<string, unknown>): Promise<void> {
    await this.send('task.complete', details);
  }
}

export const webhookNotifier = new WebhookNotifier();
export default WebhookNotifier;

// CLI
if (require.main === module || process.argv[1]?.includes('webhooks')) {
  const args = process.argv.slice(2);
  const cmd = args[0];

  switch (cmd) {
    case 'add': {
      const [, name, url, ...events] = args;
      if (!name || !url || events.length === 0) {
        console.log('Usage: webhooks add <name> <url> <event1> [event2] ...');
        console.log(
          'Events: governance.violation, security.alert, compliance.failure, task.complete, *'
        );
        process.exit(1);
      }
      webhookNotifier.addWebhook({ name, url, events });
      console.log(`✅ Webhook "${name}" added`);
      break;
    }
    case 'remove': {
      const success = webhookNotifier.removeWebhook(args[1]);
      console.log(success ? `✅ Webhook "${args[1]}" removed` : `❌ Webhook not found`);
      break;
    }
    case 'enable':
    case 'disable': {
      const enabled = cmd === 'enable';
      const success = webhookNotifier.toggleWebhook(args[1], enabled);
      console.log(success ? `✅ Webhook "${args[1]}" ${cmd}d` : `❌ Webhook not found`);
      break;
    }
    case 'list':
      console.log(JSON.stringify(webhookNotifier.listWebhooks(), null, 2));
      break;
    case 'test': {
      const [, name] = args;
      webhookNotifier
        .send('test', { message: 'Test notification', from: name || 'CLI' })
        .then((results) => console.log(JSON.stringify(results, null, 2)))
        .catch((err) => console.error('Error:', err.message));
      break;
    }
    default:
      console.log(`
Webhook Notifications CLI

Commands:
  add <name> <url> <event1> [event2] ...   Add a webhook
  remove <name>                            Remove a webhook
  enable <name>                            Enable a webhook
  disable <name>                           Disable a webhook
  list                                     List all webhooks
  test [name]                              Send test notification

Events:
  governance.violation   Policy or rule violations
  security.alert         Security scan findings
  compliance.failure     Compliance check failures
  task.complete          Task completion notifications
  *                      All events
`);
  }
}
