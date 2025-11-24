/**
 * MCP Registry - Manages available MCP servers
 */

import * as fs from 'fs';
import * as path from 'path';
import { MCPServerConfig, MCPRegistry, MCPCategory } from './types';

export class MCPRegistryManager {
  private registryPath: string;

  private registry: MCPRegistry;

  constructor(registryPath: string = path.join(__dirname, '../mcps/registry.json')) {
    this.registryPath = registryPath;
    this.registry = this.loadRegistry();
  }

  private loadRegistry(): MCPRegistry {
    try {
      if (fs.existsSync(this.registryPath)) {
        const data = fs.readFileSync(this.registryPath, 'utf-8');
        return JSON.parse(data);
      }
    } catch (error) {
      console.warn(`Could not load registry from ${this.registryPath}:`, error);
    }

    return {
      version: '1.0.0',
      lastUpdated: new Date().toISOString(),
      servers: [],
    };
  }

  public getServer(name: string): MCPServerConfig | undefined {
    return this.registry.servers.find(server => server.name === name);
  }

  public getServers(): MCPServerConfig[] {
    return this.registry.servers;
  }

  public getServersByCategory(category: MCPCategory): MCPServerConfig[] {
    return this.registry.servers.filter(server => server.category === category);
  }

  public getEnabledServers(): MCPServerConfig[] {
    return this.registry.servers.filter(server => server.enabled);
  }

  public registerServer(config: MCPServerConfig): void {
    const existing = this.registry.servers.findIndex(s => s.name === config.name);
    if (existing >= 0) {
      this.registry.servers[existing] = config;
    } else {
      this.registry.servers.push(config);
    }
    this.registry.lastUpdated = new Date().toISOString();
    this.saveRegistry();
  }

  public unregisterServer(name: string): void {
    this.registry.servers = this.registry.servers.filter(s => s.name !== name);
    this.registry.lastUpdated = new Date().toISOString();
    this.saveRegistry();
  }

  public enableServer(name: string): void {
    const server = this.getServer(name);
    if (server) {
      server.enabled = true;
      this.registry.lastUpdated = new Date().toISOString();
      this.saveRegistry();
    }
  }

  public disableServer(name: string): void {
    const server = this.getServer(name);
    if (server) {
      server.enabled = false;
      this.registry.lastUpdated = new Date().toISOString();
      this.saveRegistry();
    }
  }

  public saveRegistry(): void {
    const dir = path.dirname(this.registryPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(this.registryPath, JSON.stringify(this.registry, null, 2), 'utf-8');
  }

  public listServersByCategory(): Record<MCPCategory, MCPServerConfig[]> {
    const result: Record<string, MCPServerConfig[]> = {};
    Object.values(MCPCategory).forEach(category => {
      result[category] = this.getServersByCategory(category);
    });
    return result as Record<MCPCategory, MCPServerConfig[]>;
  }
}
