/**
 * MCP Configuration Manager - Handles MCP configuration for different environments
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { MCPServerConfig, MCPEnvironment } from './types';

export class MCPConfigManager {
  private claudeCodeConfigPath: string;

  private devContainerConfigPath: string;

  private localConfigPath: string;

  constructor() {
    // Claude Code config paths
    if (process.platform === 'win32') {
      const appDataPath = process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming');
      this.claudeCodeConfigPath = path.join(appDataPath, 'Trae', 'User', 'claude-code-mcp.json');
    } else {
      this.claudeCodeConfigPath = path.join(os.homedir(), '.config', 'trae', 'claude-code-mcp.json');
    }

    this.devContainerConfigPath = path.join(process.cwd(), '.devcontainer', 'mcp-config.json');
    this.localConfigPath = path.join(process.cwd(), '.claude', 'mcp-config.json');
  }

  /**
   * Get Claude Code MCP configuration
   */
  public getClaudeCodeConfig(): MCPEnvironment {
    return this.loadConfig(this.claudeCodeConfigPath);
  }

  /**
   * Save Claude Code MCP configuration
   * @param config
   */
  public saveClaudeCodeConfig(config: MCPEnvironment): void {
    this.ensureDir(path.dirname(this.claudeCodeConfigPath));
    fs.writeFileSync(this.claudeCodeConfigPath, JSON.stringify(config, null, 2), 'utf-8');
  }

  /**
   * Get devContainer MCP configuration
   */
  public getDevContainerConfig(): MCPEnvironment {
    return this.loadConfig(this.devContainerConfigPath);
  }

  /**
   * Save devContainer MCP configuration
   * @param config
   */
  public saveDevContainerConfig(config: MCPEnvironment): void {
    this.ensureDir(path.dirname(this.devContainerConfigPath));
    fs.writeFileSync(this.devContainerConfigPath, JSON.stringify(config, null, 2), 'utf-8');
  }

  /**
   * Get local project MCP configuration
   */
  public getLocalConfig(): MCPEnvironment {
    return this.loadConfig(this.localConfigPath);
  }

  /**
   * Save local project MCP configuration
   * @param config
   */
  public saveLocalConfig(config: MCPEnvironment): void {
    this.ensureDir(path.dirname(this.localConfigPath));
    fs.writeFileSync(this.localConfigPath, JSON.stringify(config, null, 2), 'utf-8');
  }

  /**
   * Merge configurations from multiple sources (local -> devcontainer -> claude code)
   */
  public mergeConfigs(): MCPEnvironment {
    const configs = [this.getClaudeCodeConfig(), this.getDevContainerConfig(), this.getLocalConfig()];

    const merged: MCPEnvironment = {
      mcpServers: {},
      enabled: [],
      disabled: [],
    };

    for (const config of configs) {
      merged.mcpServers = { ...merged.mcpServers, ...config.mcpServers };
      merged.enabled = [...new Set([...merged.enabled, ...config.enabled])];
      merged.disabled = [...new Set([...merged.disabled, ...config.disabled])];
    }

    // Remove disabled servers from enabled
    merged.enabled = merged.enabled.filter(name => !merged.disabled.includes(name));

    return merged;
  }

  /**
   * Add a server to the configuration
   * @param server
   * @param target
   */
  public addServer(server: MCPServerConfig, target: 'local' | 'devcontainer' | 'claude-code' = 'local'): void {
    let config: MCPEnvironment;
    let saveFn: (c: MCPEnvironment) => void;

    switch (target) {
      case 'devcontainer':
        config = this.getDevContainerConfig();
        saveFn = c => this.saveDevContainerConfig(c);
        break;
      case 'claude-code':
        config = this.getClaudeCodeConfig();
        saveFn = c => this.saveClaudeCodeConfig(c);
        break;
      case 'local':
      default:
        config = this.getLocalConfig();
        saveFn = c => this.saveLocalConfig(c);
    }

    config.mcpServers[server.name] = server;
    if (server.enabled && !config.enabled.includes(server.name)) {
      config.enabled.push(server.name);
    }
    if (!server.enabled && !config.disabled.includes(server.name)) {
      config.disabled.push(server.name);
    }

    saveFn(config);
  }

  /**
   * Remove a server from the configuration
   * @param name
   * @param target
   */
  public removeServer(name: string, target: 'local' | 'devcontainer' | 'claude-code' = 'local'): void {
    let config: MCPEnvironment;
    let saveFn: (c: MCPEnvironment) => void;

    switch (target) {
      case 'devcontainer':
        config = this.getDevContainerConfig();
        saveFn = c => this.saveDevContainerConfig(c);
        break;
      case 'claude-code':
        config = this.getClaudeCodeConfig();
        saveFn = c => this.saveClaudeCodeConfig(c);
        break;
      case 'local':
      default:
        config = this.getLocalConfig();
        saveFn = c => this.saveLocalConfig(c);
    }

    delete config.mcpServers[name];
    config.enabled = config.enabled.filter(n => n !== name);
    config.disabled = config.disabled.filter(n => n !== name);

    saveFn(config);
  }

  private loadConfig(configPath: string): MCPEnvironment {
    try {
      if (fs.existsSync(configPath)) {
        const data = fs.readFileSync(configPath, 'utf-8');
        return JSON.parse(data);
      }
    } catch (error) {
      console.warn(`Could not load config from ${configPath}:`, error);
    }

    return {
      mcpServers: {},
      enabled: [],
      disabled: [],
    };
  }

  private ensureDir(dir: string): void {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }
}
