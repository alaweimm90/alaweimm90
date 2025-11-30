// ATLAS Enterprise - Plugin Manager

import { AgentPlugin, PluginManifest, PluginConfig, PluginContext } from './types.js';
import * as fs from 'fs';
import * as path from 'path';

export class PluginManager {
  private plugins: Map<string, AgentPlugin> = new Map();
  private pluginDir: string;

  constructor(pluginDir: string = './plugins') {
    this.pluginDir = pluginDir;
    this.ensurePluginDirectory();
  }

  /**
   * Install a plugin
   */
  async installPlugin(manifest: PluginManifest): Promise<void> {
    const pluginPath = path.join(this.pluginDir, manifest.id);

    // Create plugin directory
    if (!fs.existsSync(pluginPath)) {
      fs.mkdirSync(pluginPath, { recursive: true });
    }

    // Download plugin package (simplified)
    // In a real implementation, this would download from registry
    console.log(`Downloading plugin ${manifest.id}...`);

    // Save manifest
    const manifestPath = path.join(pluginPath, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

    // Create plugin instance
    const plugin: AgentPlugin = {
      ...manifest,
      config: this.createDefaultConfig(manifest)
    };

    this.plugins.set(manifest.id, plugin);

    // Save installation record
    this.saveInstallationRecord(manifest);
  }

  /**
   * Uninstall a plugin
   */
  async uninstallPlugin(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      throw new Error(`Plugin ${pluginId} is not installed`);
    }

    // Cleanup plugin files
    const pluginPath = path.join(this.pluginDir, pluginId);
    if (fs.existsSync(pluginPath)) {
      fs.rmSync(pluginPath, { recursive: true, force: true });
    }

    // Remove from registry
    this.plugins.delete(pluginId);

    // Remove installation record
    this.removeInstallationRecord(pluginId);
  }

  /**
   * Update a plugin
   */
  async updatePlugin(manifest: PluginManifest): Promise<void> {
    await this.uninstallPlugin(manifest.id);
    await this.installPlugin(manifest);
  }

  /**
   * Load all installed plugins
   */
  async loadAllPlugins(): Promise<void> {
    const installations = this.loadInstallationRecords();

    for (const installation of installations) {
      try {
        await this.loadPlugin(installation.pluginId);
      } catch (error) {
        console.error(`Failed to load plugin ${installation.pluginId}:`, error);
      }
    }
  }

  /**
   * Load a specific plugin
   */
  async loadPlugin(pluginId: string): Promise<void> {
    const pluginPath = path.join(this.pluginDir, pluginId);
    const manifestPath = path.join(pluginPath, 'manifest.json');

    if (!fs.existsSync(manifestPath)) {
      throw new Error(`Plugin ${pluginId} manifest not found`);
    }

    const manifest: PluginManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const entryPoint = path.join(pluginPath, manifest.entryPoint);

    if (!fs.existsSync(entryPoint)) {
      throw new Error(`Plugin ${pluginId} entry point not found: ${entryPoint}`);
    }

    // Load plugin module (simplified)
    // In a real implementation, this would use proper module loading
    const pluginModule = await import(entryPoint);

    // Create plugin instance
    const plugin: AgentPlugin = {
      ...manifest,
      instance: new pluginModule.default(this.createPluginContext(manifest))
    };

    this.plugins.set(pluginId, plugin);
  }

  /**
   * Get a plugin instance
   */
  getPlugin(pluginId: string): AgentPlugin | null {
    return this.plugins.get(pluginId) || null;
  }

  /**
   * List all installed plugins
   */
  listPlugins(): AgentPlugin[] {
    return Array.from(this.plugins.values());
  }

  /**
   * Check if plugin is installed
   */
  isInstalled(pluginId: string): boolean {
    return this.plugins.has(pluginId);
  }

  private ensurePluginDirectory(): void {
    if (!fs.existsSync(this.pluginDir)) {
      fs.mkdirSync(this.pluginDir, { recursive: true });
    }
  }

  private createDefaultConfig(manifest: PluginManifest): PluginConfig {
    return {
      enabled: true,
      settings: {},
      permissions: manifest.security.permissions
    };
  }

  private createPluginContext(manifest: PluginManifest): PluginContext {
    return {
      atlas: {}, // ATLAS core instance would be injected here
      config: this.createDefaultConfig(manifest),
      logger: console,
      storage: {} // Storage interface would be injected here
    };
  }

  private saveInstallationRecord(manifest: PluginManifest): void {
    const records = this.loadInstallationRecords();
    const record = {
      pluginId: manifest.id,
      version: manifest.version,
      installedAt: new Date().toISOString(),
      checksum: manifest.security.checksum,
      config: this.createDefaultConfig(manifest)
    };

    records.push(record);
    this.saveInstallationRecords(records);
  }

  private removeInstallationRecord(pluginId: string): void {
    const records = this.loadInstallationRecords();
    const filtered = records.filter(r => r.pluginId !== pluginId);
    this.saveInstallationRecords(filtered);
  }

  private loadInstallationRecords(): any[] {
    const recordPath = path.join(this.pluginDir, 'installations.json');
    if (!fs.existsSync(recordPath)) {
      return [];
    }
    return JSON.parse(fs.readFileSync(recordPath, 'utf8'));
  }

  private saveInstallationRecords(records: any[]): void {
    const recordPath = path.join(this.pluginDir, 'installations.json');
    fs.writeFileSync(recordPath, JSON.stringify(records, null, 2));
  }
}</content>
</edit_file>