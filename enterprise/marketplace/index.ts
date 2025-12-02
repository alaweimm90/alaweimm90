// ATLAS Enterprise - Agent Marketplace

import { AgentPlugin, PluginManifest, MarketplaceConfig } from './types.js';
import { PluginManager } from './plugin-manager.js';
import { MarketplaceRegistry } from './registry.js';

export * from './types.js';
export * from './plugin-manager.js';
export * from './registry.js';

/**
 * Agent Marketplace for third-party agent plugins
 */
export class AgentMarketplace {
  private pluginManager: PluginManager;
  private registry: MarketplaceRegistry;
  private config: MarketplaceConfig;

  constructor(config?: Partial<MarketplaceConfig>) {
    this.config = {
      registryUrl: 'https://marketplace.atlas-platform.com',
      trustedPublishers: [],
      autoUpdate: true,
      securityScan: true,
      ...config,
    };

    this.pluginManager = new PluginManager();
    this.registry = new MarketplaceRegistry(this.config.registryUrl);
  }

  /**
   * Install an agent plugin
   */
  async installPlugin(pluginId: string, version?: string): Promise<void> {
    console.log(`Installing plugin: ${pluginId}${version ? `@${version}` : ''}`);

    // Fetch plugin manifest
    const manifest = await this.registry.getPluginManifest(pluginId, version);

    // Security scan
    if (this.config.securityScan) {
      await this.securityScan(manifest);
    }

    // Download and install
    await this.pluginManager.installPlugin(manifest);

    console.log(`Plugin ${pluginId} installed successfully`);
  }

  /**
   * Uninstall an agent plugin
   */
  async uninstallPlugin(pluginId: string): Promise<void> {
    console.log(`Uninstalling plugin: ${pluginId}`);

    await this.pluginManager.uninstallPlugin(pluginId);

    console.log(`Plugin ${pluginId} uninstalled successfully`);
  }

  /**
   * List installed plugins
   */
  listInstalledPlugins(): AgentPlugin[] {
    return this.pluginManager.listPlugins();
  }

  /**
   * Search marketplace for plugins
   */
  async searchPlugins(query: string, category?: string): Promise<PluginManifest[]> {
    return this.registry.searchPlugins(query, category);
  }

  /**
   * Get plugin details
   */
  async getPluginDetails(pluginId: string): Promise<PluginManifest> {
    return this.registry.getPluginManifest(pluginId);
  }

  /**
   * Update all plugins
   */
  async updateAllPlugins(): Promise<void> {
    const plugins = this.listInstalledPlugins();

    for (const plugin of plugins) {
      try {
        await this.updatePlugin(plugin.id);
      } catch (error) {
        console.error(`Failed to update plugin ${plugin.id}:`, error);
      }
    }
  }

  /**
   * Update specific plugin
   */
  async updatePlugin(pluginId: string): Promise<void> {
    const latestManifest = await this.registry.getPluginManifest(pluginId);
    const installedPlugin = this.pluginManager.getPlugin(pluginId);

    if (!installedPlugin) {
      throw new Error(`Plugin ${pluginId} is not installed`);
    }

    if (latestManifest.version !== installedPlugin.version) {
      console.log(
        `Updating ${pluginId} from ${installedPlugin.version} to ${latestManifest.version}`
      );
      await this.pluginManager.updatePlugin(latestManifest);
    }
  }

  /**
   * Publish a plugin to marketplace
   */
  async publishPlugin(manifest: PluginManifest, packagePath: string): Promise<void> {
    // Validate manifest
    this.validateManifest(manifest);

    // Upload to registry
    await this.registry.publishPlugin(manifest, packagePath);

    console.log(`Plugin ${manifest.id} published successfully`);
  }

  /**
   * Load and initialize all installed plugins
   */
  async loadPlugins(): Promise<void> {
    await this.pluginManager.loadAllPlugins();
  }

  /**
   * Get plugin instance
   */
  getPlugin(pluginId: string): AgentPlugin | null {
    return this.pluginManager.getPlugin(pluginId);
  }

  private async securityScan(manifest: PluginManifest): Promise<void> {
    // Basic security checks
    if (!manifest.publisher) {
      throw new Error('Plugin must have a publisher');
    }

    if (
      this.config.trustedPublishers.length > 0 &&
      !this.config.trustedPublishers.includes(manifest.publisher)
    ) {
      console.warn(`Plugin from untrusted publisher: ${manifest.publisher}`);
    }

    // Check for malicious patterns in code (simplified)
    // In a real implementation, this would use more sophisticated analysis
  }

  private validateManifest(manifest: PluginManifest): void {
    if (!manifest.id || !manifest.name || !manifest.version) {
      throw new Error('Plugin manifest must include id, name, and version');
    }

    if (!manifest.capabilities || manifest.capabilities.length === 0) {
      throw new Error('Plugin must declare capabilities');
    }

    if (!manifest.entryPoint) {
      throw new Error('Plugin must specify an entry point');
    }
  }
}

/**
 * Initialize agent marketplace
 */
export async function initializeMarketplace(
  config?: Partial<MarketplaceConfig>
): Promise<AgentMarketplace> {
  const marketplace = new AgentMarketplace(config);
  await marketplace.loadPlugins();
  return marketplace;
}
