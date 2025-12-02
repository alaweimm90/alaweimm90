// ATLAS Enterprise - Marketplace Registry

import { PluginManifest, MarketplaceSearchResult, SearchFilters } from './types.js';

export class MarketplaceRegistry {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  /**
   * Search for plugins
   */
  async searchPlugins(query: string, category?: string): Promise<PluginManifest[]> {
    const params = new URLSearchParams({
      q: query,
      ...(category && { category }),
    });

    const response = await fetch(`${this.baseUrl}/api/plugins/search?${params}`);
    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    const result: MarketplaceSearchResult = await response.json();
    return result.plugins;
  }

  /**
   * Get plugin manifest
   */
  async getPluginManifest(pluginId: string, version?: string): Promise<PluginManifest> {
    const url = version
      ? `${this.baseUrl}/api/plugins/${pluginId}/${version}`
      : `${this.baseUrl}/api/plugins/${pluginId}`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch plugin ${pluginId}: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get plugin versions
   */
  async getPluginVersions(pluginId: string): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/plugins/${pluginId}/versions`);
    if (!response.ok) {
      throw new Error(`Failed to fetch versions for ${pluginId}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.versions || [];
  }

  /**
   * Get trending plugins
   */
  async getTrendingPlugins(limit: number = 10): Promise<PluginManifest[]> {
    const response = await fetch(`${this.baseUrl}/api/plugins/trending?limit=${limit}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch trending plugins: ${response.statusText}`);
    }

    const data = await response.json();
    return data.plugins || [];
  }

  /**
   * Get plugins by category
   */
  async getPluginsByCategory(category: string): Promise<PluginManifest[]> {
    const response = await fetch(`${this.baseUrl}/api/plugins/category/${category}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch plugins for category ${category}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.plugins || [];
  }

  /**
   * Get plugins by publisher
   */
  async getPluginsByPublisher(publisher: string): Promise<PluginManifest[]> {
    const response = await fetch(`${this.baseUrl}/api/publishers/${publisher}/plugins`);
    if (!response.ok) {
      throw new Error(`Failed to fetch plugins for publisher ${publisher}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.plugins || [];
  }

  /**
   * Publish a plugin
   */
  async publishPlugin(manifest: PluginManifest, packagePath: string): Promise<void> {
    const formData = new FormData();
    formData.append('manifest', JSON.stringify(manifest));

    // Read package file
    const packageBuffer = await this.readFile(packagePath);
    formData.append('package', new Blob([packageBuffer]), 'plugin.zip');

    const response = await fetch(`${this.baseUrl}/api/plugins/publish`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to publish plugin: ${error}`);
    }
  }

  /**
   * Update plugin metadata
   */
  async updatePluginMetadata(pluginId: string, updates: Partial<PluginManifest>): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/plugins/${pluginId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    });

    if (!response.ok) {
      throw new Error(`Failed to update plugin metadata: ${response.statusText}`);
    }
  }

  /**
   * Delete plugin
   */
  async deletePlugin(pluginId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/plugins/${pluginId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to delete plugin: ${response.statusText}`);
    }
  }

  /**
   * Get plugin reviews
   */
  async getPluginReviews(pluginId: string): Promise<any[]> {
    const response = await fetch(`${this.baseUrl}/api/plugins/${pluginId}/reviews`);
    if (!response.ok) {
      throw new Error(`Failed to fetch reviews for ${pluginId}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.reviews || [];
  }

  /**
   * Submit plugin review
   */
  async submitReview(pluginId: string, rating: number, comment: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/plugins/${pluginId}/reviews`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ rating, comment }),
    });

    if (!response.ok) {
      throw new Error(`Failed to submit review: ${response.statusText}`);
    }
  }

  /**
   * Get plugin download stats
   */
  async getDownloadStats(pluginId: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/plugins/${pluginId}/stats`);
    if (!response.ok) {
      throw new Error(`Failed to fetch stats for ${pluginId}: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Validate plugin manifest
   */
  validateManifest(manifest: PluginManifest): boolean {
    // Basic validation
    if (!manifest.id || !manifest.name || !manifest.version) {
      return false;
    }

    if (!manifest.capabilities || manifest.capabilities.length === 0) {
      return false;
    }

    if (!manifest.entryPoint) {
      return false;
    }

    return true;
  }

  private async readFile(filePath: string): Promise<Buffer> {
    // Simplified file reading - in a real implementation, use fs/promises
    const fs = await import('fs');
    return fs.readFileSync(filePath);
  }
}
