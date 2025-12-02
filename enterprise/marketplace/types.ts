// ATLAS Enterprise - Agent Marketplace Types

export interface AgentPlugin {
  id: string;
  name: string;
  version: string;
  description: string;
  capabilities: PluginCapability[];
  publisher: string;
  entryPoint: string;
  config?: PluginConfig;
  metadata: PluginMetadata;
  instance?: any; // The loaded plugin instance
}

export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  description: string;
  publisher: string;
  homepage?: string;
  repository?: string;
  license: string;
  capabilities: PluginCapability[];
  entryPoint: string;
  dependencies?: Record<string, string>;
  configSchema?: any;
  metadata: PluginMetadata;
  security: PluginSecurity;
}

export interface PluginMetadata {
  tags: string[];
  category: PluginCategory;
  minAtlasVersion: string;
  maxAtlasVersion?: string;
  supportedLanguages?: string[];
  pricing?: PluginPricing;
  stats: PluginStats;
}

export interface PluginStats {
  downloads: number;
  rating: number;
  reviews: number;
  lastUpdated: string;
  verified: boolean;
}

export interface PluginSecurity {
  signature?: string;
  checksum: string;
  permissions: PluginPermission[];
  sandboxed: boolean;
}

export interface PluginConfig {
  enabled: boolean;
  settings: Record<string, any>;
  permissions: PluginPermission[];
}

export interface PluginPricing {
  model: 'free' | 'freemium' | 'paid' | 'subscription';
  price?: number;
  currency?: string;
  trialDays?: number;
}

export type PluginCapability =
  | 'code_generation'
  | 'code_review'
  | 'refactoring'
  | 'testing'
  | 'documentation'
  | 'debugging'
  | 'optimization'
  | 'security_analysis'
  | 'performance_analysis'
  | 'deployment'
  | 'monitoring'
  | 'custom';

export type PluginCategory =
  | 'development'
  | 'testing'
  | 'security'
  | 'performance'
  | 'deployment'
  | 'monitoring'
  | 'integration'
  | 'utilities'
  | 'ai_ml'
  | 'custom';

export type PluginPermission =
  | 'read_files'
  | 'write_files'
  | 'execute_commands'
  | 'network_access'
  | 'system_info'
  | 'user_interaction'
  | 'external_apis';

export interface MarketplaceConfig {
  registryUrl: string;
  trustedPublishers: string[];
  autoUpdate: boolean;
  securityScan: boolean;
  cacheDir: string;
  timeout: number;
}

export interface PluginInstallation {
  pluginId: string;
  version: string;
  installedAt: string;
  checksum: string;
  config: PluginConfig;
}

export interface MarketplaceSearchResult {
  plugins: PluginManifest[];
  total: number;
  page: number;
  pageSize: number;
  query: string;
  filters: SearchFilters;
}

export interface SearchFilters {
  category?: PluginCategory;
  capability?: PluginCapability;
  publisher?: string;
  minRating?: number;
  pricing?: PluginPricing['model'];
  verified?: boolean;
  tags?: string[];
}

export interface PluginReview {
  pluginId: string;
  userId: string;
  rating: number;
  comment: string;
  createdAt: string;
  verified: boolean;
}

export interface PluginHook {
  name: string;
  handler: (...args: any[]) => any;
  priority: number;
}

export interface PluginContext {
  atlas: any; // ATLAS core instance
  config: PluginConfig;
  logger: any;
  storage: any;
}
