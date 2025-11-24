/**
 * Core types for MCP (Model Context Protocol) abstractions
 */

export interface MCPServerConfig {
  name: string;
  description: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  enabled: boolean;
  category: MCPCategory;
  version?: string;
  author?: string;
  repository?: string;
}

export enum MCPCategory {
  // Core/Reference
  CORE = 'core',
  REFERENCE = 'reference',

  // Data & Analytics
  DATABASE = 'database',
  ANALYTICS = 'analytics',
  VECTOR_SEARCH = 'vector_search',

  // Development
  DEVELOPMENT = 'development',
  CODE_EXECUTION = 'code_execution',
  VERSION_CONTROL = 'version_control',
  CODING_AGENT = 'coding_agent',

  // Cloud & Infrastructure
  CLOUD = 'cloud',
  INFRASTRUCTURE = 'infrastructure',

  // Communication
  COMMUNICATION = 'communication',
  COLLABORATION = 'collaboration',

  // Search & Content
  SEARCH = 'search',
  CONTENT = 'content',
  BROWSER = 'browser',

  // Security & Compliance
  SECURITY = 'security',
  MONITORING = 'monitoring',

  // Business
  CRM = 'crm',
  BUSINESS = 'business',
  PRODUCTIVITY = 'productivity',

  // Specialized
  SPECIALIZED = 'specialized',
  OTHER = 'other',
}

export interface MCPRegistry {
  version: string;
  lastUpdated: string;
  servers: MCPServerConfig[];
}

export interface MCPInstallOptions {
  name: string;
  force?: boolean;
  verbose?: boolean;
}

export interface MCPEnvironment {
  mcpServers: Record<string, MCPServerConfig>;
  enabled: string[];
  disabled: string[];
}
