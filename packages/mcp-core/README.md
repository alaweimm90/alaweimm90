# @monorepo/mcp-core

Core abstractions for MCP (Model Context Protocol) servers and configuration management.

## Features

- MCP server registry and management
- Configuration loading and merging
- Support for multiple configuration sources (local, devcontainer, Claude Code)
- Environment-aware path handling

## Installation

```bash
pnpm install @monorepo/mcp-core
```

## Usage

### MCP Registry Manager

```typescript
import { MCPRegistryManager, MCPCategory } from '@monorepo/mcp-core';

const registry = new MCPRegistryManager();

// Get a specific MCP
const filesystem = registry.getServer('filesystem');

// Get all MCPs
const all = registry.getServers();

// Get MCPs by category
const databases = registry.getServersByCategory(MCPCategory.DATABASE);

// Get enabled MCPs
const enabled = registry.getEnabledServers();

// Register a new MCP
registry.registerServer({
  name: 'my-mcp',
  description: 'My custom MCP',
  command: 'npx',
  args: ['@custom/mcp-server'],
  enabled: true,
  category: MCPCategory.CUSTOM,
});

// Enable/disable MCPs
registry.enableServer('github');
registry.disableServer('postgres');
```

### MCP Config Manager

```typescript
import { MCPConfigManager } from '@monorepo/mcp-core';

const config = new MCPConfigManager();

// Get Claude Code configuration
const claudeConfig = config.getClaudeCodeConfig();

// Get devContainer configuration
const devConfig = config.getDevContainerConfig();

// Get local configuration
const localConfig = config.getLocalConfig();

// Merge configurations (local overrides others)
const merged = config.mergeConfigs();

// Add a server to local config
config.addServer(serverConfig, 'local');

// Remove a server
config.removeServer('github', 'local');

// Save configuration
config.saveClaudeCodeConfig(updatedConfig);
```

## Configuration Format

```json
{
  "mcpServers": {
    "filesystem": {
      "name": "filesystem",
      "description": "File operations",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem"],
      "enabled": true,
      "category": "core",
      "version": "latest"
    }
  },
  "enabled": ["filesystem", "git"],
  "disabled": ["github"]
}
```

## Types

### MCPServerConfig

```typescript
interface MCPServerConfig {
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
```

### MCPCategory

```typescript
enum MCPCategory {
  CORE = 'core',
  DATABASE = 'database',
  DEVELOPMENT = 'development',
  CLOUD = 'cloud',
  COMMUNICATION = 'communication',
  SEARCH = 'search',
  SECURITY = 'security',
  // ... more categories
}
```

## Configuration Locations

- **Claude Code**: `~/.config/trae/claude-code-mcp.json` (Linux/Mac) or `%APPDATA%/Trae/User/claude-code-mcp.json` (Windows)
- **DevContainer**: `.devcontainer/mcp-config.json`
- **Local Project**: `.claude/mcp-config.json`

## Best Practices

1. **Store base configuration in `.claude/mcp-config.json`**
2. **Use category filtering for UI/discovery**
3. **Keep enabled list synchronized**
4. **Validate configuration before saving**
5. **Use environment variables for secrets**

## See Also

- [MCP_AGENTS_ORCHESTRATION.md](../../docs/MCP_AGENTS_ORCHESTRATION.md)
- [MCP_SERVERS_GUIDE.md](../../docs/MCP_SERVERS_GUIDE.md)
