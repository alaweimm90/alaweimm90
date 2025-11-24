#!/usr/bin/env ts-node

/**
 * MCP Setup Script
 * Initializes MCP configuration for Claude Code, devcontainers, and local development
 */

import { MCPConfigManager, MCPCategory, MCPServerConfig } from '@monorepo/mcp-core';



async function setupMCP(): Promise<void> {
  console.log('🚀 Setting up MCP (Model Context Protocol)...\n');

  const configManager = new MCPConfigManager();

  // Default MCP servers to install
  const defaultServers: MCPServerConfig[] = [
    {
      name: 'filesystem',
      description: 'Secure file system operations',
      command: 'npx',
      args: ['@modelcontextprotocol/server-filesystem'],
      enabled: true,
      category: MCPCategory.CORE,
      version: 'latest',
    },
    {
      name: 'git',
      description: 'Git repository operations',
      command: 'npx',
      args: ['@modelcontextprotocol/server-git'],
      enabled: true,
      category: MCPCategory.CORE,
      version: 'latest',
    },
    {
      name: 'fetch',
      description: 'Web content retrieval',
      command: 'npx',
      args: ['@modelcontextprotocol/server-fetch'],
      enabled: true,
      category: MCPCategory.CORE,
      version: 'latest',
    },
  ];

  // Add default servers to local config
  console.log('📝 Configuring local MCP servers...');
  for (const server of defaultServers) {
    configManager.addServer(server, 'local');
    console.log(`  ✓ Added ${server.name}`);
  }

  // Create Claude Code configuration
  console.log('\n🎯 Setting up Claude Code configuration...');
  const claudeConfig = configManager.getLocalConfig();
  configManager.saveClaudeCodeConfig(claudeConfig);
  console.log('  ✓ Claude Code MCP config created');

  // Create devContainer configuration
  console.log('\n🐳 Setting up DevContainer configuration...');
  const devContainerConfig = configManager.getLocalConfig();
  configManager.saveDevContainerConfig(devContainerConfig);
  console.log('  ✓ DevContainer MCP config created');

  // Display merged configuration
  console.log('\n📊 Final MCP Configuration:');
  const mergedConfig = configManager.mergeConfigs();
  console.log(`  Enabled MCPs: ${mergedConfig.enabled.join(', ')}`);
  console.log(`  Total MCPs configured: ${Object.keys(mergedConfig.mcpServers).length}`);

  console.log('\n✨ MCP setup complete!\n');
  console.log('Next steps:');
  console.log('1. Install MCP servers: npm install @modelcontextprotocol/server-*');
  console.log('2. Review configuration in .claude/mcp-config.json');
  console.log('3. Start using MCPs in Claude Code!\n');
}

setupMCP().catch(error => {
  console.error('❌ Error setting up MCP:', error);
  process.exit(1);
});
