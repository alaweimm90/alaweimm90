#!/usr/bin/env node

/**
 * MCP Setup Script - Initialize MCP configuration for the project
 * Usage: node scripts/mcp-setup.js [options]
 *
 * Options:
 *   --install-all     Install all recommended MCPs
 *   --interactive     Interactive mode to select MCPs
 *   --devcontainer    Configure for devcontainer
 *   --local           Configure for local development
 *   --verbose         Verbose output
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class MCPSetup {
  constructor(options = {}) {
    this.options = options;
    this.rootDir = process.cwd();
    this.claudeDir = path.join(this.rootDir, '.claude');
    this.devcontainerDir = path.join(this.rootDir, '.devcontainer');
    this.log(`Initializing MCP Setup in ${this.rootDir}`);
  }

  log(message) {
    if (this.options.verbose || !message.startsWith('[DEBUG]')) {
      console.log(`[MCP Setup] ${message}`);
    }
  }

  error(message) {
    console.error(`[MCP Setup ERROR] ${message}`);
  }

  /**
   * Ensure .claude directory exists
   */
  ensureClaudeDir() {
    if (!fs.existsSync(this.claudeDir)) {
      fs.mkdirSync(this.claudeDir, { recursive: true });
      this.log(`Created .claude directory`);
    }
  }

  /**
   * Get MCP server definitions
   */
  getMCPServers() {
    return {
      // Core servers
      filesystem: {
        name: 'filesystem',
        description: 'Secure file system operations',
        command: 'npx',
        args: ['@modelcontextprotocol/server-filesystem'],
        category: 'core',
        recommended: true,
        essential: true,
      },
      git: {
        name: 'git',
        description: 'Git repository operations',
        command: 'npx',
        args: ['@modelcontextprotocol/server-git'],
        category: 'core',
        recommended: true,
        essential: true,
      },
      fetch: {
        name: 'fetch',
        description: 'Web content retrieval',
        command: 'npx',
        args: ['@modelcontextprotocol/server-fetch'],
        category: 'core',
        recommended: true,
        essential: false,
      },

      // Development servers
      github: {
        name: 'github',
        description: 'GitHub API integration',
        command: 'npx',
        args: ['@modelcontextprotocol/server-github'],
        category: 'development',
        recommended: true,
        essential: false,
      },

      // Database servers
      postgres: {
        name: 'postgres',
        description: 'PostgreSQL database access',
        command: 'npx',
        args: ['@modelcontextprotocol/server-postgres'],
        category: 'database',
        recommended: false,
        essential: false,
      },

      // Search servers
      'brave-search': {
        name: 'brave-search',
        description: 'Brave Search integration',
        command: 'npx',
        args: ['@modelcontextprotocol/server-brave-search'],
        category: 'search',
        recommended: false,
        essential: false,
      },
    };
  }

  /**
   * Create initial MCP config
   */
  createMCPConfig() {
    const config = {
      mcpServers: {},
      enabled: [],
      disabled: [],
    };

    const servers = this.getMCPServers();

    // Add all servers to config
    Object.values(servers).forEach(server => {
      config.mcpServers[server.name] = {
        name: server.name,
        description: server.description,
        command: server.command,
        args: server.args,
        enabled: server.recommended,
        category: server.category,
        version: 'latest',
      };

      if (server.recommended) {
        config.enabled.push(server.name);
      } else {
        config.disabled.push(server.name);
      }
    });

    return config;
  }

  /**
   * Write MCP config
   * @param config
   */
  writeMCPConfig(config) {
    const configPath = path.join(this.claudeDir, 'mcp-config.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
    this.log(`Created MCP configuration at ${configPath}`);
  }

  /**
   * Install MCP packages
   */
  installMCPPackages() {
    const servers = this.getMCPServers();
    const toInstall = Object.values(servers).filter(s => s.recommended || this.options.installAll);

    if (toInstall.length === 0) {
      this.log('No MCP packages to install');
      return;
    }

    this.log(`Installing ${toInstall.length} MCP packages...`);

    try {
      for (const server of toInstall) {
        this.log(`Installing ${server.name}...`);
        const npmPackage = `@modelcontextprotocol/server-${server.name}`;
        execSync(`npm install -g ${npmPackage}`, { stdio: 'inherit' });
      }
      this.log('MCP packages installed successfully');
    } catch (error) {
      this.error(`Failed to install MCP packages: ${error.message}`);
      throw error;
    }
  }

  /**
   * Create agents configuration
   */
  createAgentsConfig() {
    return {
      agents: [
        {
          id: 'code-agent',
          name: 'Code Agent',
          description: 'Code manipulation and analysis',
          version: '1.0.0',
          type: 'code',
          capabilities: ['code-review', 'code-fix', 'refactor'],
          requiredMcps: ['filesystem', 'git'],
          enabled: true,
        },
        {
          id: 'analysis-agent',
          name: 'Analysis Agent',
          description: 'Code analysis and testing',
          version: '1.0.0',
          type: 'analysis',
          capabilities: ['analyze', 'test', 'lint', 'security-scan'],
          requiredMcps: ['filesystem', 'git'],
          enabled: true,
        },
      ],
    };
  }

  /**
   * Write agents configuration
   * @param config
   */
  writeAgentsConfig(config) {
    const configPath = path.join(this.claudeDir, 'agents.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
    this.log(`Created agents configuration at ${configPath}`);
  }

  /**
   * Create orchestration rules
   */
  createOrchestrationRules() {
    return {
      version: '1.0.0',
      rules: [
        {
          id: 'code-review-rule',
          name: 'Code Review',
          description: 'Trigger code review on changes',
          trigger: 'pull_request_opened',
          actions: [
            {
              type: 'execute',
              target: 'code-review-workflow',
            },
          ],
          enabled: true,
        },
      ],
    };
  }

  /**
   * Write orchestration rules
   * @param config
   */
  writeOrchestrationRules(config) {
    const configPath = path.join(this.claudeDir, 'orchestration.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
    this.log(`Created orchestration rules at ${configPath}`);
  }

  /**
   * Run full setup
   */
  async run() {
    try {
      this.log('Starting MCP setup...');

      // Create directories
      this.ensureClaudeDir();

      // Create configurations
      const mcpConfig = this.createMCPConfig();
      this.writeMCPConfig(mcpConfig);

      const agentsConfig = this.createAgentsConfig();
      this.writeAgentsConfig(agentsConfig);

      const orchestrationRules = this.createOrchestrationRules();
      this.writeOrchestrationRules(orchestrationRules);

      // Install packages (optional)
      if (this.options.installPackages) {
        this.installMCPPackages();
      }

      this.log('MCP setup completed successfully!');
      this.printSummary();
    } catch (error) {
      this.error(`Setup failed: ${error.message}`);
      process.exit(1);
    }
  }

  /**
   * Print summary
   */
  printSummary() {
    console.log('\n--- MCP Setup Summary ---');
    console.log(`Configuration directory: ${this.claudeDir}`);
    console.log('Created files:');
    console.log('  - mcp-config.json (MCP server configuration)');
    console.log('  - agents.json (Agent definitions)');
    console.log('  - orchestration.json (Orchestration rules)');
    console.log('\nNext steps:');
    console.log('1. Install MCP packages: npm run mcp:install');
    console.log('2. View configuration: npm run mcp:status');
    console.log('3. Enable/disable MCPs: npm run mcp:configure');
    console.log('4. Read documentation: docs/MCP_SETUP.md');
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  verbose: args.includes('--verbose'),
  installPackages: args.includes('--install') || args.includes('--install-all'),
  interactive: args.includes('--interactive'),
};

// Run setup
const setup = new MCPSetup(options);
setup.run().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
