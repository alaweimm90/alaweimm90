#!/usr/bin/env node

/**
 * {{project-name}} MCP Server
 * {{description}}
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

// Server capabilities
const SERVER_INFO = {
  name: '{{project-name}}',
  version: '0.1.0',
};

// Available tools
const TOOLS = [
  {
    name: '{{tool-name}}',
    description: '{{tool-description}}',
    inputSchema: {
      type: 'object',
      properties: {
        // Add your tool input properties here
      },
      required: [],
    },
  },
];

class {{project-name-class}}Server {
  private server: Server;

  constructor() {
    this.server = new Server(SERVER_INFO, {
      capabilities: {
        tools: {},
      },
    });

    this.setupHandlers();
  }

  private setupHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return { tools: TOOLS };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case '{{tool-name}}':
          return await this.handle{{tool-name-class}}Tool(args);
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  private async handle{{tool-name-class}}Tool(args: any) {
    try {
      // Implement your tool logic here
      console.log('{{tool-name}} called with args:', args);

      return {
        content: [
          {
            type: 'text',
            text: `{{tool-name}} executed successfully with result: ${JSON.stringify(args)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error executing {{tool-name}}: ${error}`,
          },
        ],
        isError: true,
      };
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('{{project-name}} MCP server running on stdio');
  }
}

// Run the server
const server = new {{project-name-class}}Server();
server.run().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
