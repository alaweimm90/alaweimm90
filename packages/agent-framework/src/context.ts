/**
 * Context Provider - Manages shared context for agents and MCPs
 */

import { AgentContext } from './types';
import * as path from 'path';
import * as os from 'os';

export class ContextProvider {
  private static instance: ContextProvider;

  private context: AgentContext;

  private metadata: Map<string, unknown> = new Map();

  private constructor() {
    this.context = this.initializeContext();
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): ContextProvider {
    if (!ContextProvider.instance) {
      ContextProvider.instance = new ContextProvider();
    }
    return ContextProvider.instance;
  }

  /**
   * Initialize context from environment
   */
  private initializeContext(): AgentContext {
    const workspaceRoot = process.env.WORKSPACE_ROOT || process.cwd();
    const projectName = path.basename(workspaceRoot);

    return {
      workspaceRoot,
      projectName,
      environment: process.env,
      mcpServers: this.getMCPServers(),
      metadata: {},
    };
  }

  /**
   * Get list of available MCP servers
   */
  private getMCPServers(): string[] {
    const mcpsEnv = process.env.MCP_SERVERS || '';
    return mcpsEnv.split(',').filter(s => s.trim());
  }

  /**
   * Get current context
   */
  public getContext(): AgentContext {
    return this.context;
  }

  /**
   * Update context
   * @param partial
   */
  public updateContext(partial: Partial<AgentContext>): void {
    this.context = { ...this.context, ...partial };
  }

  /**
   * Set metadata
   * @param key
   * @param value
   */
  public setMetadata(key: string, value: unknown): void {
    this.metadata.set(key, value);
    if (!this.context.metadata) {
      this.context.metadata = {};
    }
    this.context.metadata[key] = value;
  }

  /**
   * Get metadata
   * @param key
   */
  public getMetadata(key: string): unknown {
    return this.metadata.get(key);
  }

  /**
   * Get all metadata
   */
  public getAllMetadata(): Record<string, unknown> {
    return Object.fromEntries(this.metadata);
  }

  /**
   * Get workspace root
   */
  public getWorkspaceRoot(): string {
    return this.context.workspaceRoot;
  }

  /**
   * Get project name
   */
  public getProjectName(): string {
    return this.context.projectName;
  }

  /**
   * Get environment variable
   * @param name
   */
  public getEnvVar(name: string): string | undefined {
    return this.context.environment[name];
  }

  /**
   * Set environment variable
   * @param name
   * @param value
   */
  public setEnvVar(name: string, value: string): void {
    this.context.environment[name] = value;
    process.env[name] = value;
  }

  /**
   * Get user home directory
   */
  public getHomeDirectory(): string {
    return os.homedir();
  }

  /**
   * Get platform
   */
  public getPlatform(): NodeJS.Platform {
    return process.platform;
  }

  /**
   * Resolve path relative to workspace root
   * @param {...any} segments
   */
  public resolvePath(...segments: string[]): string {
    return path.join(this.context.workspaceRoot, ...segments);
  }

  /**
   * Create a sub-context for a specific task
   * @param taskId
   * @param taskData
   */
  public createSubContext(taskId: string, taskData: Record<string, unknown>): AgentContext {
    return {
      ...this.context,
      metadata: {
        ...this.context.metadata,
        taskId,
        ...taskData,
      },
    };
  }
}
