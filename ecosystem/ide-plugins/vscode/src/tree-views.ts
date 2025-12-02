import * as vscode from 'vscode';
import { AtlasAPI, AtlasTask, AtlasAgent } from './atlas-api';

class AtlasTasksProvider implements vscode.TreeDataProvider<TaskItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<TaskItem | undefined | null | void> =
    new vscode.EventEmitter<TaskItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<TaskItem | undefined | null | void> =
    this._onDidChangeTreeData.fire;

  private tasks: AtlasTask[] = [];
  private refreshTimer?: NodeJS.Timeout;

  constructor(private atlasAPI?: AtlasAPI) {
    if (atlasAPI) {
      this.startAutoRefresh();
    }
  }

  private startAutoRefresh() {
    // Refresh every 30 seconds
    this.refreshTimer = setInterval(() => {
      this.refresh();
    }, 30000);
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: TaskItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: TaskItem): Promise<TaskItem[]> {
    if (!this.atlasAPI) {
      return [
        new TaskItem(
          'Configure API Key',
          'Please configure your ATLAS API key',
          vscode.TreeItemCollapsibleState.None,
          'configure'
        ),
      ];
    }

    if (!element) {
      // Root level - show task categories
      try {
        const tasksData = await this.atlasAPI.listTasks({ limit: 50 });
        this.tasks = tasksData.tasks;

        const runningTasks = this.tasks.filter((t) => t.status === 'running');
        const queuedTasks = this.tasks.filter((t) => t.status === 'queued');
        const completedTasks = this.tasks.filter((t) => t.status === 'completed');
        const failedTasks = this.tasks.filter((t) => t.status === 'failed');

        const items: TaskItem[] = [];

        if (runningTasks.length > 0) {
          items.push(
            new TaskItem(
              `Running (${runningTasks.length})`,
              '',
              vscode.TreeItemCollapsibleState.Expanded,
              'category',
              runningTasks
            )
          );
        }
        if (queuedTasks.length > 0) {
          items.push(
            new TaskItem(
              `Queued (${queuedTasks.length})`,
              '',
              vscode.TreeItemCollapsibleState.Collapsed,
              'category',
              queuedTasks
            )
          );
        }
        if (completedTasks.length > 0) {
          items.push(
            new TaskItem(
              `Completed (${completedTasks.length})`,
              '',
              vscode.TreeItemCollapsibleState.Collapsed,
              'category',
              completedTasks
            )
          );
        }
        if (failedTasks.length > 0) {
          items.push(
            new TaskItem(
              `Failed (${failedTasks.length})`,
              '',
              vscode.TreeItemCollapsibleState.Collapsed,
              'category',
              failedTasks
            )
          );
        }

        return items;
      } catch (error) {
        return [
          new TaskItem(
            'Error loading tasks',
            error.message,
            vscode.TreeItemCollapsibleState.None,
            'error'
          ),
        ];
      }
    } else if (element.type === 'category') {
      // Show tasks in this category
      return element.tasks.map(
        (task) =>
          new TaskItem(
            `${task.type}: ${task.description.substring(0, 50)}...`,
            `ID: ${task.task_id} | Agent: ${task.agent_id || 'N/A'} | Cost: $${task.cost_usd || 0}`,
            vscode.TreeItemCollapsibleState.None,
            'task',
            [],
            task
          )
      );
    }

    return [];
  }

  dispose() {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
    }
  }
}

class AtlasAgentsProvider implements vscode.TreeDataProvider<AgentItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<AgentItem | undefined | null | void> =
    new vscode.EventEmitter<AgentItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<TaskItem | undefined | null | void> =
    this._onDidChangeTreeData.fire;

  private agents: AtlasAgent[] = [];
  private refreshTimer?: NodeJS.Timeout;

  constructor(private atlasAPI?: AtlasAPI) {
    if (atlasAPI) {
      this.startAutoRefresh();
    }
  }

  private startAutoRefresh() {
    // Refresh every 60 seconds
    this.refreshTimer = setInterval(() => {
      this.refresh();
    }, 60000);
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: AgentItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: AgentItem): Promise<AgentItem[]> {
    if (!this.atlasAPI) {
      return [
        new AgentItem(
          'Configure API Key',
          'Please configure your ATLAS API key',
          vscode.TreeItemCollapsibleState.None,
          'configure'
        ),
      ];
    }

    if (!element) {
      // Root level - show agents grouped by provider
      try {
        const agentsData = await this.atlasAPI.listAgents();
        this.agents = agentsData.agents;

        const providers = [...new Set(this.agents.map((a) => a.provider))];
        return providers.map((provider) => {
          const providerAgents = this.agents.filter((a) => a.provider === provider);
          return new AgentItem(
            `${provider.toUpperCase()} (${providerAgents.length})`,
            '',
            vscode.TreeItemCollapsibleState.Expanded,
            'provider',
            providerAgents
          );
        });
      } catch (error) {
        return [
          new AgentItem(
            'Error loading agents',
            error.message,
            vscode.TreeItemCollapsibleState.None,
            'error'
          ),
        ];
      }
    } else if (element.type === 'provider') {
      // Show agents for this provider
      return element.agents.map((agent) => {
        const icon =
          agent.health.status === 'healthy' ? '✓' : agent.health.status === 'degraded' ? '⚠' : '✗';
        return new AgentItem(
          `${icon} ${agent.name}`,
          `${agent.model} | Success: ${(agent.performance.success_rate * 100).toFixed(1)}% | Tasks: ${agent.performance.total_tasks_completed}`,
          vscode.TreeItemCollapsibleState.None,
          'agent',
          [],
          agent
        );
      });
    }

    return [];
  }

  dispose() {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
    }
  }
}

class TaskItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly tooltip: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly type: string,
    public readonly tasks: AtlasTask[] = [],
    public readonly task?: AtlasTask
  ) {
    super(label, collapsibleState);
    this.tooltip = tooltip;

    // Set icons based on type
    if (type === 'category') {
      this.iconPath = new vscode.ThemeIcon('folder');
    } else if (type === 'task') {
      if (task?.status === 'running') {
        this.iconPath = new vscode.ThemeIcon('sync~spin');
      } else if (task?.status === 'completed') {
        this.iconPath = new vscode.ThemeIcon('check');
      } else if (task?.status === 'failed') {
        this.iconPath = new vscode.ThemeIcon('error');
      } else if (task?.status === 'queued') {
        this.iconPath = new vscode.ThemeIcon('clock');
      }
    } else if (type === 'configure') {
      this.iconPath = new vscode.ThemeIcon('gear');
      this.command = {
        command: 'atlas.configureApiKey',
        title: 'Configure API Key',
      };
    }

    // Set context value for menus
    this.contextValue = type;

    // Add command for tasks
    if (type === 'task' && task) {
      this.command = {
        command: 'atlas.showTaskDetails',
        title: 'Show Task Details',
        arguments: [task],
      };
    }
  }
}

class AgentItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly tooltip: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly type: string,
    public readonly agents: AtlasAgent[] = [],
    public readonly agent?: AtlasAgent
  ) {
    super(label, collapsibleState);
    this.tooltip = tooltip;

    // Set icons based on type
    if (type === 'provider') {
      this.iconPath = new vscode.ThemeIcon('package');
    } else if (type === 'agent') {
      this.iconPath = new vscode.ThemeIcon('robot');
    } else if (type === 'configure') {
      this.iconPath = new vscode.ThemeIcon('gear');
      this.command = {
        command: 'atlas.configureApiKey',
        title: 'Configure API Key',
      };
    }

    // Set context value for menus
    this.contextValue = type;

    // Add command for agents
    if (type === 'agent' && agent) {
      this.command = {
        command: 'atlas.showAgentDetails',
        title: 'Show Agent Details',
        arguments: [agent],
      };
    }
  }
}

let tasksProvider: AtlasTasksProvider;
let agentsProvider: AtlasAgentsProvider;

export function registerTreeViews(
  context: vscode.ExtensionContext,
  atlasAPI: AtlasAPI | undefined
) {
  tasksProvider = new AtlasTasksProvider(atlasAPI);
  agentsProvider = new AtlasAgentsProvider(atlasAPI);

  // Register tree data providers
  vscode.window.registerTreeDataProvider('atlasTasks', tasksProvider);
  vscode.window.registerTreeDataProvider('atlasAgents', agentsProvider);

  // Register commands for tree items
  const showTaskDetailsCommand = vscode.commands.registerCommand(
    'atlas.showTaskDetails',
    (task: AtlasTask) => {
      const outputChannel = vscode.window.createOutputChannel(`ATLAS Task ${task.task_id}`);
      outputChannel.show();
      outputChannel.appendLine(`=== ATLAS Task Details ===`);
      outputChannel.appendLine(`Task ID: ${task.task_id}`);
      outputChannel.appendLine(`Type: ${task.type}`);
      outputChannel.appendLine(`Status: ${task.status}`);
      outputChannel.appendLine(`Description: ${task.description}`);
      outputChannel.appendLine(`Agent: ${task.agent_id || 'Not assigned'}`);
      outputChannel.appendLine(`Created: ${new Date(task.created_at).toLocaleString()}`);
      if (task.completed_at) {
        outputChannel.appendLine(`Completed: ${new Date(task.completed_at).toLocaleString()}`);
      }
      if (task.cost_usd) {
        outputChannel.appendLine(`Cost: $${task.cost_usd}`);
      }
      if (task.result) {
        outputChannel.appendLine('');
        outputChannel.appendLine('Result:');
        outputChannel.appendLine(
          typeof task.result === 'string' ? task.result : JSON.stringify(task.result, null, 2)
        );
      }
      if (task.error) {
        outputChannel.appendLine('');
        outputChannel.appendLine(`Error: ${task.error}`);
      }
    }
  );

  const showAgentDetailsCommand = vscode.commands.registerCommand(
    'atlas.showAgentDetails',
    (agent: AtlasAgent) => {
      const outputChannel = vscode.window.createOutputChannel(`ATLAS Agent ${agent.agent_id}`);
      outputChannel.show();
      outputChannel.appendLine(`=== ATLAS Agent Details ===`);
      outputChannel.appendLine(`Agent ID: ${agent.agent_id}`);
      outputChannel.appendLine(`Name: ${agent.name}`);
      outputChannel.appendLine(`Provider: ${agent.provider}`);
      outputChannel.appendLine(`Model: ${agent.model}`);
      outputChannel.appendLine(`Status: ${agent.status}`);
      outputChannel.appendLine(`Capabilities: ${agent.capabilities.join(', ')}`);
      outputChannel.appendLine('');
      outputChannel.appendLine('Health:');
      outputChannel.appendLine(`  Status: ${agent.health.status}`);
      outputChannel.appendLine(`  Uptime: ${(agent.health.uptime_percentage * 100).toFixed(1)}%`);
      outputChannel.appendLine(`  Response Time: ${agent.health.avg_response_time_ms}ms`);
      outputChannel.appendLine('');
      outputChannel.appendLine('Performance:');
      outputChannel.appendLine(
        `  Success Rate: ${(agent.performance.success_rate * 100).toFixed(1)}%`
      );
      outputChannel.appendLine(
        `  Quality Score: ${agent.performance.avg_quality_score.toFixed(1)}`
      );
      outputChannel.appendLine(`  Total Tasks: ${agent.performance.total_tasks_completed}`);
    }
  );

  // Register refresh commands
  const refreshTasksCommand = vscode.commands.registerCommand('atlas.refreshTasks', () => {
    tasksProvider.refresh();
  });

  const refreshAgentsCommand = vscode.commands.registerCommand('atlas.refreshAgents', () => {
    agentsProvider.refresh();
  });

  context.subscriptions.push(
    showTaskDetailsCommand,
    showAgentDetailsCommand,
    refreshTasksCommand,
    refreshAgentsCommand
  );

  // Listen for API key changes to update providers
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('atlas.apiKey') || e.affectsConfiguration('atlas.apiUrl')) {
        const apiKey = vscode.workspace.getConfiguration('atlas').get('apiKey') as string;
        const apiUrl = vscode.workspace.getConfiguration('atlas').get('apiUrl') as string;

        if (apiKey) {
          const newAtlasAPI = new AtlasAPI(apiKey, apiUrl);
          tasksProvider = new AtlasTasksProvider(newAtlasAPI);
          agentsProvider = new AtlasAgentsProvider(newAtlasAPI);
          vscode.window.registerTreeDataProvider('atlasTasks', tasksProvider);
          vscode.window.registerTreeDataProvider('atlasAgents', agentsProvider);
        }
      }
    })
  );
}
