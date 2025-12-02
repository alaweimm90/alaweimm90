import * as vscode from 'vscode';
import { AtlasAPI } from './atlas-api';
import { registerCommands } from './commands';
import { registerTreeViews } from './tree-views';
import { registerDashboard } from './dashboard';

let atlasAPI: AtlasAPI | undefined;

export function activate(context: vscode.ExtensionContext) {
  console.log('ATLAS extension is now active!');

  // Initialize ATLAS API client
  const apiKey = vscode.workspace.getConfiguration('atlas').get('apiKey') as string;
  const apiUrl = vscode.workspace.getConfiguration('atlas').get('apiUrl') as string;

  if (apiKey) {
    atlasAPI = new AtlasAPI(apiKey, apiUrl);
  }

  // Register all commands
  registerCommands(context, atlasAPI);

  // Register tree views for tasks and agents
  registerTreeViews(context, atlasAPI);

  // Register dashboard webview
  registerDashboard(context, atlasAPI);

  // Check for API key on activation
  if (!apiKey) {
    vscode.window
      .showInformationMessage(
        'ATLAS: Please configure your API key to use the extension.',
        'Configure API Key'
      )
      .then((selection) => {
        if (selection === 'Configure API Key') {
          vscode.commands.executeCommand('atlas.configureApiKey');
        }
      });
  }

  // Listen for configuration changes
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('atlas.apiKey') || e.affectsConfiguration('atlas.apiUrl')) {
        const newApiKey = vscode.workspace.getConfiguration('atlas').get('apiKey') as string;
        const newApiUrl = vscode.workspace.getConfiguration('atlas').get('apiUrl') as string;

        if (newApiKey && newApiKey !== apiKey) {
          atlasAPI = new AtlasAPI(newApiKey, newApiUrl);
          vscode.window.showInformationMessage('ATLAS: API key updated successfully!');
        } else if (!newApiKey) {
          atlasAPI = undefined;
          vscode.window.showWarningMessage(
            'ATLAS: API key removed. Extension functionality disabled.'
          );
        }
      }
    })
  );
}

export function deactivate() {
  console.log('ATLAS extension is now deactivated!');
}

export function getAtlasAPI(): AtlasAPI | undefined {
  return atlasAPI;
}
