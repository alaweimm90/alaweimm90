import * as vscode from 'vscode';
import { AtlasAPI, AtlasTask } from './atlas-api';

export function registerCommands(context: vscode.ExtensionContext, atlasAPI: AtlasAPI | undefined) {
  // Configure API Key
  const configureApiKeyCommand = vscode.commands.registerCommand(
    'atlas.configureApiKey',
    async () => {
      const apiKey = await vscode.window.showInputBox({
        prompt: 'Enter your ATLAS API Key',
        password: true,
        placeHolder: 'Your ATLAS API key',
      });

      if (apiKey) {
        await vscode.workspace
          .getConfiguration('atlas')
          .update('apiKey', apiKey, vscode.ConfigurationTarget.Global);
        vscode.window.showInformationMessage('ATLAS: API key configured successfully!');
      }
    }
  );

  // Show Dashboard
  const showDashboardCommand = vscode.commands.registerCommand('atlas.showDashboard', () => {
    vscode.commands.executeCommand('atlas-dashboard.focus');
  });

  // View Tasks
  const viewTasksCommand = vscode.commands.registerCommand('atlas.viewTasks', () => {
    vscode.commands.executeCommand('atlasTasks.focus');
  });

  // View Agents
  const viewAgentsCommand = vscode.commands.registerCommand('atlas.viewAgents', () => {
    vscode.commands.executeCommand('atlasAgents.focus');
  });

  // Analyze Code
  const analyzeCodeCommand = vscode.commands.registerCommand('atlas.analyzeCode', async () => {
    if (!atlasAPI) {
      vscode.window.showErrorMessage('ATLAS: Please configure your API key first.');
      return;
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('ATLAS: No active editor found.');
      return;
    }

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);

    if (!selectedText) {
      vscode.window.showErrorMessage('ATLAS: Please select some code to analyze.');
      return;
    }

    try {
      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'ATLAS: Analyzing code...',
          cancellable: false,
        },
        async (progress) => {
          const task = await atlasAPI.submitTask(
            'code_review',
            `Analyze the following code for issues, improvements, and best practices:\n\n${selectedText}`,
            {
              language: editor.document.languageId,
              file_path: vscode.workspace.asRelativePath(editor.document.uri),
              selected_lines: `${selection.start.line + 1}-${selection.end.line + 1}`,
            }
          );

          // Poll for completion
          let taskResult: AtlasTask;
          do {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            taskResult = await atlasAPI.getTask(task.task_id);
            progress.report({ message: `${taskResult.status}...` });
          } while (taskResult.status === 'queued' || taskResult.status === 'running');

          if (taskResult.status === 'completed' && taskResult.result) {
            // Show results in output channel
            const outputChannel = vscode.window.createOutputChannel('ATLAS Analysis');
            outputChannel.show();
            outputChannel.appendLine('=== ATLAS Code Analysis Results ===');
            outputChannel.appendLine(`Task ID: ${taskResult.task_id}`);
            outputChannel.appendLine(`Agent: ${taskResult.agent_id}`);
            outputChannel.appendLine(`Cost: $${taskResult.cost_usd}`);
            outputChannel.appendLine('');
            outputChannel.appendLine(taskResult.result.explanation || taskResult.result.code);
            if (taskResult.result.suggestions) {
              outputChannel.appendLine('');
              outputChannel.appendLine('Suggestions:');
              taskResult.result.suggestions.forEach((suggestion: string, index: number) => {
                outputChannel.appendLine(`${index + 1}. ${suggestion}`);
              });
            }
          } else {
            vscode.window.showErrorMessage(
              `ATLAS: Analysis failed - ${taskResult.error || 'Unknown error'}`
            );
          }
        }
      );
    } catch (error) {
      vscode.window.showErrorMessage(`ATLAS: Analysis failed - ${error}`);
    }
  });

  // Generate Code
  const generateCodeCommand = vscode.commands.registerCommand('atlas.generateCode', async () => {
    if (!atlasAPI) {
      vscode.window.showErrorMessage('ATLAS: Please configure your API key first.');
      return;
    }

    const description = await vscode.window.showInputBox({
      prompt: 'Describe the code you want to generate',
      placeHolder: 'e.g., Create a REST API endpoint for user authentication',
    });

    if (!description) {
      return;
    }

    const editor = vscode.window.activeTextEditor;
    const context = editor
      ? {
          language: editor.document.languageId,
          file_path: vscode.workspace.asRelativePath(editor.document.uri),
          existing_code: editor.document.getText(),
        }
      : undefined;

    try {
      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'ATLAS: Generating code...',
          cancellable: false,
        },
        async (progress) => {
          const task = await atlasAPI.submitTask('code_generation', description, context);

          let taskResult: AtlasTask;
          do {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            taskResult = await atlasAPI.getTask(task.task_id);
            progress.report({ message: `${taskResult.status}...` });
          } while (taskResult.status === 'queued' || taskResult.status === 'running');

          if (taskResult.status === 'completed' && taskResult.result) {
            const outputChannel = vscode.window.createOutputChannel('ATLAS Generation');
            outputChannel.show();
            outputChannel.appendLine('=== ATLAS Code Generation Results ===');
            outputChannel.appendLine(`Task ID: ${taskResult.task_id}`);
            outputChannel.appendLine(`Agent: ${taskResult.agent_id}`);
            outputChannel.appendLine(`Cost: $${taskResult.cost_usd}`);
            outputChannel.appendLine('');
            outputChannel.appendLine('Generated Code:');
            outputChannel.appendLine(taskResult.result.code);
            if (taskResult.result.explanation) {
              outputChannel.appendLine('');
              outputChannel.appendLine('Explanation:');
              outputChannel.appendLine(taskResult.result.explanation);
            }

            // Offer to insert code
            const insert = await vscode.window.showInformationMessage(
              'ATLAS: Code generated successfully! Would you like to insert it into the editor?',
              'Insert Code',
              'Copy to Clipboard'
            );

            if (insert === 'Insert Code' && editor) {
              editor.edit((editBuilder) => {
                editBuilder.insert(editor.selection.active, taskResult.result.code);
              });
            } else if (insert === 'Copy to Clipboard') {
              vscode.env.clipboard.writeText(taskResult.result.code);
              vscode.window.showInformationMessage('ATLAS: Code copied to clipboard!');
            }
          } else {
            vscode.window.showErrorMessage(
              `ATLAS: Code generation failed - ${taskResult.error || 'Unknown error'}`
            );
          }
        }
      );
    } catch (error) {
      vscode.window.showErrorMessage(`ATLAS: Code generation failed - ${error}`);
    }
  });

  // Review Code
  const reviewCodeCommand = vscode.commands.registerCommand('atlas.reviewCode', async () => {
    if (!atlasAPI) {
      vscode.window.showErrorMessage('ATLAS: Please configure your API key first.');
      return;
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('ATLAS: No active editor found.');
      return;
    }

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);

    if (!selectedText) {
      vscode.window.showErrorMessage('ATLAS: Please select some code to review.');
      return;
    }

    try {
      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'ATLAS: Reviewing code...',
          cancellable: false,
        },
        async (progress) => {
          const task = await atlasAPI.submitTask(
            'code_review',
            `Review the following code for bugs, security issues, performance problems, and best practices:\n\n${selectedText}`,
            {
              language: editor.document.languageId,
              file_path: vscode.workspace.asRelativePath(editor.document.uri),
              selected_lines: `${selection.start.line + 1}-${selection.end.line + 1}`,
            }
          );

          let taskResult: AtlasTask;
          do {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            taskResult = await atlasAPI.getTask(task.task_id);
            progress.report({ message: `${taskResult.status}...` });
          } while (taskResult.status === 'queued' || taskResult.status === 'running');

          if (taskResult.status === 'completed' && taskResult.result) {
            const outputChannel = vscode.window.createOutputChannel('ATLAS Code Review');
            outputChannel.show();
            outputChannel.appendLine('=== ATLAS Code Review Results ===');
            outputChannel.appendLine(`Task ID: ${taskResult.task_id}`);
            outputChannel.appendLine(`Agent: ${taskResult.agent_id}`);
            outputChannel.appendLine(`Cost: $${taskResult.cost_usd}`);
            outputChannel.appendLine('');
            outputChannel.appendLine(taskResult.result.explanation || taskResult.result.code);
            if (taskResult.result.suggestions) {
              outputChannel.appendLine('');
              outputChannel.appendLine('Issues Found:');
              taskResult.result.suggestions.forEach((suggestion: string, index: number) => {
                outputChannel.appendLine(`${index + 1}. ${suggestion}`);
              });
            }
          } else {
            vscode.window.showErrorMessage(
              `ATLAS: Code review failed - ${taskResult.error || 'Unknown error'}`
            );
          }
        }
      );
    } catch (error) {
      vscode.window.showErrorMessage(`ATLAS: Code review failed - ${error}`);
    }
  });

  // Refactor Code
  const refactorCodeCommand = vscode.commands.registerCommand('atlas.refactorCode', async () => {
    if (!atlasAPI) {
      vscode.window.showErrorMessage('ATLAS: Please configure your API key first.');
      return;
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('ATLAS: No active editor found.');
      return;
    }

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);

    if (!selectedText) {
      vscode.window.showErrorMessage('ATLAS: Please select some code to refactor.');
      return;
    }

    try {
      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'ATLAS: Refactoring code...',
          cancellable: false,
        },
        async (progress) => {
          const task = await atlasAPI.submitTask(
            'refactoring',
            `Refactor the following code to improve readability, maintainability, and performance:\n\n${selectedText}`,
            {
              language: editor.document.languageId,
              file_path: vscode.workspace.asRelativePath(editor.document.uri),
              selected_lines: `${selection.start.line + 1}-${selection.end.line + 1}`,
            }
          );

          let taskResult: AtlasTask;
          do {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            taskResult = await atlasAPI.getTask(task.task_id);
            progress.report({ message: `${taskResult.status}...` });
          } while (taskResult.status === 'queued' || taskResult.status === 'running');

          if (taskResult.status === 'completed' && taskResult.result) {
            const outputChannel = vscode.window.createOutputChannel('ATLAS Refactoring');
            outputChannel.show();
            outputChannel.appendLine('=== ATLAS Refactoring Results ===');
            outputChannel.appendLine(`Task ID: ${taskResult.task_id}`);
            outputChannel.appendLine(`Agent: ${taskResult.agent_id}`);
            outputChannel.appendLine(`Cost: $${taskResult.cost_usd}`);
            outputChannel.appendLine('');
            outputChannel.appendLine('Refactored Code:');
            outputChannel.appendLine(taskResult.result.code);
            if (taskResult.result.explanation) {
              outputChannel.appendLine('');
              outputChannel.appendLine('Explanation:');
              outputChannel.appendLine(taskResult.result.explanation);
            }

            // Offer to replace code
            const replace = await vscode.window.showInformationMessage(
              'ATLAS: Code refactored successfully! Would you like to replace the selected code?',
              'Replace Code',
              'Copy to Clipboard'
            );

            if (replace === 'Replace Code' && editor) {
              editor.edit((editBuilder) => {
                editBuilder.replace(selection, taskResult.result.code);
              });
            } else if (replace === 'Copy to Clipboard') {
              vscode.env.clipboard.writeText(taskResult.result.code);
              vscode.window.showInformationMessage('ATLAS: Refactored code copied to clipboard!');
            }
          } else {
            vscode.window.showErrorMessage(
              `ATLAS: Refactoring failed - ${taskResult.error || 'Unknown error'}`
            );
          }
        }
      );
    } catch (error) {
      vscode.window.showErrorMessage(`ATLAS: Refactoring failed - ${error}`);
    }
  });

  // Register all commands
  context.subscriptions.push(
    configureApiKeyCommand,
    showDashboardCommand,
    viewTasksCommand,
    viewAgentsCommand,
    analyzeCodeCommand,
    generateCodeCommand,
    reviewCodeCommand,
    refactorCodeCommand
  );
}
