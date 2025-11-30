import * as vscode from 'vscode';
import { AtlasAPI, AtlasMetrics, AtlasTask } from './atlas-api';

class AtlasDashboardProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'atlas-dashboard';
    private _view?: vscode.WebviewView;

    constructor(private readonly _extensionUri: vscode.Uri, private atlasAPI?: AtlasAPI) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.type) {
                    case 'refresh':
                        await this.refresh();
                        break;
                    case 'submitTask':
                        await this.handleSubmitTask(message.data);
                        break;
                    case 'cancelTask':
                        await this.handleCancelTask(message.taskId);
                        break;
                }
            },
            undefined,
            context.subscriptions
        );

        // Initial load
        this.refresh();
    }

    private async refresh() {
        if (!this.atlasAPI || !this._view) {
            this._view?.webview.postMessage({
                type: 'update',
                data: { configured: false }
            });
            return;
        }

        try {
            const [metrics, tasks, health] = await Promise.all([
                this.atlasAPI.getMetrics('24h'),
                this.atlasAPI.listTasks({ limit: 10 }),
                this.atlasAPI.getHealth()
            ]);

            this._view.webview.postMessage({
                type: 'update',
                data: {
                    configured: true,
                    metrics,
                    recentTasks: tasks.tasks,
                    health
                }
            });
        } catch (error) {
            this._view?.webview.postMessage({
                type: 'error',
                data: { message: error.message }
            });
        }
    }

    private async handleSubmitTask(data: any) {
        if (!this.atlasAPI) return;

        try {
            const task = await this.atlasAPI.submitTask(data.type, data.description, data.context);
            vscode.window.showInformationMessage(`ATLAS: Task submitted - ${task.task_id}`);
            await this.refresh();
        } catch (error) {
            vscode.window.showErrorMessage(`ATLAS: Failed to submit task - ${error.message}`);
        }
    }

    private async handleCancelTask(taskId: string) {
        if (!this.atlasAPI) return;

        try {
            await this.atlasAPI.cancelTask(taskId);
            vscode.window.showInformationMessage(`ATLAS: Task cancelled - ${taskId}`);
            await this.refresh();
        } catch (error) {
            vscode.window.showErrorMessage(`ATLAS: Failed to cancel task - ${error.message}`);
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const nonce = getNonce();

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ATLAS Dashboard</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    font-size: var(--vscode-font-size);
                    background-color: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                    margin: 0;
                    padding: 10px;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }
                .section {
                    background: var(--vscode-editorWidget-background);
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 3px;
                    padding: 10px;
                }
                .section h3 {
                    margin: 0 0 10px 0;
                    color: var(--vscode-textLink-foreground);
                    border-bottom: 1px solid var(--vscode-panel-border);
                    padding-bottom: 5px;
                }
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 10px;
                }
                .metric {
                    text-align: center;
                    padding: 8px;
                    background: var(--vscode-input-background);
                    border-radius: 3px;
                }
                .metric-value {
                    font-size: 1.5em;
                    font-weight: bold;
                    color: var(--vscode-textLink-foreground);
                }
                .metric-label {
                    font-size: 0.8em;
                    color: var(--vscode-descriptionForeground);
                }
                .task-list {
                    max-height: 200px;
                    overflow-y: auto;
                }
                .task-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 5px;
                    border-bottom: 1px solid var(--vscode-list-inactiveSelectionBackground);
                }
                .task-info {
                    flex: 1;
                }
                .task-type {
                    font-weight: bold;
                    color: var(--vscode-textLink-foreground);
                }
                .task-desc {
                    font-size: 0.9em;
                    color: var(--vscode-descriptionForeground);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    max-width: 200px;
                }
                .task-status {
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 0.8em;
                    font-weight: bold;
                }
                .status-running { background: var(--vscode-charts-yellow); color: black; }
                .status-completed { background: var(--vscode-charts-green); color: white; }
                .status-failed { background: var(--vscode-charts-red); color: white; }
                .status-queued { background: var(--vscode-charts-blue); color: white; }
                .button {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    padding: 5px 10px;
                    border-radius: 3px;
                    cursor: pointer;
                    font-size: 0.9em;
                }
                .button:hover {
                    background: var(--vscode-button-hoverBackground);
                }
                .button.secondary {
                    background: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                }
                .form-group {
                    margin-bottom: 10px;
                }
                .form-group label {
                    display: block;
                    margin-bottom: 3px;
                    font-weight: bold;
                }
                .form-group select,
                .form-group textarea {
                    width: 100%;
                    padding: 5px;
                    background: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 3px;
                }
                .not-configured {
                    text-align: center;
                    padding: 20px;
                    color: var(--vscode-descriptionForeground);
                }
                .error {
                    color: var(--vscode-errorForeground);
                    background: var(--vscode-inputValidation-errorBackground);
                    border: 1px solid var(--vscode-inputValidation-errorBorder);
                    padding: 10px;
                    border-radius: 3px;
                    margin: 10px 0;
                }
                .health-status {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                .health-indicator {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                }
                .health-healthy { background: var(--vscode-charts-green); }
                .health-degraded { background: var(--vscode-charts-yellow); }
                .health-unhealthy { background: var(--vscode-charts-red); }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="section">
                    <h3>ATLAS Dashboard</h3>
                    <div id="content">
                        <div class="not-configured">
                            <p>ATLAS API key not configured.</p>
                            <button class="button" onclick="configureApiKey()">Configure API Key</button>
                        </div>
                    </div>
                </div>
            </div>

            <script nonce="${nonce}">
                const vscode = acquireVsCodeApi();

                function configureApiKey() {
                    vscode.postMessage({ type: 'configureApiKey' });
                }

                function refresh() {
                    vscode.postMessage({ type: 'refresh' });
                }

                function submitTask() {
                    const type = document.getElementById('task-type').value;
                    const description = document.getElementById('task-description').value;
                    if (description.trim()) {
                        vscode.postMessage({
                            type: 'submitTask',
                            data: { type, description }
                        });
                        document.getElementById('task-description').value = '';
                    }
                }

                function cancelTask(taskId) {
                    vscode.postMessage({ type: 'cancelTask', taskId });
                }

                window.addEventListener('message', event => {
                    const message = event.data;

                    switch (message.type) {
                        case 'update':
                            updateDashboard(message.data);
                            break;
                        case 'error':
                            showError(message.data.message);
                            break;
                    }
                });

                function updateDashboard(data) {
                    const content = document.getElementById('content');

                    if (!data.configured) {
                        content.innerHTML = \`
                            <div class="not-configured">
                                <p>ATLAS API key not configured.</p>
                                <button class="button" onclick="configureApiKey()">Configure API Key</button>
                            </div>
                        \`;
                        return;
                    }

                    const metrics = data.metrics;
                    const tasks = data.recentTasks || [];
                    const health = data.health;

                    content.innerHTML = \`
                        <div class="section">
                            <h3>System Health</h3>
                            <div class="health-status">
                                <div class="health-indicator health-\${health.status === 'healthy' ? 'healthy' : health.status === 'degraded' ? 'degraded' : 'unhealthy'}"></div>
                                <span>Status: \${health.status}</span>
                                <span>•</span>
                                <span>Uptime: \${(health.uptime_seconds / 86400 * 100).toFixed(1)}%</span>
                            </div>
                        </div>

                        <div class="section">
                            <h3>24h Metrics</h3>
                            <div class="metric-grid">
                                <div class="metric">
                                    <div class="metric-value">\${metrics.total_tasks}</div>
                                    <div class="metric-label">Tasks</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">\${(metrics.success_rate * 100).toFixed(1)}%</div>
                                    <div class="metric-label">Success Rate</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">\${(metrics.avg_duration_ms / 1000).toFixed(1)}s</div>
                                    <div class="metric-label">Avg Duration</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">$\${metrics.total_cost_usd.toFixed(2)}</div>
                                    <div class="metric-label">Total Cost</div>
                                </div>
                            </div>
                        </div>

                        <div class="section">
                            <h3>Quick Task</h3>
                            <div class="form-group">
                                <label for="task-type">Task Type:</label>
                                <select id="task-type">
                                    <option value="code_generation">Code Generation</option>
                                    <option value="code_review">Code Review</option>
                                    <option value="refactoring">Refactoring</option>
                                    <option value="debugging">Debugging</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="task-description">Description:</label>
                                <textarea id="task-description" rows="3" placeholder="Describe what you want ATLAS to do..."></textarea>
                            </div>
                            <button class="button" onclick="submitTask()">Submit Task</button>
                        </div>

                        <div class="section">
                            <h3>Recent Tasks</h3>
                            <div class="task-list" id="task-list">
                                \${tasks.map(task => \`
                                    <div class="task-item">
                                        <div class="task-info">
                                            <div class="task-type">\${task.type}</div>
                                            <div class="task-desc">\${task.description}</div>
                                        </div>
                                        <div class="task-status status-\${task.status}">\${task.status}</div>
                                        \${task.status === 'running' || task.status === 'queued' ? \`<button class="button secondary" onclick="cancelTask('\${task.task_id}')">Cancel</button>\` : ''}
                                    </div>
                                \`).join('')}
                            </div>
                            <button class="button secondary" onclick="refresh()" style="margin-top: 10px;">Refresh</button>
                        </div>
                    \`;
                }

                function showError(message) {
                    const content = document.getElementById('content');
                    content.innerHTML = \`
                        <div class="error">
                            <strong>Error:</strong> \${message}
                        </div>
                        \${content.innerHTML}
                    \`;
                }
            </script>
        </body>
        </html>`;
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}

let dashboardProvider: AtlasDashboardProvider;

export function registerDashboard(context: vscode.ExtensionContext, atlasAPI: AtlasAPI | undefined) {
    dashboardProvider = new AtlasDashboardProvider(context.extensionUri, atlasAPI);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(AtlasDashboardProvider.viewType, dashboardProvider)
    );

    // Listen for API key changes to update dashboard
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('atlas.apiKey') || e.affectsConfiguration('atlas.apiUrl')) {
                const apiKey = vscode.workspace.getConfiguration('atlas').get('apiKey') as string;
                const apiUrl = vscode.workspace.getConfiguration('atlas').get('apiUrl') as string;

                if (apiKey) {
                    dashboardProvider = new AtlasDashboardProvider(context.extensionUri, new AtlasAPI(apiKey, apiUrl));
                    vscode.window.registerWebviewViewProvider(AtlasDashboardProvider.viewType, dashboardProvider);
                }
            }
        })
    );
}