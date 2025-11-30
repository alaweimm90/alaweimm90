import axios, { AxiosInstance, AxiosResponse } from 'axios';

export interface AtlasTask {
    task_id: string;
    type: 'code_generation' | 'code_review' | 'refactoring' | 'debugging' | 'analysis';
    description: string;
    status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
    agent_id?: string;
    result?: any;
    created_at: string;
    completed_at?: string;
    cost_usd?: number;
    error?: string;
}

export interface AtlasAgent {
    agent_id: string;
    name: string;
    provider: 'anthropic' | 'openai' | 'google';
    model: string;
    status: 'active' | 'inactive';
    capabilities: string[];
    health: {
        status: 'healthy' | 'degraded' | 'offline';
        uptime_percentage: number;
        avg_response_time_ms: number;
    };
    performance: {
        success_rate: number;
        avg_quality_score: number;
        total_tasks_completed: number;
    };
}

export interface AtlasAnalysis {
    analysis_id: string;
    status: 'running' | 'completed' | 'failed';
    repository: string;
    summary?: {
        total_files: number;
        files_analyzed: number;
        avg_chaos_score: number;
        high_chaos_files: number;
        total_opportunities: number;
        estimated_debt_hours: number;
    };
    opportunities?: Array<{
        opportunity_id: string;
        type: string;
        file_path: string;
        description: string;
        impact: any;
        risk: { level: string };
    }>;
}

export interface AtlasMetrics {
    period: string;
    total_tasks: number;
    success_rate: number;
    avg_duration_ms: number;
    total_cost_usd: number;
    agents: Record<string, {
        tasks: number;
        success_rate: number;
        avg_quality: number;
        cost_usd: number;
    }>;
}

export class AtlasAPI {
    private client: AxiosInstance;

    constructor(apiKey: string, baseURL: string = 'https://api.atlas-platform.com/v1') {
        this.client = axios.create({
            baseURL,
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            timeout: 30000
        });

        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401) {
                    throw new Error('Invalid API key');
                } else if (error.response?.status === 429) {
                    throw new Error('Rate limit exceeded');
                } else if (error.response?.data?.error) {
                    throw new Error(error.response.data.error.message);
                }
                throw error;
            }
        );
    }

    // Task Management
    async submitTask(
        type: AtlasTask['type'],
        description: string,
        context?: any,
        requirements?: any
    ): Promise<{ task_id: string; status: string }> {
        const response: AxiosResponse = await this.client.post('/tasks', {
            type,
            description,
            context,
            requirements
        });
        return response.data.data;
    }

    async getTask(taskId: string): Promise<AtlasTask> {
        const response: AxiosResponse = await this.client.get(`/tasks/${taskId}`);
        return response.data.data;
    }

    async listTasks(params?: {
        status?: string;
        type?: string;
        limit?: number;
        offset?: number;
    }): Promise<{ tasks: AtlasTask[]; total: number }> {
        const response: AxiosResponse = await this.client.get('/tasks', { params });
        return response.data.data;
    }

    async cancelTask(taskId: string): Promise<void> {
        await this.client.delete(`/tasks/${taskId}`);
    }

    // Agent Management
    async listAgents(params?: {
        provider?: string;
        capability?: string;
        status?: string;
    }): Promise<{ agents: AtlasAgent[]; total: number }> {
        const response: AxiosResponse = await this.client.get('/agents', { params });
        return response.data.data;
    }

    async getAgent(agentId: string): Promise<AtlasAgent> {
        const response: AxiosResponse = await this.client.get(`/agents/${agentId}`);
        return response.data.data;
    }

    async registerAgent(agent: {
        agent_id: string;
        name: string;
        provider: string;
        model: string;
        capabilities: string[];
    }): Promise<{ agent_id: string; status: string }> {
        const response: AxiosResponse = await this.client.post('/agents', agent);
        return response.data.data;
    }

    // Repository Analysis
    async startAnalysis(options: {
        repository_path?: string;
        repository_url?: string;
        branch?: string;
        analysis_type?: string;
    }): Promise<{ analysis_id: string; status: string }> {
        const response: AxiosResponse = await this.client.post('/analyze', options);
        return response.data.data;
    }

    async getAnalysis(analysisId: string): Promise<AtlasAnalysis> {
        const response: AxiosResponse = await this.client.get(`/analyze/${analysisId}`);
        return response.data.data;
    }

    // Refactoring
    async applyRefactoring(opportunityId: string, options?: {
        dry_run?: boolean;
        create_pr?: boolean;
    }): Promise<{ refactoring_id: string; status: string }> {
        const response: AxiosResponse = await this.client.post('/refactor', {
            opportunity_id: opportunityId,
            ...options
        });
        return response.data.data;
    }

    // Metrics
    async getMetrics(period: string = '24h'): Promise<AtlasMetrics> {
        const response: AxiosResponse = await this.client.get('/metrics', {
            params: { period }
        });
        return response.data.data;
    }

    async getAgentMetrics(agentId: string, period: string = '24h'): Promise<any> {
        const response: AxiosResponse = await this.client.get(`/metrics/agents/${agentId}`, {
            params: { period }
        });
        return response.data.data;
    }

    // System Health
    async getHealth(): Promise<any> {
        const response: AxiosResponse = await this.client.get('/health');
        return response.data.data;
    }

    async getStatus(): Promise<any> {
        const response: AxiosResponse = await this.client.get('/status');
        return response.data.data;
    }
}